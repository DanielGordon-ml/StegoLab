# StegoLab Frontend Plan

Status: researched implementation plan. Research date: 2026-09-23.
Related plans: [Backend](backend.md), [Infrastructure](infrastructure.md), [Execution order](execution_order.md). Backend contracts and model limits are authoritative.

## 1. Product and technical choices

- Build Encode, Decode, Train, and Config for a non-technical, single user. Support the complete first-release workflow, including local and remote dataset preparation.
- Use React, TypeScript strict mode, Vite, TanStack Query, and Recharts in user_interface/. Bundle fonts/icons/assets locally; installed local workflows work offline.
- Serve the browser application and /api/v1 through one origin. EC2 access is private through an SSH tunnel.
- Use TanStack Query for backend snapshots and lists. Use local React state/reducers for forms and temporary state. Derive available buttons from the server's job status and available actions; do not keep conflicting copies of status. [React guidance](https://react.dev/learn/choosing-the-state-structure), [TanStack Query](https://tanstack.com/query/latest/docs/framework/react/overview)
- Keep all committed code files under 300 lines, with small components and readable names. Pin versions in a dependency lockfile.

## 2. Shared contracts and components

- Use backend Pydantic/OpenAPI and per-entity JSON Schemas as the structural source of truth.
- First run a bounded code-generation proof for domain-sized TypeScript clients and runtime validators. Accept generated code only when it satisfies the file-size rule. Do not hand-edit generated output.
- Predetermined fallback if the generator cannot emit compliant modules: small handwritten transport adapters and per-entity TypeScript types, with runtime validation against backend-exported JSON Schemas and CI contract fixtures. Do not create a custom source-code splitter or silently exempt generated code.
- Relevant generation tools are documented by [FastAPI](https://fastapi.tiangolo.com/advanced/generate-clients/) and [Hey API](https://heyapi.dev/docs/openapi/typescript/plugins/zod). The proof gate selects generated or fallback bindings before page implementation.
- Shared components: image picker, transfer progress, image comparison, model/profile selector, password input, queue status, job status, error panel, notifications, metric cards, checkpoint table, and configuration fields.
- Shared records: capabilities, image summary, capacity result, dataset manifest/summary, job snapshot/event, evaluation metrics, checkpoint summary, model manifest, artifact, configuration, and structured error.
- Backend validation is authoritative. Browser validation gives earlier feedback and must use the advertised profile limits.
- Keep plaintext and passwords in temporary memory. Never put them in URLs, local/session storage, analytics, logs, or exported configuration. Clear passwords after submission; interrupted work may require input again.

## 3. Encode

- Show image upload, text input, shared password, active compatible model, and tested encoding profile.
- Accept PNG/JPEG covers through file input and drag/drop. Show filename, dimensions, preparation warnings, and early format/size errors.
- Use TextEncoder to count UTF-8 bytes. Display 'X of Y bytes used' and an optional character count; never treat JavaScript string length as byte capacity. Fetch capacity when image/model/profile changes and ignore stale responses. [TextEncoder](https://developer.mozilla.org/en-US/docs/Web/API/TextEncoder)
- Require a valid image, a compatible model, a password, and text within capacity. Empty messages remain supported if deliberately submitted.
- Freeze submitted inputs. Use a request identifier to prevent duplicate jobs after double-clicks or uncertain responses.
- Show Queued, Processing, Verifying saved image, and Ready phases. Explain GPU waiting without claiming a job has started.
- Display prepared original and encoded output side by side at matching aspect ratios, with fit/actual-size controls and dimensions.
- Download the exact verified PNG bytes returned by the backend. Never redraw the downloadable image through canvas, resize it, or recompress it. Preview scaling affects only the display.
- Use server Content-Disposition filenames. Say 'Download started' rather than claiming the browser has saved the file. [MDN download headers](https://developer.mozilla.org/en-US/docs/Web/HTTP/Reference/Headers/Content-Disposition)
- Explain that the first tested profile requires the PNG to remain unchanged during sharing.

## 4. Decode

- Upload a supported encoded PNG, select the matching decoder/profile, and enter the password.
- Show queue, processing, and needs-input states using backend snapshots.
- Display output only after full backend authentication. Render it as plain text in a selectable read-only field; provide Copy and Clear.
- Retrieve plaintext through the temporary decoded_text endpoint into local component memory, outside the shared query cache. The response is no-store and expires on the server after five minutes. Clear deletes the server result as well as the local text; expired/lost results prompt password re-entry and a new decode attempt.
- Use the recovery-failure wording from backend.md. Keep image-format/size errors separate.
- Never show guessed text, partial unauthenticated text, or a claim that failure proves the image contains no message.
- Clear recovered plaintext on explicit Clear or when leaving the page. Do not restore plaintext after a reload.

## 5. Train and dataset preparation

- Separate dataset preparation, run setup, live training, checkpoints, and run comparison within the Train tab.
- Source choices: Server folder, Upload files/archive, Hugging Face, and HTTPS download. Explain that server folders are visible to the backend; a browser path is not a remote filesystem path.
- Hugging Face form: repository identifier, available configuration, split, revision, and image column. Show access-required errors with setup guidance; Hugging Face tokens remain server-side.
- Before fetching, show estimated bytes and disk availability when known. During work show actual byte progress only when the total is known; otherwise use an activity indicator.
- Source capability controls whether Pause is available. Pause retains partial data where supported. Cancel shows Cancelling/Cleaning until backend confirmation.
- Show a validation summary: accepted/rejected image counts, dimensions, storage, source/revision, split, and optional text-corpus information. Do not call a dataset compatible solely because files downloaded.
- Run modes: Train from scratch, Fine-tune, Resume checkpoint. Show compatibility results before resume.
- Display frozen run settings, epoch/step, current phase, optional ETA, queue state, latest update, latest checkpoint, and remaining pilot budget.
- Show separate charts for exact-message recovery, bit accuracy, and image quality. Charts include readable targets and a table alternative. Keep history bounded and fetch aggregate history for long runs. [Recharts accessibility](https://github.com/recharts/recharts/blob/main/storybook/stories/API/Accessibility.mdx)
- Update a fixed original/encoded preview after validation. Label previews with model/checkpoint and validation step.
- Show Meets target, Needs attention, Below target, and Not measured with both text and color. 'Meets target' does not mean SOTA or undetectable.
- Pause and Stop both wait for a safe save. Pause returns to a resumable paused state; Stop closes the active run and preserves a checkpoint that can seed a new resume job. Export is a separate visible operation.
- Checkpoint table shows recovery/latest/best/pinned status, metrics, compatibility, and storage. Users can pin selected checkpoints and export encoder and decoder packages separately.
- Compare runs only at matching dataset/profile/payload settings, or clearly mark the comparison as unmatched.

## 6. Config

- Expose tested model profiles instead of an unvalidated continuous strength slider.
- Provide compatible model selection, CPU/GPU availability, batch-size presets, checkpoint interval, dataset/cache information, and defaults for new runs.
- Validate changes before applying. Show capacity changes and memory estimates without promising OOM prevention.
- Running jobs keep frozen settings. Display this separately from defaults for future jobs.
- Support versioned JSON import/export and Reset to defaults. Validate before replacing active settings.
- Export portable model/profile identifiers and settings. Flag unavailable model references on import. Exclude passwords, access tokens, and host-specific secrets.

## 7. Progress, recovery, and error handling

- Use one application-level SSE connection. Commands use normal HTTP requests.
- Track event identifiers, deduplicate replayed events, reconnect automatically, and refresh snapshots when event history is unavailable. [MDN SSE](https://developer.mozilla.org/en-US/docs/Web/API/Server-sent_events/Using_server-sent_events)
- When progress events stop, show connection status and poll snapshots while reconnecting. A broken connection does not imply backend failure.
- Refresh restores active jobs without creating duplicates. Backend restart produces interrupted training or needs-input inference, not false completion.
- Separate browser-upload bytes from server-download bytes and training progress. Use an XMLHttpRequest adapter for actual upload progress. [MDN file handling](https://developer.mozilla.org/en-US/docs/Web/API/File_API/Using_files_from_web_applications)
- Use object URLs for local previews and revoke them on replacement/unmount.
- Display structured plain-English errors and a diagnostic reference. Never expose raw tracebacks or secret-bearing logs.
- Polling fallback: every 5 seconds while disconnected and an active job exists; stop polling on reconnection or no active jobs. Keep one timer per application.

## 8. Accessibility and verification

- Target WCAG 2.2 AA: keyboard-operable tabs/forms, visible focus, labels, readable contrast, text with status colors, and restrained status announcements. [W3C tabs](https://www.w3.org/WAI/ARIA/apg/patterns/tabs/), [WCAG reference](https://www.w3.org/WAI/WCAG22/quickref/)
- Unit/component tests: multilingual byte boundaries, stale capacity requests, form validation, action availability, event deduplication, needs-input recovery, secret clearing, and configuration validation.
- Contract checks: schema/client drift and runtime validation of snapshots/events/errors.
- Browser tests: upload → encode → verified PNG download → decode exact original; wrong password; ordinary/damaged image; missing model; busy GPU; duplicate submission; reconnect/reload; dataset pause/cancel/cache; checkpoint save/resume/export.
- Inspect the downloaded PNG's dimensions and bytes, not only its screenshot. Use Playwright retrying assertions and capture actual files. [Uploads](https://playwright.dev/docs/input), [Downloads](https://playwright.dev/docs/downloads), [Assertions](https://playwright.dev/docs/test-assertions)
- Run automated accessibility checks and manual keyboard/screen-reader checks. Automated checks alone are not full accessibility proof. [Playwright accessibility](https://playwright.dev/docs/accessibility-testing)
- Delivery order: contracts and shell → Encode/Decode → datasets and Train → Config → recovery/accessibility verification.
