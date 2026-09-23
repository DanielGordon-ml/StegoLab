# Sprint 1 — Working Local Foundation

Status: implemented and locally verified. Planned duration: ten working days.
Owner: project owner + Codex. Date: 2026-09-23.

## Goal

Start StegoLab locally, open four accessible tabs, save checkpoint frequency,
restart the backend, reload the saved value, and reset to five minutes.

This implements foundation work from [execution order](execution_order.md).
Encoding, decoding, training, dataset downloads, GPU use, and cloud deployment
remain outside this sprint. There is no model-quality or SOTA claim.

## Backlog

| Work | Planned days | Delivery evidence |
|---|---|---|
| Project packages, locks, containers, initial CI | 1–2 | Complete; clean checkout and new volume start successfully |
| Strict contracts, settings, metadata storage, CLI | 3–4 | Complete; 36 backend tests pass |
| Frontend binding proof, four tabs, Config | 5–6 | Complete; 10 frontend tests pass |
| Failure handling, integration, guides, Archify map | 7–8 | Complete; guides and validated interactive map delivered |
| Acceptance checks, fixes, working demonstration | 9–10 | Complete; real browser restart workflow passes |

## Decisions

- FastAPI/Pydantic, React/TypeScript/Vite, TanStack Query, and SQLite.
- Two non-root CPU containers; only `127.0.0.1:8080` is published.
- The backend validates settings and saves them atomically. Repeated mutation
  identifiers return the original result; conflicting reuse returns an error.
- Checkpoint frequency accepts positive whole minutes, stored as seconds.
  The default is 300 seconds. CPU and no-model information are read-only.
- Shared schemas are authoritative. Browser validation provides early feedback.
- Capabilities advertise no installed models and zero usable payload capacity.
- Job records and event contracts exist; worker execution and live events do not.
- Separate feature branches isolate backend, interface, and infrastructure work.
- The existing `CLAUDE.md` and `AGENT.md` files remain unchanged.

## Acceptance record

| Check | Result |
|---|---|
| Backend unit and API tests | Passed: 36 tests |
| Backend formatting, lint, and types | Passed: Ruff and mypy |
| Frontend component and contract tests | Passed: 10 tests, including delayed-read/save race and uncertain retry |
| Frontend lint, types, and production build | Passed: TypeScript, ESLint, Prettier, Vite |
| OpenAPI / entity schema drift check | Passed: exported contracts match source |
| Source files below 300 lines | Passed: all 50 application/test/tool source files |
| CPU container build and healthy startup | Passed on Linux arm64 through Docker on Apple Silicon |
| Browser save → backend restart → reload → reset | Passed: one Playwright test with actual Compose restart |
| Invalid input, duplicate/conflicting requests, safe errors | Passed: strict validation, concurrent retries, operation conflicts, no private error leakage |
| Failed writes preserve state; corrupt saved state fails clearly | Passed: transaction rollback, missing fields, malformed settings, unsupported database version |
| Job metadata persists after restart | Passed: validated synthetic job records |
| Archify artifact checks, browser evidence, and visual review | Passed: 9/9 checks; four viewport sizes; light/dark review |

### Environment and evidence

- Application source verified at `ff6c201` on `codex/sprint-01-foundation`.
  Subsequent acceptance-document updates do not change application behavior.
- Backend container: Python 3.12.14; frontend build container: Node.js 24.
  Dependency locks and multi-platform image digests are committed.
- Local test tools: Python 3.12.13, Node.js 26.5.0, Docker 29.6.2.
  The frontend production build also passed inside its pinned Node.js 24 image.
- A separate clean Git worktree and new disposable volume built and started on
  port 8082. Health, default configuration, empty models, and empty jobs matched
  their contracts. Its containers and test volume were removed afterward.
- The working app remains at [localhost:8081](http://127.0.0.1:8081), because an
  unrelated project already occupies port 8080. That project was left running.
- Both containers run as non-root with read-only root filesystems. Only the
  frontend has a published host port; the backend has none.
- The browser test found no page errors or automated accessibility violations.
  Desktop and 390-pixel-wide mobile screenshots were visually reviewed; the
  mobile page has no horizontal overflow. This is not a full accessibility audit.
- [Binding proof](../docs/client_binding_proof.md): generated clients exceeded
  the code-length limit. Small typed adapters interpret the backend's schemas
  without relaxing the browser security policy.
- [Archify delivery record](../docs/architecture.md) contains checksums and the
  separate artifact, browser, and visual-review evidence.
- The CI workflow is configured, and its relevant checks passed locally. No
  GitHub Actions run or cloud deployment was triggered in this implementation.
- Non-blocking dependency notices remain: Starlette recommends `httpx2` for its
  test client; npm reports deprecated ESLint 9 and a transitive encoding package.
  These did not fail checks; the installed frontend dependency audit reported
  zero vulnerabilities.
- No GPU hours were used, and no models were trained or evaluated.

## Follow-on sprint

Implement the versioned message/image protocol with known-answer fixtures before
model training. Define framing, authenticated encryption, correction overhead,
capacity accounting, image preparation, and saved-PNG round trips together.
