# Interactive architecture map

Open the locally generated [StegoLab map](architecture.html) in a browser.
It supports search, component focus, theme switching, guided views, and export.
Solid paths show the existing settings flow and Sprint 2 local tools.
Dashed paths show future work.

The reviewed source is [architecture.architecture.json](architecture.architecture.json).
Generated HTML and screenshots are local artifacts, excluded from source control.
The map represents the two-container design and must be updated when deployment
boundaries or persistence change.

## Sprint 2 services

- The frontend and backend remain separate containers. Only the frontend port
  is published, and settings still use the persistent SQLite volume.
- Local commands call reusable Python services directly. They add no HTTP
  routes, database migration, worker process, or model package.
- `verify_protocol` runs bundled public examples through message framing,
  authenticated encryption, error correction, and the test payload map.
- `inspect_image` checks a complete input image. `prepare_image` applies the
  supported orientation/color rules and saves RGB or RGBA PNG pixels.
- The reusable PNG reader preserves stored integer pixels. It applies no
  orientation or color conversion during recovery.
- Public capacity remains zero. Training, model inference, and model exports
  remain planned. The test payload map does not prove image hiding quality.

Implementation evidence was checked in the backend application and routes,
message services, image preparation/output services, and Compose configuration.
The diagram groups message and image modules into one service box; that box is
part of the backend Python package, not another container.

## Reproduce

Install the [Archify skill](https://github.com/tt-a1i/archify), then replace
`/path/to/archify` with its installed location:

```sh
node /path/to/archify/bin/archify.mjs validate architecture docs/architecture.architecture.json --quality showcase --json
node /path/to/archify/bin/archify.mjs deliver architecture docs/architecture.architecture.json docs/architecture.html --quality showcase --json
node /path/to/archify/bin/archify.mjs visual-check docs/architecture.html --json
```

If Chrome is not found, set `ARCHIFY_CHROME` to a Chrome or Chromium executable
and rerun the last command. The application itself does not depend on Archify.

## Sprint 2 delivery record — 2026-09-23

- Diagram type: architecture.
- Archify version: 2.17.
- Specification: 4,950 bytes; SHA-256
  `b3db5876991d81f6ccdc17a9e855047a1698d1d044b10e81500b566ff5017039`.
- Artifact: 808,592 bytes; SHA-256
  `99d83e309aba199579d6f0fa7c81f59b57b438c022e00c746703cce0029b06d8`.
- Validation: 9/9 showcase checks; zero errors and zero warnings.
- Automated browser evidence: passed at 1440×900, 1600×1000, 1920×1080,
  and 2048×1320; no page overflow.
- Visual review: passed after inspecting all four light/dark screenshots at
  1440×900 and 2048×1320. Labels, relationships, cards, and the future-work
  distinction are readable; no visible crossings or clipping.
- Geometry corrections: the planned-jobs path uses right-side ports and a
  diagnosed label position to avoid the persistent volume.
- Browser correction rounds: one, to reduce excess top/row spacing and contain
  the diagram and cards without changing typography or clipping content.
- Browser receipt and screenshots: local `architecture.visual-check.*` sidecars.
- Browser runtime: existing Playwright Chromium headless shell, selected with
  `ARCHIFY_CHROME`. The initial default lookup was unavailable; the final run
  completed all required measurements and screenshots.

Artifact validation, automated browser checks, and visual review are separate
results. None is evidence of model quality or successful training.
