# Interactive architecture map

Open the locally generated [StegoLab map](architecture.html) in a browser.
It supports search, component focus, theme switching, guided views, and export.
Solid paths show the Sprint 1 settings flow. Dashed paths show future work.

The reviewed source is [architecture.architecture.json](architecture.architecture.json).
Generated HTML and screenshots are local artifacts, excluded from source control.
The map represents the two-container design and must be updated when deployment
boundaries or persistence change.

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

## Delivery record — 2026-09-23

- Diagram type: architecture.
- Archify version: 2.17.
- Specification: 3,775 bytes; SHA-256
  `242722c7884b78a6b8f383083741e294adda773221d195c87a3e99e4587c7497`.
- Artifact: 804,906 bytes; SHA-256
  `be273c8a6197ad4798f8150c3ba2ec0c9c4d6a91f1cf928944db4c933fcfac8c`.
- Validation: 9/9 showcase checks; zero errors and zero warnings.
- Automated browser evidence: passed at 1440×900, 1600×1000, 1920×1080,
  and 2048×1320; no page overflow.
- Visual review: passed after inspecting all four light/dark screenshots at
  1440×900 and 2048×1320. Labels, relationships, cards, and the future-work
  distinction are readable; no visible crossings or clipping.
- Geometry correction rounds: one, to specify the backend-to-storage direction.
- Browser receipt and screenshots: local `architecture.visual-check.*` sidecars.

Artifact validation, automated browser checks, and visual review are separate
results. None is evidence of model quality or successful training.
