# Full system architecture map

Read [Architecture.md](../Architecture.md) for the full system design and the
[README overview](../README.md#system-architecture) for its embedded diagram.
Open the generated [interactive map](architecture.html) locally for search,
component focus, relationship tracing, theme switching, and export.
GitHub displays the [static SVG](architecture.svg); it does not run this HTML viewer.

## Scope and evidence

- Solid paths show implemented settings, protocol, image tools, and local
  UHD-IQA dataset preparation with frozen manifests.
  Dashed paths and components labelled **planned** show the target system.
- The target keeps two application containers. Protocol/image/dataset modules
  and future model/data worker processes belong to the backend. Local dataset
  commands run synchronously; they do not start background workers.
- The API will own scheduling and job transitions for both kinds of worker.
  The map shows the main paths; detailed control, return, event, and file flows
  are described in `Architecture.md` rather than repeated as crossing arrows.
- The local dataset path preserves official source splits, verifies prepared
  copies and checksums, and works offline after the source is unmounted.
  Remote downloads, shared cache, browser uploads, models, workers, training,
  checkpoints, independent model packages, SSE, CUDA, and EC2 remain planned.
- Public capacity remains zero. Protocol tests and PNG pixel preservation do
  not establish actual image hiding, model quality, or detection resistance.
- Evidence: application/routes/storage, frontend components/contracts, protocol,
  image and dataset modules, schemas, Docker/Compose, CI/tests, and delivery
  plans. This map includes the Sprint 3 working-tree implementation on the
  `abcef42` application baseline; it does not imply a new committed revision.
  `Architecture.md` records how local draft notes differ from the approved plans.

The source is [architecture.architecture.json](architecture.architecture.json).
Keep this specification, the static SVG, and the architecture documents in source
control. The HTML viewer, delivery receipts, and screenshots are generated local
artifacts, excluded by `.gitignore`. Regenerate them after a fresh checkout.

## Reproduce

Use [Archify](https://github.com/tt-a1i/archify), version 2.17 for the receipt below.
Replace `/path/to/archify` with the installed skill location. Run from the project
root, preserving this order so the SVG comes from a successfully delivered HTML:

```sh
node /path/to/archify/bin/archify.mjs validate architecture docs/architecture.architecture.json --quality showcase --json
node /path/to/archify/bin/archify.mjs deliver architecture docs/architecture.architecture.json docs/architecture.html --quality showcase --json > docs/architecture.delivery.json
node /path/to/archify/bin/archify.mjs visual-check docs/architecture.html --json
python3 scripts/export_architecture_svg.py
```

Each command must succeed before continuing. If Chrome is unavailable, set
`ARCHIFY_CHROME` to an existing Chrome/Chromium executable and rerun `visual-check`.
Then inspect its light/dark screenshots and the generated README image.
The application itself does not depend on Archify or a browser renderer.

The small [SVG exporter](../scripts/export_architecture_svg.py) verifies both
specification and HTML hashes against the delivery receipt. It copies the checked
SVG geometry, embedded font, and static light-theme styles without executing
viewer code. The README image includes full node tags, excludes viewer controls,
and uses a white background in either GitHub theme. It is a derived documentation
asset, not an additional Archify validation receipt. Review the extractor if the
Archify template changes; do not hand-edit generated SVG geometry.

## Delivery record — Sprint 3 local dataset path, 2026-09-23

| Evidence | Result |
|---|---|
| Diagram type | Architecture; Archify 2.17; static Classic preset |
| Deterministic artifact validation | 9/9 showcase checks; zero errors and warnings |
| Automated browser evidence | Passed: 1440×900, 1600×1000, 1920×1080, 2048×1320; no horizontal or vertical page overflow |
| Perceptual visual review | Passed: all four light/dark screenshots at 1440×900 and 2048×1320 inspected |
| README image review | Passed: SVG rasterized with Sharp and inspected for labels, arrows, tags, and clipping |
| Browser/visual correction rounds | 0; the delivered composition passed its first browser review |

- Specification: 6,647 bytes; SHA-256
  `a1a4787da741a228fee8a6a87e30df88955ee3470280fc4a250d41f7a4121e52`.
- Checked HTML: 818,683 bytes; SHA-256
  `fe434f644bdadd40cd922e002716afa34e574cdf82504ac98bed80d8dceeb3ac`.
- Static SVG: SHA-256
  `36f698877fecdc48b7c3c8724e52209036c86bbd200c871d46878e95a2130975`.
- Local deterministic receipt: `architecture.delivery.json`.
- Automated browser receipt: `architecture.visual-check.json`; its
  `visualReview: pending` field intentionally makes no perceptual claim.
- Separate image-review record: `architecture.review.json`.
- Browser engine: existing Playwright Chromium headless shell through
  `ARCHIFY_CHROME`. The packaged `visual-check` completed all measurements and
  captures before image review.
- The existing 12-node geometry and routing were preserved. Dataset labels and
  implemented path styles changed; validation and browser review passed on the
  first candidate without geometry corrections.

This record replaces the earlier full-system map record. Historical sprint
acceptance remains in Git history. Artifact validation, browser evidence, and
visual review are separate checks; none proves model or training quality.
