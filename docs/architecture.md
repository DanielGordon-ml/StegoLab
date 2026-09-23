# Full system architecture map

Read [Architecture.md](../Architecture.md) for the full system design and the
[README overview](../README.md#system-architecture) for its embedded diagram.
Open the generated [interactive map](architecture.html) locally for search,
component focus, relationship tracing, theme switching, and export.
GitHub displays the [static SVG](architecture.svg); it does not run this HTML viewer.

## Scope and evidence

- Solid paths show implemented settings, protocol, and image-tool behavior.
  Dashed paths and components labelled **planned** show the target system.
- The target keeps two application containers. Protocol/image modules and future
  model/data worker processes belong to the backend, not separate containers.
- The API will own scheduling and job transitions for both kinds of worker.
  The map shows the main paths; detailed control, return, event, and file flows
  are described in `Architecture.md` rather than repeated as crossing arrows.
- Dataset sources, validated data/cache, training/evaluation, checkpoints, and
  independent encoder/decoder packages are included. Models, workers, downloads,
  SSE, CUDA, and EC2 operations are still planned.
- Public capacity remains zero. Protocol tests and PNG pixel preservation do
  not establish actual image hiding, model quality, or detection resistance.
- Evidence: application/routes/storage, frontend components/contracts, protocol
  and image modules, schemas, Docker/Compose, CI/tests, and all nine local plan
  documents. Application baseline: `6fe874010f0c03d71b55f236c9e22723c01f8a9b`.
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

## Delivery record — full-system overview, 2026-09-23

| Evidence | Result |
|---|---|
| Diagram type | Architecture; Archify 2.17; static Classic preset |
| Deterministic artifact validation | 9/9 showcase checks; zero errors and warnings |
| Automated browser evidence | Passed: 1440×900, 1600×1000, 1920×1080, 2048×1320; no horizontal or vertical page overflow |
| Perceptual visual review | Passed: all four light/dark screenshots at 1440×900 and 2048×1320 inspected |
| README image review | Passed: SVG rasterized with Sharp and inspected for labels, arrows, tags, and clipping |
| Browser/visual correction rounds | 0; the delivered composition passed its first browser review |

- Specification: 6,653 bytes; SHA-256
  `892df6b8b84ab9867169351838155efc4d169748450c5d0ff6775bb1e8d92368`.
- Checked HTML: 818,705 bytes; SHA-256
  `0ff70cacd086612624eb4fb47fc4f352df26713e54b18a25d6a1af2ecf88f973`.
- Static SVG: SHA-256
  `256814406024d33850de769b4235f2e4dc333f59cbd7b161e13deb8d15472f8b`.
- Local deterministic receipt: `architecture.delivery.json`.
- Automated browser receipt: `architecture.visual-check.json`; its
  `visualReview: pending` field intentionally makes no perceptual claim.
- Separate image-review record: `architecture.review.json`.
- Browser engine: existing Playwright Chromium headless shell through
  `ARCHIFY_CHROME`. The packaged `visual-check` completed all measurements and
  captures before image review.
- Before delivery, the validator diagnosed a job-edge label overlap and a
  desktop-width issue. A label position and viewBox width correction resolved
  them without removing meaning or shrinking typography.

This record replaces the narrower Sprint 2 map record. Historical sprint
acceptance remains in Git history. Artifact validation, browser evidence, and
visual review are separate checks; none proves model or training quality.
