# Sprint 1 — Working Local Foundation

Status: implementation in progress. Planned duration: ten working days.
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
| Project packages, locks, containers, initial CI | 1–2 | Pending integration |
| Strict contracts, settings, metadata storage, CLI | 3–4 | Pending integration |
| Frontend binding proof, four tabs, Config | 5–6 | Pending integration |
| Failure handling, integration, guides, Archify map | 7–8 | Pending verification |
| Acceptance checks, fixes, working demonstration | 9–10 | Pending verification |

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
| Backend unit and API tests | Pending |
| Backend formatting, lint, and types | Pending |
| Frontend component and contract tests | Pending |
| Frontend lint, types, and production build | Pending |
| OpenAPI / entity schema drift check | Pending |
| Source files below 300 lines | Pending |
| CPU container build and healthy startup | Pending |
| Browser save → backend restart → reload → reset | Pending |
| Invalid input, duplicate/conflicting requests, safe errors | Pending |
| Failed writes preserve state; corrupt saved state fails clearly | Pending |
| Job metadata persists after restart | Pending |
| Archify artifact checks, browser evidence, and visual review | Pending |

## Follow-on sprint

Implement the versioned message/image protocol with known-answer fixtures before
model training. Define framing, authenticated encryption, correction overhead,
capacity accounting, image preparation, and saved-PNG round trips together.
