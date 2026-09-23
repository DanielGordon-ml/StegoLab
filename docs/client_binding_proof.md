# Frontend binding proof

Date: 2026-09-23. The bounded proof completed in less than one hour.

## Decision

Use small handwritten TypeScript adapters and types with runtime validation against the backend's exported, per-entity Draft 2020-12 JSON Schemas. The browser imports those files directly, so it keeps no second schema definition. Backend export checks catch drift; frontend contract tests exercise valid and invalid response shapes and nested requests.

## Observed generation results

Following the [FastAPI client guide](https://fastapi.tiangolo.com/advanced/generate-clients/) and [Hey API runtime-validation plugin](https://heyapi.dev/docs/openapi/typescript/plugins/zod):

- `npx @hey-api/openapi-ts@0.99.0 --help` failed on this machine with its resolved TypeScript 7 dependency (`SyntaxKind.AnyKeyword` was undefined).
- An isolated install using TypeScript 5.9.3 succeeded. The proof used the actual Sprint 1 `contracts/openapi.json` with `@hey-api/typescript`, `@hey-api/sdk`, `@hey-api/client-fetch`, and `zod` plugins.
- Measured output: `types.gen.ts` 545 lines; `client/utils.gen.ts` 316 lines; `zod.gen.ts` 187 lines. Two generated code files exceeded the strict limit of fewer than 300 lines. No generated source was split, shortened, or exempted.

Reproduce in a temporary folder, outside the repository:

```sh
npm install --prefix /tmp/stegolab-binding-proof @hey-api/openapi-ts@0.99.0 typescript@5.9.3 zod@4.1.13
/tmp/stegolab-binding-proof/node_modules/.bin/openapi-ts \
  -i "$PWD/contracts/openapi.json" \
  -o /tmp/stegolab-binding-proof/generated \
  -p @hey-api/typescript @hey-api/sdk @hey-api/client-fetch zod --no-log-file
```

## Runtime validation and browser policy

Ajv runtime compilation uses code generation, which conflicts with the browser policy's restriction on `unsafe-eval`. A build-time Ajv standalone proof produced formatted modules of 383 lines for ConfigurationUpdate, 510 for Capabilities, and 310 for ErrorEnvelope, so it also failed the size rule.

The selected fallback uses pinned `@cfworker/json-schema` 4.1.1. Its [upstream documentation](https://github.com/cfworker/cfworker/tree/main/packages/json-schema) confirms Draft 2020-12 support and interpreted validation without `eval` or `new Function`. This keeps the strict browser policy. No schema-derived source generation is needed.

Server errors are validated before display; malformed responses become a safe generic error. Outgoing configuration and reset requests are also validated. A network failure keeps the submitted setting and request identifier for a safe retry. Input values are never logged or included in transport errors.

## Verification

`npm --prefix user_interface run test` checks missing fields, wrong types, unknown fields, whole-minute constraints, nested request validation, safe errors, keyboard tabs, pending/success states, and exact retry reuse. The browser test uses real backend requests, checks accessibility, and enables a real Compose backend restart when `STEGOLAB_RESTART_BACKEND=1`.
