# Protocol fixture provenance

The frozen examples in `backend_service/fixtures/protocol_v1.json` are public test
data. Their passwords, keys, salts, and nonces are intentionally visible. Never use
them to protect private data. The application must generate fresh salt and nonce
bytes when packaging a real message.

## Independent construction

`scripts/generate_protocol_fixtures.py` was written from the protocol description
without importing the application or `reedsolo`. It constructs the header, padded
plaintext, authenticated context, interleaved blocks, and payload map directly.

- Key: Argon2id v1.3, 32 output bytes, 67,108,864 memory bytes, three operations.
- Encryption: XChaCha20-Poly1305-IETF through low-level `nacl.bindings` calls.
- Correction: direct polynomial division over the eight-bit field with polynomial
  `0x11d`; generator `2`; roots `2^0` through `2^31`; 223 data bytes and 32 parity
  bytes per block. No codec library or lookup tables are used in the generator.
- Interleaving: all block bytes at column zero, then all at column one, through
  column 254. Bits are read from the most-significant bit first.
- Map checksum: SHA-256 of exactly `width × height` bytes. Each byte holds one
  repeated bit as integer zero or one, in row order. No shape or array header is
  included in the checksum.

The context is the ASCII text `public_fixture_v1`. Authenticated bytes are the
domain `StegoLab/message_protocol/v1` followed by one zero byte, the 45-byte header,
two-byte big-endian context length, context, four-byte big-endian width, and
four-byte big-endian height. The header stores `SGLB`, version byte `1`, salt bytes
`0..15`, and nonce bytes `0..23`.

These are independent framing and correction examples, not an independent
cryptographic implementation. Generation and the cross-check both use the same
underlying libsodium cryptographic primitives.

## Inputs and review

| Example | Dimensions | Message bytes | Frame bytes | Protected bytes |
|---|---:|---:|---:|---:|
| Empty text | 512 × 512 | 0 | 446 | 510 |
| Multilingual text | 513 × 517 | 38 | 446 | 510 |
| Maximum small message | 512 × 512 | 256 | 446 | 510 |
| Maximum message | 1024 × 1024 | 1024 | 1115 | 1275 |

The multilingual example includes Hebrew, an emoji, Chinese characters, newlines,
a decomposed accent, and leading/trailing spaces. Its bytes must remain unchanged.
The public password also contains Hebrew and an emoji. The two maximum examples
contain repeated ASCII `A` and `Z`, respectively.

The examples were generated and cross-checked on 2026-09-23 with PyNaCl 1.6.2,
reedsolo 1.7.0, and NumPy 2.5.3. The separate cross-check:

1. Derived each key through `nacl.pwhash.argon2id.kdf` and compared every byte.
2. Used `nacl.secret.Aead.decrypt` on the frozen frame; checked its declared length,
   exact message bytes, and all zero-padding bytes.
3. Encoded each frame block with `reedsolo.RSCodec(nsym=32, nsize=255, fcr=0,
   prim=0x11d, generator=2, c_exp=8)` and compared every interleaved protected byte.
4. Built maps through NumPy `unpackbits(bitorder="big")` and `resize`; compared
   each full-map checksum with the independently generated checksum.

All four examples passed. Fixture file SHA-256:
`cceb8ff19f20c9026f3f526452b7eccea8f4aa616a978a05662fd4137143d3fd`.
The checked-in bytes are the test oracle; tests must read them, not regenerate
them from application code. This checks protocol compatibility, not secrecy in
images or learned-model quality.

## Regeneration

Run from the repository root after `uv sync --locked`:

```sh
fixture_directory=$(mktemp -d)
uv run --locked python scripts/generate_protocol_fixtures.py \
  --output "$fixture_directory/protocol_v1.json"
cmp backend_service/fixtures/protocol_v1.json "$fixture_directory/protocol_v1.json"
```

The script requires an explicit output path and rejects an existing file. It does
not update checked-in fixtures automatically. Any changed expected bytes need a
review of the specification and independent cross-check before replacement.

## References

- [PyNaCl password hashing and key derivation](https://pynacl.readthedocs.io/en/latest/password_hashing/)
- [PyNaCl authenticated secret-key encryption](https://pynacl.readthedocs.io/en/latest/secret/)
- [Reed-Solomon codec parameters and correction limits](https://github.com/tomerfiliba-org/reedsolomon)
