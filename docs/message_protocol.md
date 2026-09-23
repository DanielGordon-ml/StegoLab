# Message protocol version 1

Sprint 2 implements a controlled byte/bit channel. It does not hide a message in
an image, train a model, or prove resistance to image changes or steganalysis.
The only supported layout profile is `test_only_v1`. It never appears in the
public model/profile list, and public usable payload capacity remains zero.

## Run the public demonstration

```sh
uv sync --frozen
uv run stegolab verify_protocol
```

The command accepts no messages, passwords, or fixture paths. It reports safe
JSON with the number of public vectors and passed checks. Exit codes are zero
for success, one for a failed operation, and two for invalid/unavailable commands.
The examples are packaged with the application and also work without network
access. Their fixed salts and nonces are **public tests only**.

## Services and trusted context

- `encode_message(message, password, context)` returns protected frame bytes.
- `decode_message(protected, password, context)` returns authenticated text.
- `create_payload_map(protected, context)` returns a NumPy `uint8` array.
- `recover_payload_bytes(values, context)` accepts hard bits (`uint8`) or finite
  logits (`float32`) and returns protected bytes.
- `calculate_capacity(context)` returns a strict `PayloadCapacity` record.

These functions are in the message protocol, payload map, and payload capacity
modules under `backend_service`. `ProtocolContext` is an immutable strict record
under `schemas`. It carries integer width/height, protocol version `1`, profile
`test_only_v1`, and a trusted compatibility identifier. The identifier is 1–128
ASCII characters matching `[a-z0-9][a-z0-9_.-]{0,127}`; the public examples use
`public_fixture_v1`. A future caller obtains this value from its trusted model
package, never from recovered bytes. There is no model registry in this sprint.

Context is revalidated at service boundaries, including records constructed
without validation. Width and height are each 512–4096 and their product is at
most 8,850,000. Application, database, configuration, preprocessing, and protocol
versions are independent. A wire-rule change requires a new protocol/profile and
new fixed examples; dependency upgrades must preserve the current examples.

## Capacity and frame bytes

Let `C = min(1024, floor(width × height / 1024))` and
`B = ceil((C + 63) / 223)`. Reject a layout if its protected bits do not fit.
All messages at the same dimensions have the same frame length.

| Offset | Length | Content |
|---|---|---|
| 0 | 4 | ASCII `SGLB` |
| 4 | 1 | Unsigned protocol version `1` |
| 5 | 16 | Fresh random Argon2id salt |
| 21 | 24 | Fresh random XChaCha20 nonce |
| 45 | `223B − 61` | Encrypted body |
| `223B − 16` | 16 | Authentication tag |

Before encryption, the body is a two-byte **big-endian** message-byte length,
the exact UTF-8 message bytes, and zero bytes to fill `223B − 61` bytes.
The frame has exactly `223B` bytes; no external block padding is needed.

Messages preserve spaces, line endings, combining characters, Hebrew, emoji,
and embedded zero characters. Empty messages are valid. Invalid Unicode and
byte overflow fail before key derivation; there is no truncation or normalization.
Passwords also use exact strict UTF-8. Our input policy permits 1–1,024 encoded
bytes, without trimming, normalization, or case conversion.

| Dimensions | Maximum user bytes | Frame bytes | Correction bytes | Protected bits |
|---|---:|---:|---:|---:|
| 512 × 512 | 256 | 446 | 64 | 4,080 |
| 513 × 517 | 259 | 446 | 64 | 4,080 |
| 1024 × 1024 | 1,024 | 1,115 | 160 | 10,200 |
| 3840 × 2160 | 1,024 | 1,115 | 160 | 10,200 |
| 4096 × 2160 | 1,024 | 1,115 | 160 | 10,200 |

The record also reports zero padding at maximum input, total map bits, repeated
bits, minimum repetitions, and the number of bits receiving one extra repetition.
At 1024 × 1024, net capacity is 0.0078125 user bits per pixel while the map still
uses one channel bit per pixel. These numbers describe layout fit, not learned
model performance. A 4096 × 4096 image exceeds the area limit.

## Encryption and authenticated context

Derive a 32-byte key with explicit Argon2id, operation limit 3 and memory limit
67,108,864 bytes. Use PyNaCl `Aead` (XChaCha20-Poly1305-IETF). Generate salt and
nonce independently with operating-system randomness for every new frame.
There is no production deterministic mode or caller-selected cost setting.

The associated data is this exact concatenation:

```text
ASCII "StegoLab/message_protocol/v1" + one zero byte
45-byte header
two-byte big-endian compatibility identifier byte length
ASCII compatibility identifier
four-byte big-endian width
four-byte big-endian height
```

Store `Aead.encrypt(...).ciphertext`, which includes the tag but excludes the
nonce already in the header. Corrected input must have the exact expected frame
length and header before key derivation. Authenticate the entire ciphertext and
context, then require length ≤ `C`, all remaining padding zero, and strict UTF-8.

Sources: [PyNaCl key derivation](https://pynacl.readthedocs.io/en/latest/password_hashing/)
and [authenticated encryption](https://pynacl.readthedocs.io/en/latest/secret/).

## Correction, interleaving, and spatial map

- Split the complete frame into `B` blocks of 223 bytes. Each encoded block
  appends 32 parity bytes, producing 255-byte codewords.
- Explicit codec parameters: `nsym=32`, `nsize=255`, `prim=0x11d`, `generator=2`,
  `fcr=0`, `c_exp=8`. No erasure hints are used.
- Interleave columns: byte zero of every codeword in block order, then byte one
  of every codeword, through byte 254. Expand each byte most-significant bit first.
- If there are `L = 255B × 8` protected bits, flattened map position `i` stores
  protected bit `i mod L`. Shape is `[1, height, width]`, row-major. All positions
  are used; the last repetition can be partial.
- On recovery, sum each protected bit's logits in float64. Hard bits are
  equivalent to −1/+1 votes. A positive sum means one; zero or a negative sum
  means zero. Wrong shapes, unsupported dtypes, nonbinary hard values, and
  non-finite logits are rejected without implicit conversion.
- Pack recovered bits MSB-first, undo interleaving, correct each block, then
  authenticate the recovered frame. Neither original cover nor encoder weights
  nor a per-message side file participates.

Reed–Solomon corrects up to 16 unknown damaged **byte symbols per codeword**.
Beyond that bound it can fail or return incorrect data; only authenticated exact
recovery is acceptable. [Codec correction limits](https://github.com/tomerfiliba-org/reedsolomon)

## Failure and privacy contract

Missing, damaged, and unauthenticated messages use the same failure:

> No valid hidden message could be recovered. Check the password, model, and image.

Invalid source formats, unsupported context, bad password input, overflow, and
resource failures have separate safe errors. No partial or guessed text returns.
Protocol services write no files, logs, database records, metadata, or environment
variables. Messages, passwords, and keys remain in local memory only. Python
cannot guarantee memory erasure; scope exit releases references without such a claim.

See [fixture provenance](protocol_fixture_provenance.md) for the independent
generator and committed expected bytes. Runtime verification reads those bytes;
it never replaces them with values produced by the implementation under test.
