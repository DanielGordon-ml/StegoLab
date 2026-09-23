"""Generate public protocol examples independently of application code."""

import argparse
import hashlib
import json
from pathlib import Path

from nacl.bindings import (
    crypto_aead_xchacha20poly1305_ietf_encrypt,
    crypto_pwhash_alg,
    crypto_pwhash_ALG_ARGON2ID13,
)

WARNING = "PUBLIC TESTS ONLY: never reuse these passwords, salts, or nonces."
DOMAIN = b"StegoLab/message_protocol/v1\x00"
COMPATIBILITY_IDENTIFIER = "public_fixture_v1"


def multiply_field(left: int, right: int) -> int:
    """Multiply two bytes using polynomial arithmetic over the chosen field."""
    product = 0
    while right:
        if right & 1:
            product ^= left
        right >>= 1
        left <<= 1
        if left & 0x100:
            left ^= 0x11D
    return product


def generator_coefficients() -> list[int]:
    """Build the polynomial with roots 2 to powers zero through thirty-one."""
    coefficients = [1]
    root = 1
    for _ in range(32):
        expanded = [0] * (len(coefficients) + 1)
        for index, value in enumerate(coefficients):
            expanded[index] ^= value
            expanded[index + 1] ^= multiply_field(value, root)
        coefficients = expanded
        root = multiply_field(root, 2)
    return coefficients


def protect_block(data: bytes) -> bytes:
    """Append the thirty-two-byte polynomial remainder to a full data block."""
    if len(data) != 223:
        raise ValueError("Fixture data blocks must contain exactly 223 bytes.")
    divisor = generator_coefficients()
    remainder = list(data) + [0] * 32
    for position in range(223):
        leading = remainder[position]
        for offset, coefficient in enumerate(divisor):
            remainder[position + offset] ^= multiply_field(leading, coefficient)
    return data + bytes(remainder[-32:])


def payload_digest(protected: bytes, pixel_count: int) -> str:
    """Hash row-order map bytes, where each byte is a single zero or one."""
    bits = bytes(
        (value >> shift) & 1 for value in protected for shift in range(7, -1, -1)
    )
    complete_repeats, remaining_bits = divmod(pixel_count, len(bits))
    payload = bits * complete_repeats + bits[:remaining_bits]
    return hashlib.sha256(payload).hexdigest()


def make_vector(
    identifier: str, width: int, height: int, message: str
) -> dict[str, object]:
    """Construct one deterministic example using public input bytes only."""
    password = "PUBLIC TESTS ONLY: passphrase \u05e1\u05d9\u05e1\u05de\u05d4 \U0001f510"
    salt = bytes(range(16))
    nonce = bytes(range(24))
    message_bytes = message.encode("utf-8", errors="strict")
    capacity = min(1024, width * height // 1024)
    block_count = (capacity + 63 + 222) // 223
    if len(message_bytes) > capacity:
        raise ValueError("Fixture message exceeds its candidate capacity.")
    header = b"SGLB\x01" + salt + nonce
    context = COMPATIBILITY_IDENTIFIER.encode("ascii")
    associated_data = (
        DOMAIN
        + header
        + len(context).to_bytes(2, "big")
        + context
        + width.to_bytes(4, "big")
        + height.to_bytes(4, "big")
    )
    unpadded = len(message_bytes).to_bytes(2, "big") + message_bytes
    plaintext = unpadded.ljust(223 * block_count - 61, b"\x00")
    key = crypto_pwhash_alg(
        32,
        password.encode("utf-8", errors="strict"),
        salt,
        3,
        67_108_864,
        crypto_pwhash_ALG_ARGON2ID13,
    )
    encrypted = crypto_aead_xchacha20poly1305_ietf_encrypt(
        plaintext, associated_data, nonce, key
    )
    frame = header + encrypted
    blocks = [
        protect_block(frame[offset : offset + 223])
        for offset in range(0, len(frame), 223)
    ]
    protected = bytes(block[column] for column in range(255) for block in blocks)
    return {
        "identifier": identifier,
        "width": width,
        "height": height,
        "compatibility_identifier": COMPATIBILITY_IDENTIFIER,
        "message": message,
        "password": password,
        "salt_hex": salt.hex(),
        "nonce_hex": nonce.hex(),
        "candidate_message_bytes": capacity,
        "block_count": block_count,
        "message_utf8_hex": message_bytes.hex(),
        "expected_key_hex": key.hex(),
        "expected_header_hex": header.hex(),
        "expected_associated_data_hex": associated_data.hex(),
        "expected_frame_hex": frame.hex(),
        "expected_protected_hex": protected.hex(),
        "expected_payload_map_sha256": payload_digest(protected, width * height),
    }


def main() -> None:
    """Write fixtures only to a caller-selected new file."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    vectors = [
        make_vector("empty_512_square", 512, 512, ""),
        make_vector(
            "multilingual_odd_dimensions",
            513,
            517,
            "  Hello, \u05e9\u05dc\u05d5\u05dd \U0001f30d\nCafe\u0301\n\u4e16\u754c  ",
        ),
        make_vector("maximum_256_bytes", 512, 512, "A" * 256),
        make_vector("maximum_1024_bytes", 1024, 1024, "Z" * 1024),
    ]
    fixture = {
        "protocol_version": 1,
        "warning": WARNING,
        "derivation": {
            "generation_script": "scripts/generate_protocol_fixtures.py",
            "key_algorithm": "Argon2id v1.3",
            "key_bytes": 32,
            "memory_bytes": 67_108_864,
            "operation_limit": 3,
            "encryption": "XChaCha20-Poly1305-IETF",
            "data_bytes_per_block": 223,
            "parity_bytes_per_block": 32,
            "field_polynomial": "0x11d",
            "generator": 2,
            "first_root": 0,
            "symbol_bits": 8,
            "generator_polynomial_hex": bytes(generator_coefficients()).hex(),
            "payload_map_digest_format": "sha256 of row-order uint8 zero/one bytes",
        },
        "vectors": vectors,
    }
    with arguments.output.open("x", encoding="utf-8") as output:
        output.write(json.dumps(fixture, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
