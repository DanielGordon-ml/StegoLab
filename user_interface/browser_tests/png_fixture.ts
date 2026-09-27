import { deflateSync } from 'node:zlib';

/** Compute the checksum every PNG chunk and every zip entry carries. */
export function crc32(bytes: Uint8Array) {
  let crc = 0xffffffff;
  for (const byte of bytes) {
    crc ^= byte;
    for (let bit = 0; bit < 8; bit += 1)
      crc = crc & 1 ? (crc >>> 1) ^ 0xedb88320 : crc >>> 1;
  }
  return (crc ^ 0xffffffff) >>> 0;
}

/** Wrap one chunk with its length, name and checksum. */
export function chunk(name: string, data: Uint8Array) {
  const header = Buffer.alloc(8);
  header.writeUInt32BE(data.length, 0);
  header.write(name, 4, 'ascii');
  const body = Buffer.concat([Buffer.from(name, 'ascii'), Buffer.from(data)]);
  const footer = Buffer.alloc(4);
  footer.writeUInt32BE(crc32(body), 0);
  return Buffer.concat([header, Buffer.from(data), footer]);
}

/**
 * Build a smooth RGB PNG in memory so the test needs no image files.
 * A shade from 0 to 35 brightens the blue channel so two images differ.
 */
export function png(width: number, height: number, shade = 0) {
  const ihdr = Buffer.alloc(13);
  ihdr.writeUInt32BE(width, 0);
  ihdr.writeUInt32BE(height, 4);
  ihdr.set([8, 2, 0, 0, 0], 8);
  const rows = Buffer.alloc((width * 3 + 1) * height);
  for (let y = 0; y < height; y += 1) {
    const offset = y * (width * 3 + 1);
    rows[offset] = 0;
    for (let x = 0; x < width; x += 1) {
      const pixel = offset + 1 + x * 3;
      rows[pixel] = (40 + (x * 200) / width) | 0;
      rows[pixel + 1] = (40 + (y * 200) / height) | 0;
      rows[pixel + 2] = Math.min(
        255,
        (120 + shade + ((x + y) * 100) / (width + height)) | 0,
      );
    }
  }
  return Buffer.concat([
    Buffer.from([137, 80, 78, 71, 13, 10, 26, 10]),
    chunk('IHDR', ihdr),
    chunk('IDAT', deflateSync(rows)),
    chunk('IEND', new Uint8Array()),
  ]);
}
