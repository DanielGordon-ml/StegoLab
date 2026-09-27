import { crc32 } from './png_fixture';

/** One file to place in the archive, stored without compression. */
export interface ZipEntry {
  name: string;
  data: Uint8Array;
}

const LOCAL_HEADER_SIGNATURE = 0x04034b50;
const CENTRAL_HEADER_SIGNATURE = 0x02014b50;
const END_RECORD_SIGNATURE = 0x06054b50;
const VERSION_MADE_BY = 20;
const VERSION_NEEDED = 10;
const STORE_METHOD = 0;
/** 1 January 1980, 00:00, the earliest moment the zip format can record. */
const DOS_TIME = 0;
const DOS_DATE = (1 << 5) | 1;

/** The fields both header kinds share, in the order the format lists them. */
function shared_fields(entry: ZipEntry, name: Buffer) {
  const fields = Buffer.alloc(26);
  fields.writeUInt16LE(VERSION_NEEDED, 0);
  fields.writeUInt16LE(0, 2);
  fields.writeUInt16LE(STORE_METHOD, 4);
  fields.writeUInt16LE(DOS_TIME, 6);
  fields.writeUInt16LE(DOS_DATE, 8);
  fields.writeUInt32LE(crc32(entry.data), 10);
  fields.writeUInt32LE(entry.data.length, 14);
  fields.writeUInt32LE(entry.data.length, 18);
  fields.writeUInt16LE(name.length, 22);
  fields.writeUInt16LE(0, 24);
  return fields;
}

/** The record that precedes one stored file inside the archive. */
function local_header(entry: ZipEntry, name: Buffer) {
  const signature = Buffer.alloc(4);
  signature.writeUInt32LE(LOCAL_HEADER_SIGNATURE, 0);
  return Buffer.concat([signature, shared_fields(entry, name), name]);
}

/** The directory record that points back at one local header. */
function central_header(entry: ZipEntry, name: Buffer, offset: number) {
  const head = Buffer.alloc(6);
  head.writeUInt32LE(CENTRAL_HEADER_SIGNATURE, 0);
  head.writeUInt16LE(VERSION_MADE_BY, 4);
  // Comment length, disk number, internal and external attributes, then the offset.
  const tail = Buffer.alloc(14);
  tail.writeUInt16LE(0, 0);
  tail.writeUInt16LE(0, 2);
  tail.writeUInt16LE(0, 4);
  tail.writeUInt32LE(0, 6);
  tail.writeUInt32LE(offset, 10);
  return Buffer.concat([head, shared_fields(entry, name), tail, name]);
}

/** Build a store-only zip archive in memory with a full central directory. */
export function zip(entries: ZipEntry[]) {
  const parts: Buffer[] = [];
  const directory: Buffer[] = [];
  let offset = 0;
  for (const entry of entries) {
    const name = Buffer.from(entry.name, 'utf8');
    const header = local_header(entry, name);
    directory.push(central_header(entry, name, offset));
    parts.push(header, Buffer.from(entry.data));
    offset += header.length + entry.data.length;
  }
  const directory_bytes = Buffer.concat(directory);
  const end = Buffer.alloc(22);
  end.writeUInt32LE(END_RECORD_SIGNATURE, 0);
  end.writeUInt16LE(0, 4);
  end.writeUInt16LE(0, 6);
  end.writeUInt16LE(entries.length, 8);
  end.writeUInt16LE(entries.length, 10);
  end.writeUInt32LE(directory_bytes.length, 12);
  end.writeUInt32LE(offset, 16);
  end.writeUInt16LE(0, 20);
  return Buffer.concat([...parts, directory_bytes, end]);
}
