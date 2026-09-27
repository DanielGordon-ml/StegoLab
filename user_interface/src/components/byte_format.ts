const UNITS = ['B', 'kB', 'MB', 'GB', 'TB'] as const;
export type ByteUnit = (typeof UNITS)[number];

/** Pick the SI unit that keeps a byte count below 1000, capped at terabytes. */
export function byte_unit(bytes: number): ByteUnit {
  let value = Number.isFinite(bytes) ? Math.max(0, bytes) : 0;
  let index = 0;
  while (value >= 999.95 && index < UNITS.length - 1) {
    value /= 1000;
    index += 1;
  }
  return UNITS[index];
}

/** Show a byte count in short SI units with one decimal, such as "12.0 MB". */
export function format_bytes(
  bytes: number,
  unit: ByteUnit = byte_unit(bytes),
): string {
  if (!Number.isFinite(bytes) || bytes < 0) return '0 B';
  const scaled = bytes / 1000 ** UNITS.indexOf(unit);
  return unit === 'B'
    ? `${Math.round(scaled)} B`
    : `${scaled.toFixed(1)} ${unit}`;
}

/** Show progress with one shared unit, such as "12.0 of 100.0 MB". */
export function format_byte_progress(received: number, total: number): string {
  const unit = byte_unit(total);
  const amount = (value: number) =>
    format_bytes(value, unit).slice(0, -(unit.length + 1));
  return `${amount(received)} of ${amount(total)} ${unit}`;
}
