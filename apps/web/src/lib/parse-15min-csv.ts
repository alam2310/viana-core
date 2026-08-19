export interface Aggregate15MinRow {
  windowStart: string;
  windowEnd: string;
  windowLabel: string;
  vehicleClass: string;
  countIn: number;
  countOut: number;
}

/** Extract HH:MM from either ISO timestamps or already-normalized HH:MM values. */
export function isoTimestampToHm(value: string): string {
  if (!value.trim()) {
    return "—";
  }
  const hmMatch = value.match(/^(\d{2}):(\d{2})$/);
  if (hmMatch) {
    return `${hmMatch[1]}:${hmMatch[2]}`;
  }
  const match = value.match(/T(\d{2}):(\d{2})/);
  if (match) {
    return `${match[1]}:${match[2]}`;
  }
  return value;
}

export function formatAggregateWindow(windowStart: string, windowEnd: string): string {
  return `${isoTimestampToHm(windowStart)} – ${isoTimestampToHm(windowEnd)}`;
}

function parseCsvLine(line: string): string[] {
  const fields: string[] = [];
  let current = "";
  let inQuotes = false;
  for (let i = 0; i < line.length; i += 1) {
    const ch = line[i];
    if (ch === '"') {
      inQuotes = !inQuotes;
      continue;
    }
    if (ch === "," && !inQuotes) {
      fields.push(current);
      current = "";
      continue;
    }
    current += ch;
  }
  fields.push(current);
  return fields;
}

export function parse15MinCsv(text: string): Aggregate15MinRow[] {
  const trimmed = text.trim();
  if (!trimmed) {
    return [];
  }
  const lines = trimmed.split(/\r?\n/);
  if (lines.length < 2) {
    return [];
  }

  const header = parseCsvLine(lines[0]);
  const col = (name: string) => header.indexOf(name);
  const startIdx = col("window_start");
  const endIdx = col("window_end");
  const classIdx = col("class_name");
  const directionIdx = col("direction");
  const countIdx = col("count");

  if (
    startIdx < 0 ||
    endIdx < 0 ||
    classIdx < 0 ||
    directionIdx < 0 ||
    countIdx < 0
  ) {
    return [];
  }

  const map = new Map<string, Aggregate15MinRow>();

  for (const line of lines.slice(1)) {
    if (!line.trim()) {
      continue;
    }
    const cols = parseCsvLine(line);
    const windowStart = cols[startIdx] ?? "";
    const windowEnd = cols[endIdx] ?? "";
    const vehicleClass = cols[classIdx] ?? "—";
    const direction = (cols[directionIdx] ?? "").toLowerCase();
    const count = Number.parseInt(cols[countIdx] ?? "0", 10) || 0;
    const key = `${windowStart}|${windowEnd}|${vehicleClass}`;

    let row = map.get(key);
    if (!row) {
      row = {
        windowStart,
        windowEnd,
        windowLabel: formatAggregateWindow(windowStart, windowEnd),
        vehicleClass,
        countIn: 0,
        countOut: 0,
      };
      map.set(key, row);
    }
    if (direction === "in") {
      row.countIn = count;
    } else if (direction === "out") {
      row.countOut = count;
    }
  }

  return Array.from(map.values());
}

export async function fetch15MinCsv(hostPath: string): Promise<string> {
  const query = `?path=${encodeURIComponent(hostPath)}`;
  const response = await fetch(`/api/fs/read${query}`);
  const data = (await response.json()) as { content?: string; detail?: string };
  if (!response.ok) {
    throw new Error(data.detail ?? `Read failed (${response.status})`);
  }
  return data.content ?? "";
}
