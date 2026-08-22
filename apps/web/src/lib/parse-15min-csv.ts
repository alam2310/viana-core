export interface Aggregate15MinRow {
  date: string;
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

export function formatAggregateDateWindow(
  date: string,
  windowStart: string,
  windowEnd: string,
): string {
  const day = date.trim();
  const window = formatAggregateWindow(windowStart, windowEnd);
  return day ? `${day} ${window}` : window;
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

function colIndex(header: string[], ...names: string[]): number {
  for (const name of names) {
    const idx = header.indexOf(name);
    if (idx >= 0) {
      return idx;
    }
  }
  return -1;
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
  const dateIdx = colIndex(header, "Date", "date");
  const startIdx = colIndex(header, "Window Start", "window_start");
  const endIdx = colIndex(header, "Window End", "window_end");
  const classIdx = colIndex(header, "Class Name", "class_name");
  const directionIdx = colIndex(header, "Direction", "direction");
  const countIdx = colIndex(header, "Count", "count");

  if (
    startIdx < 0 ||
    endIdx < 0 ||
    dateIdx < 0 ||
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
    const date = cols[dateIdx] ?? "";
    const windowStart = cols[startIdx] ?? "";
    const windowEnd = cols[endIdx] ?? "";
    const vehicleClass = cols[classIdx] ?? "—";
    const direction = (cols[directionIdx] ?? "").toLowerCase();
    const count = Number.parseInt(cols[countIdx] ?? "0", 10) || 0;
    const key = `${date}|${windowStart}|${windowEnd}|${vehicleClass}`;

    let row = map.get(key);
    if (!row) {
      row = {
        date,
        windowStart,
        windowEnd,
        windowLabel: formatAggregateDateWindow(date, windowStart, windowEnd),
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
