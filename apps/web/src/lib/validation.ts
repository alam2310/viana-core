/** Client-side metadata validation (mirrors G4 / job.py). */

export const TIME_PATTERN = /^([01]\d|2[0-3]):[0-5]\d:[0-5]\d$/;
export const DATE_PATTERN = /^(0[1-9]|[12]\d|3[01])-(0[1-9]|1[0-2])-\d{4}$/;

export function validateMetadataFields(fields: {
  user_start_time?: string;
  user_start_date?: string;
  location?: string;
}): string[] {
  const issues: string[] = [];
  const time = fields.user_start_time?.trim() ?? "";
  const date = fields.user_start_date?.trim() ?? "";
  const location = fields.location?.trim() ?? "";

  if (!time) {
    issues.push("Start time is required (HH:MM:SS)");
  } else if (!TIME_PATTERN.test(time)) {
    issues.push("Start time must match HH:MM:SS");
  }
  if (!date) {
    issues.push("Start date is required (DD-MM-YYYY)");
  } else if (!DATE_PATTERN.test(date)) {
    issues.push("Start date must match DD-MM-YYYY");
  }
  if (!location) {
    issues.push("Location is required");
  }
  return issues;
}

export function formatEta(seconds: number | undefined): string {
  if (seconds === undefined || !Number.isFinite(seconds) || seconds <= 0) {
    return "—";
  }
  const total = Math.round(seconds);
  const hours = Math.floor(total / 3600);
  const minutes = Math.floor((total % 3600) / 60);
  const secs = total % 60;
  if (hours > 0) {
    return `~${hours}h ${minutes}m left`;
  }
  if (minutes > 0) {
    return `~${minutes}m ${secs}s left`;
  }
  return `~${secs}s left`;
}
