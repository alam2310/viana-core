const PROJECT_KEY = "viana.project_id";
const OUTPUT_DIR_KEY = "viana.output_dir";
const TELEMETRY_KEY = "viana.telemetry_detail";
const TASK_TYPE_KEY = "viana.task_type";
const INTAKE_BROWSE_KEY = "viana.browse_path_intake";
const OUTPUT_BROWSE_KEY = "viana.browse_path_output";
const THEME_KEY = "viana.theme";

export const DEFAULT_PROJECT_ID = "nh44";

const LEGACY_PROJECT_IDS = new Set(["test_ui", "nh48"]);

export type BrowsePurpose = "intake" | "output_dir";

function browseKey(purpose: BrowsePurpose): string {
  return purpose === "intake" ? INTAKE_BROWSE_KEY : OUTPUT_BROWSE_KEY;
}

export function readBrowsePath(purpose: BrowsePurpose): string | null {
  if (typeof window === "undefined") {
    return null;
  }
  return window.localStorage.getItem(browseKey(purpose));
}

export function writeBrowsePath(purpose: BrowsePurpose, dirPath: string): void {
  window.localStorage.setItem(browseKey(purpose), dirPath);
}

export function readProjectId(): string {
  if (typeof window === "undefined") {
    return DEFAULT_PROJECT_ID;
  }
  const stored = window.localStorage.getItem(PROJECT_KEY);
  if (!stored || LEGACY_PROJECT_IDS.has(stored)) {
    return DEFAULT_PROJECT_ID;
  }
  return stored;
}

export function writeProjectId(projectId: string): void {
  window.localStorage.setItem(PROJECT_KEY, projectId);
}

export function readOutputDir(): string {
  if (typeof window === "undefined") {
    return "";
  }
  return window.localStorage.getItem(OUTPUT_DIR_KEY) ?? "";
}

export function writeOutputDir(outputDir: string): void {
  window.localStorage.setItem(OUTPUT_DIR_KEY, outputDir);
}

export function readTelemetryDetail(): boolean {
  if (typeof window === "undefined") {
    return false;
  }
  return window.localStorage.getItem(TELEMETRY_KEY) === "true";
}

export function writeTelemetryDetail(enabled: boolean): void {
  window.localStorage.setItem(TELEMETRY_KEY, enabled ? "true" : "false");
}

export type TaskTypePref = "ViAna_Moving" | "ViAnaNP" | "ViAnaJunction";
export type UiTheme = "light" | "dark";

export function readTaskType(): TaskTypePref {
  if (typeof window === "undefined") {
    return "ViAna_Moving";
  }
  const raw = window.localStorage.getItem(TASK_TYPE_KEY);
  if (raw === "ViAnaNP" || raw === "ViAnaJunction" || raw === "ViAna_Moving") {
    return raw;
  }
  return "ViAna_Moving";
}

export function writeTaskType(taskType: TaskTypePref): void {
  window.localStorage.setItem(TASK_TYPE_KEY, taskType);
}

export function readThemePreference(): UiTheme | null {
  if (typeof window === "undefined") {
    return null;
  }
  const raw = window.localStorage.getItem(THEME_KEY);
  return raw === "light" || raw === "dark" ? raw : null;
}

export function writeThemePreference(theme: UiTheme): void {
  window.localStorage.setItem(THEME_KEY, theme);
}
