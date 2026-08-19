import type { MountMapping } from "@/lib/container-paths";
import { toHostPath } from "@/lib/container-paths";

/** Container path for job artifacts: `{base}/{project_id}`. */
export function projectOutputContainerPath(
  baseDir: string,
  projectId: string,
): string {
  const base = baseDir.trim().replace(/\/+$/, "");
  if (!base) {
    return "";
  }
  return `${base}/${projectId}`;
}

async function mkdirHost(parentHost: string, name: string): Promise<void> {
  const response = await fetch("/api/fs/mkdir", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ parent: parentHost, name }),
  });
  if (!response.ok) {
    const data = (await response.json()) as { detail?: string };
    if (!data.detail?.includes("EEXIST")) {
      throw new Error(data.detail ?? `Could not create folder ${name}`);
    }
  }
}

/** Create `{outputBase}/{projectId}` on host when browsed base is set; returns container path for intake. */
export async function ensureProjectOutputDir(
  outputBase: string,
  projectId: string,
  mounts: MountMapping[],
): Promise<string> {
  const containerPath = projectOutputContainerPath(outputBase, projectId);
  if (!containerPath) {
    return "";
  }
  const baseHost = toHostPath(outputBase.trim().replace(/\/+$/, ""), mounts);
  if (!baseHost) {
    return containerPath;
  }
  await mkdirHost(baseHost, projectId);
  return containerPath;
}
