/** Host → container path translation (docker-compose volume mounts). */

export interface MountMapping {
  host: string;
  container: string;
  label?: string;
}

export interface MountConfig {
  mounts: MountMapping[];
  defaultBrowsePath: string;
}

function normalizePath(p: string): string {
  return p.replace(/\\/g, "/").replace(/\/+$/, "");
}

/** Map a host filesystem path to the path the orchestrator container can read. */
export function toContainerPath(
  hostPath: string,
  mounts: MountMapping[],
): { containerPath: string; translated: boolean } {
  const normalized = normalizePath(hostPath);
  const sorted = [...mounts].sort(
    (a, b) => normalizePath(b.host).length - normalizePath(a.host).length,
  );
  for (const mount of sorted) {
    const hostNorm = normalizePath(mount.host);
    if (normalized === hostNorm || normalized.startsWith(`${hostNorm}/`)) {
      const suffix = normalized.slice(hostNorm.length);
      return {
        containerPath: mount.container + suffix,
        translated: true,
      };
    }
  }
  return { containerPath: normalized, translated: false };
}

export function translatePaths(
  hostPaths: string[],
  mounts: MountMapping[],
): { paths: string[]; untranslated: string[] } {
  const paths: string[] = [];
  const untranslated: string[] = [];
  for (const hostPath of hostPaths) {
    const { containerPath, translated } = toContainerPath(hostPath, mounts);
    paths.push(containerPath);
    if (!translated) {
      untranslated.push(hostPath);
    }
  }
  return { paths, untranslated };
}

/** Map container path back to host for fs/mkdir and browse APIs. */
export function toHostPath(
  containerPath: string,
  mounts: MountMapping[],
): string | null {
  const normalized = normalizePath(containerPath);
  const sorted = [...mounts].sort(
    (a, b) => normalizePath(b.container).length - normalizePath(a.container).length,
  );
  for (const mount of sorted) {
    const containerNorm = normalizePath(mount.container);
    if (normalized === containerNorm || normalized.startsWith(`${containerNorm}/`)) {
      const suffix = normalized.slice(containerNorm.length);
      return normalizePath(mount.host) + suffix;
    }
  }
  return null;
}

export const CONTAINER_PATH_HINT =
  "Videos must live under a mounted directory. With default docker-compose: ./data → /data (e.g. data/raw/clip.mp4).";
