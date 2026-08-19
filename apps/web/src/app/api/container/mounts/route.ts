import path from "node:path";

import { NextResponse } from "next/server";

import type { MountConfig } from "@/lib/container-paths";

/** Expose docker-compose bind mounts so the UI can translate host paths for intake. */
export async function GET(): Promise<NextResponse<MountConfig>> {
  const repoRoot = path.resolve(process.cwd(), "../..");
  const dataRoot = process.env.VIANA_DATA_ROOT
    ? path.resolve(process.env.VIANA_DATA_ROOT)
    : path.join(repoRoot, "data");
  const rawDir = path.join(dataRoot, "raw");

  return NextResponse.json({
    mounts: [
      {
        host: dataRoot,
        container: "/data",
        label: "Video data (host ./data → container /data)",
      },
      {
        host: repoRoot,
        container: "/app/ViAna",
        label: "Project root (host repo → container /app/ViAna)",
      },
    ],
    defaultBrowsePath: rawDir,
  });
}
