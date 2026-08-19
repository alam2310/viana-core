import { readdir, stat } from "node:fs/promises";
import path from "node:path";

import { NextResponse } from "next/server";

const VIDEO_EXTENSIONS = new Set([
  ".mp4",
  ".avi",
  ".mkv",
  ".mov",
  ".webm",
  ".m4v",
]);

export interface FsEntry {
  name: string;
  path: string;
  type: "file" | "directory";
  isVideo?: boolean;
}

export interface FsBrowseResponse {
  path: string;
  parent: string | null;
  entries: FsEntry[];
}

function isVideoFile(name: string): boolean {
  const ext = path.extname(name).toLowerCase();
  return VIDEO_EXTENSIONS.has(ext);
}

async function listDirectory(dirPath: string): Promise<FsBrowseResponse> {
  const resolved = path.resolve(dirPath);
  const parent = path.dirname(resolved);
  const entries: FsEntry[] = [];

  const dirents = await readdir(resolved, { withFileTypes: true });

  for (const dirent of dirents) {
    const entryPath = path.join(resolved, dirent.name);
    if (dirent.isDirectory()) {
      entries.push({ name: dirent.name, path: entryPath, type: "directory" });
    } else if (dirent.isFile()) {
      entries.push({
        name: dirent.name,
        path: entryPath,
        type: "file",
        isVideo: isVideoFile(dirent.name),
      });
    }
  }

  entries.sort((a, b) => {
    if (a.type !== b.type) {
      return a.type === "directory" ? -1 : 1;
    }
    return a.name.localeCompare(b.name);
  });

  return {
    path: resolved,
    parent: parent !== resolved ? parent : null,
    entries,
  };
}

export async function GET(request: Request): Promise<NextResponse> {
  const { searchParams } = new URL(request.url);
  const requested = searchParams.get("path");

  let startPath: string;
  if (!requested) {
    startPath = process.env.HOME ?? "/";
  } else {
    startPath = path.resolve(requested);
  }

  try {
    const info = await stat(startPath);
    if (!info.isDirectory()) {
      return NextResponse.json(
        { detail: "path is not a directory" },
        { status: 400 },
      );
    }
    const payload = await listDirectory(startPath);
    return NextResponse.json(payload);
  } catch (err) {
    const message = err instanceof Error ? err.message : String(err);
    return NextResponse.json({ detail: message }, { status: 400 });
  }
}
