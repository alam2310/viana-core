import { execFile } from "node:child_process";
import path from "node:path";
import { promisify } from "node:util";

import { NextResponse } from "next/server";

const execFileAsync = promisify(execFile);

async function openInFileManager(dirPath: string): Promise<void> {
  const resolved = path.resolve(dirPath);
  const platform = process.platform;
  if (platform === "win32") {
    await execFileAsync("explorer.exe", [resolved]);
    return;
  }
  if (platform === "darwin") {
    await execFileAsync("open", [resolved]);
    return;
  }
  await execFileAsync("xdg-open", [resolved]);
}

export async function POST(request: Request): Promise<NextResponse> {
  let body: { path?: string };
  try {
    body = (await request.json()) as { path?: string };
  } catch {
    return NextResponse.json({ detail: "Invalid JSON body" }, { status: 400 });
  }

  const requested = body.path?.trim();
  if (!requested) {
    return NextResponse.json({ detail: "path is required" }, { status: 400 });
  }

  try {
    await openInFileManager(requested);
    return NextResponse.json({ ok: true });
  } catch (err) {
    const message = err instanceof Error ? err.message : String(err);
    return NextResponse.json({ detail: message }, { status: 400 });
  }
}
