import { readFile, stat } from "node:fs/promises";
import path from "node:path";

import { NextResponse } from "next/server";

export async function GET(request: Request): Promise<NextResponse> {
  const { searchParams } = new URL(request.url);
  const requested = searchParams.get("path");

  if (!requested?.trim()) {
    return NextResponse.json({ detail: "path is required" }, { status: 400 });
  }

  const resolved = path.resolve(requested.trim());

  try {
    const info = await stat(resolved);
    if (!info.isFile()) {
      return NextResponse.json({ detail: "path is not a file" }, { status: 400 });
    }
    const content = await readFile(resolved, "utf-8");
    return NextResponse.json({ path: resolved, content });
  } catch (err) {
    const message = err instanceof Error ? err.message : String(err);
    return NextResponse.json({ detail: message }, { status: 400 });
  }
}
