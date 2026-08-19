import { mkdir } from "node:fs/promises";
import path from "node:path";

import { NextResponse } from "next/server";

const DIR_NAME_PATTERN = /^[a-zA-Z0-9][a-zA-Z0-9._-]*$/;

export async function POST(request: Request): Promise<NextResponse> {
  let body: { parent?: string; name?: string };
  try {
    body = (await request.json()) as { parent?: string; name?: string };
  } catch {
    return NextResponse.json({ detail: "invalid JSON body" }, { status: 400 });
  }

  const parent = body.parent?.trim();
  const name = body.name?.trim();
  if (!parent || !name) {
    return NextResponse.json(
      { detail: "parent and name are required" },
      { status: 400 },
    );
  }
  if (!DIR_NAME_PATTERN.test(name)) {
    return NextResponse.json(
      { detail: "invalid directory name" },
      { status: 400 },
    );
  }

  const target = path.resolve(path.join(parent, name));
  const resolvedParent = path.resolve(parent);
  if (!target.startsWith(resolvedParent + path.sep) && target !== resolvedParent) {
    return NextResponse.json({ detail: "invalid target path" }, { status: 400 });
  }

  try {
    await mkdir(target, { recursive: true });
    return NextResponse.json({ path: target });
  } catch (err) {
    const message = err instanceof Error ? err.message : String(err);
    return NextResponse.json({ detail: message }, { status: 400 });
  }
}
