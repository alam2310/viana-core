import { NextResponse } from "next/server";

import { startContainer } from "@/lib/container-manager";

export async function POST() {
  const status = await startContainer();
  const ok = status.running;
  return NextResponse.json(status, { status: ok ? 200 : 503 });
}
