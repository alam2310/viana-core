import { NextResponse } from "next/server";

import { stopContainer } from "@/lib/container-manager";

export async function POST() {
  const status = await stopContainer();
  return NextResponse.json(status, { status: status.running ? 503 : 200 });
}
