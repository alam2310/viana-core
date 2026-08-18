import { NextResponse } from "next/server";

import { getContainerStatus } from "@/lib/container-manager";

export async function GET() {
  const status = await getContainerStatus();
  return NextResponse.json(status);
}
