/** Public env consumed by the browser api-client. */

export const API_BASE_URL =
  process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

/** Fixture mode until endpoints are ✅ in docs/PROJECT_STATUS.md. */
export const USE_MOCKS = process.env.NEXT_PUBLIC_USE_MOCKS !== "false";
