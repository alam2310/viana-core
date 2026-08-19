import type { JobStatus } from "@viana/contracts";

/** Bar fill color from orange (start) through amber/yellow to green (complete). */
export function progressBarColor(pct: number, status?: JobStatus): string {
  if (status === "COMPLETED" || pct >= 100) {
    return "bg-emerald-500";
  }
  if (status === "FAILED" || status === "CANCELLED") {
    return "bg-red-500";
  }
  if (pct >= 85) {
    return "bg-lime-500";
  }
  if (pct >= 65) {
    return "bg-yellow-500";
  }
  if (pct >= 40) {
    return "bg-amber-500";
  }
  if (pct >= 15) {
    return "bg-orange-500";
  }
  return "bg-orange-600";
}
