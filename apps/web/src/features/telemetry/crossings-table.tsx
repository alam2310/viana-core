"use client";

import { useMemo } from "react";

import type { CrossingRow } from "@/features/telemetry/telemetry-formatters";
import { crossingArrowClass } from "@/features/telemetry/telemetry-formatters";

export function CrossingsTable({
  rows,
  maxRows = 30,
  reverse = true,
}: {
  rows: CrossingRow[];
  maxRows?: number;
  reverse?: boolean;
}) {
  // ⚡ Bolt: Memoize the visible rows slice and reversal.
  // Impact: Prevents O(N) array slicing and reversal operations on every render,
  // particularly when telemetry messages update rapidly.
  const visible = useMemo(
    () => (reverse ? rows.slice(-maxRows).reverse() : rows.slice(0, maxRows)),
    [rows, maxRows, reverse]
  );

  if (visible.length === 0) {
    return null;
  }

  return (
    <div className="max-h-48 overflow-y-auto">
      <table className="w-full text-left text-xs leading-tight">
        <thead className="sticky top-0 bg-card">
          <tr>
            <th className="px-2 py-0.5 font-medium text-muted">Time (HH:MM:SS)</th>
            <th className="px-2 py-0.5 font-medium text-muted">Class</th>
            <th className="px-2 py-0.5 text-center font-medium text-muted">
              Dir
            </th>
          </tr>
        </thead>
        <tbody>
          {visible.map((row) => (
            <tr key={row.id} className="border-t border-border">
              <td className="px-2 py-0.5 font-mono">{row.time}</td>
              <td className="px-2 py-0.5">{row.vehicle}</td>
              <td
                className={`px-2 py-0.5 text-center text-base leading-none ${crossingArrowClass(row.direction)}`}
              >
                {row.arrow}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
