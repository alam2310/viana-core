"use client";

import { useEffect, useState } from "react";

import type { Aggregate15MinRow } from "@/lib/parse-15min-csv";
import {
  fetch15MinCsv,
  formatAggregateWindow,
  parse15MinCsv,
} from "@/lib/parse-15min-csv";

export function Aggregate15MinTable({ csvHostPath }: { csvHostPath: string }) {
  const [rows, setRows] = useState<Aggregate15MinRow[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    setError(null);
    void (async () => {
      try {
        const text = await fetch15MinCsv(csvHostPath);
        if (cancelled) {
          return;
        }
        const parsed = parse15MinCsv(text);
        setRows(parsed);
      } catch (err) {
        if (!cancelled) {
          setError(err instanceof Error ? err.message : String(err));
        }
      } finally {
        if (!cancelled) {
          setLoading(false);
        }
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [csvHostPath]);

  if (loading) {
    return <p className="px-2 py-2 text-xs text-muted">Loading 15-minute report…</p>;
  }

  if (error) {
    return <p className="px-2 py-2 text-xs text-red-600 dark:text-red-400">{error}</p>;
  }

  if (rows.length === 0) {
    return (
      <p className="px-2 py-2 text-xs text-muted">
        Start time was not set or the report is empty — 15-minute aggregation is
        unavailable.
      </p>
    );
  }

  return (
    <div className="max-h-48 overflow-x-auto overflow-y-auto">
      <table className="w-full table-fixed text-left text-xs leading-tight">
        <colgroup>
          <col style={{ width: "46%" }} />
          <col style={{ width: "30%" }} />
          <col style={{ width: "12%" }} />
          <col style={{ width: "12%" }} />
        </colgroup>
        <thead className="sticky top-0 bg-card">
          <tr>
            <th className="px-1.5 py-0.5 font-medium text-muted whitespace-nowrap">
              15 min window
            </th>
            <th className="px-1.5 py-0.5 font-medium text-muted whitespace-nowrap">
              Class
            </th>
            <th className="px-1.5 py-0.5 text-right font-medium text-muted whitespace-nowrap">
              In
            </th>
            <th className="px-1.5 py-0.5 text-right font-medium text-muted whitespace-nowrap">
              Out
            </th>
          </tr>
        </thead>
        <tbody>
          {rows.map((row) => (
            <tr
              key={`${row.date}-${row.windowStart}-${row.windowEnd}-${row.vehicleClass}`}
              className="border-t border-border"
            >
              <td className="truncate px-1.5 py-0.5 font-mono whitespace-nowrap">
                {formatAggregateWindow(row.windowStart, row.windowEnd)}
              </td>
              <td className="truncate px-1.5 py-0.5 whitespace-nowrap">
                {row.vehicleClass}
              </td>
              <td className="px-1.5 py-0.5 text-right tabular-nums whitespace-nowrap">
                {row.countIn}
              </td>
              <td className="px-1.5 py-0.5 text-right tabular-nums whitespace-nowrap">
                {row.countOut}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
