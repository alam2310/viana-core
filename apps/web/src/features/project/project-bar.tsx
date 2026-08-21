"use client";

import { Button } from "@/components/ui/button";
import { TASK_TYPE_OPTIONS } from "@/lib/task-types";
import { PROJECT_ID_PATTERN } from "@/lib/geometry";
import { DEFAULT_PROJECT_ID, type TaskTypePref } from "@/lib/prefs";

export function ProjectBar({
  projectId,
  outputDir,
  taskType,
  projectValid,
  onProjectId,
  onOutputDir,
  onTaskType,
  onBrowseOutputDir,
}: {
  projectId: string;
  outputDir: string;
  taskType: TaskTypePref;
  projectValid: boolean;
  onProjectId: (value: string) => void;
  onOutputDir: (value: string) => void;
  onTaskType: (value: TaskTypePref) => void;
  onBrowseOutputDir: () => void;
}) {
  const selected = TASK_TYPE_OPTIONS.find((opt) => opt.id === taskType);
  const fieldClass =
    "mt-1 w-full rounded border border-input bg-card px-2 py-1.5 text-sm text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-border";

  return (
    <section className="rounded-lg border border-border bg-card p-4">
      <div className="grid gap-3 lg:grid-cols-[minmax(0,1fr)_10rem_minmax(0,1.5fr)] lg:items-end">
        <label className="text-sm font-medium">
          Analytics Type
          <select
            className={fieldClass}
            value={taskType}
            onChange={(event) => onTaskType(event.target.value as TaskTypePref)}
          >
            {TASK_TYPE_OPTIONS.map((opt) => (
              <option key={opt.id} value={opt.id} disabled={!opt.enabled}>
                {opt.label}
                {!opt.enabled ? " — Coming soon" : ""}
              </option>
            ))}
          </select>
        </label>
        <label className="text-sm font-medium">
          Project ID
          <input
            className={`${fieldClass} font-mono text-xs`}
            value={projectId}
            onFocus={(event) => {
              if (projectId === DEFAULT_PROJECT_ID) {
                onProjectId("");
                event.currentTarget.select();
              }
            }}
            onBlur={() => {
              if (!projectId.trim()) {
                onProjectId(DEFAULT_PROJECT_ID);
              }
            }}
            onChange={(event) => onProjectId(event.target.value)}
          />
        </label>
        <label className="text-sm font-medium">
          Output Directory
          <div className="mt-1 flex gap-2">
            <input
              className="min-w-0 flex-1 rounded border border-input bg-card px-2 py-1.5 font-mono text-xs text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-border"
              placeholder="/data/viana-outputs"
              value={outputDir}
              onChange={(event) => onOutputDir(event.target.value)}
            />
            <Button
              type="button"
              size="sm"
              variant="outline"
              onClick={onBrowseOutputDir}
            >
              Browse
            </Button>
          </div>
        </label>
      </div>
      {!projectValid ? (
        <p className="mt-2 text-xs text-red-600 dark:text-red-400">
          Project ID must match {PROJECT_ID_PATTERN.toString()}
        </p>
      ) : null}
      {selected ? (
        <p className="mt-2 text-sm text-muted">{selected.description}</p>
      ) : null}
    </section>
  );
}
