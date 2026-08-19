"use client";

import { ContainerPanel } from "@/features/container/container-panel";
import { Button } from "@/components/ui/button";
import type { ContainerStatus } from "@/lib/container-types";
import { TASK_TYPE_OPTIONS } from "@/lib/task-types";
import { PROJECT_ID_PATTERN } from "@/lib/geometry";
import type { TaskTypePref } from "@/lib/prefs";

export function ProjectBar({
  projectId,
  outputDir,
  taskType,
  projectValid,
  onProjectId,
  onOutputDir,
  onTaskType,
  onBrowseOutputDir,
  onContainerStatus,
}: {
  projectId: string;
  outputDir: string;
  taskType: TaskTypePref;
  projectValid: boolean;
  onProjectId: (value: string) => void;
  onOutputDir: (value: string) => void;
  onTaskType: (value: TaskTypePref) => void;
  onBrowseOutputDir: () => void;
  onContainerStatus?: (status: ContainerStatus) => void;
}) {
  const selected = TASK_TYPE_OPTIONS.find((opt) => opt.id === taskType);

  return (
    <section className="rounded-lg border border-neutral-200 bg-white p-4">
      <h2 className="text-sm font-semibold tracking-wide text-neutral-500 uppercase">
        Project
      </h2>
      <div className="mt-3 grid gap-3 lg:grid-cols-[minmax(0,1fr)_minmax(0,1.4fr)_auto_auto] lg:items-end">
        <label className="text-sm">
          project_id
          <input
            className="mt-1 w-full rounded border border-neutral-300 px-2 py-1 font-mono text-sm"
            value={projectId}
            onChange={(event) => onProjectId(event.target.value)}
          />
          {!projectValid ? (
            <span className="mt-1 block text-xs text-red-700">
              Must match {PROJECT_ID_PATTERN.toString()}
            </span>
          ) : null}
        </label>
        <label className="text-sm">
          output_dir
          <div className="mt-1 flex gap-2">
            <input
              className="min-w-0 flex-1 rounded border border-neutral-300 px-2 py-1 font-mono text-xs"
              placeholder="/data/viana-outputs/{project_id}"
              value={outputDir}
              onChange={(event) => onOutputDir(event.target.value)}
            />
            <Button type="button" size="sm" variant="outline" onClick={onBrowseOutputDir}>
              Browse
            </Button>
          </div>
        </label>
        <label className="text-sm">
          Task type
          <select
            className="mt-1 w-full rounded border border-neutral-300 px-2 py-1 text-sm"
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
          {selected ? (
            <span className="mt-1 block text-xs text-neutral-500">{selected.description}</span>
          ) : null}
        </label>
        <div className="lg:pb-0.5">
          <ContainerPanel compact onStatusChange={onContainerStatus} />
        </div>
      </div>
    </section>
  );
}
