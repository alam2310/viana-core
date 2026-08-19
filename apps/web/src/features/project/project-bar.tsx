"use client";

import { EngineControls } from "@/features/container/engine-controls";
import { Button } from "@/components/ui/button";
import { TASK_TYPE_OPTIONS } from "@/lib/task-types";
import { PROJECT_ID_PATTERN } from "@/lib/geometry";
import { projectOutputContainerPath } from "@/lib/output-paths";
import type { TaskTypePref } from "@/lib/prefs";
import type { ContainerStatus } from "@/lib/container-types";

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
  const fieldClass =
    "mt-1 w-full rounded border border-neutral-300 px-2 py-1.5 text-sm";
  const effectiveOutput = projectOutputContainerPath(outputDir, projectId);

  return (
    <section className="rounded-lg border border-neutral-200 bg-white p-4">
      <h2 className="text-sm font-semibold tracking-wide text-neutral-500 uppercase">
        Project
      </h2>
      <div className="mt-3 grid gap-3 lg:grid-cols-[minmax(0,1fr)_minmax(0,1.4fr)_minmax(0,16rem)_auto] lg:items-end">
        <label className="text-sm font-medium text-neutral-700">
          Project ID
          <input
            className={`${fieldClass} font-mono`}
            value={projectId}
            onChange={(event) => onProjectId(event.target.value)}
          />
        </label>
        <label className="text-sm font-medium text-neutral-700">
          Output Directory
          <div className="mt-1 flex gap-2">
            <input
              className="min-w-0 flex-1 rounded border border-neutral-300 px-2 py-1.5 font-mono text-xs"
              placeholder="/data/viana-outputs"
              value={outputDir}
              onChange={(event) => onOutputDir(event.target.value)}
            />
            <Button type="button" size="sm" variant="outline" onClick={onBrowseOutputDir}>
              Browse
            </Button>
          </div>
        </label>
        <label className="text-sm font-medium text-neutral-700">
          Analytics Type
          <select
            className={`${fieldClass} min-w-[14rem]`}
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
        <div className="flex flex-col justify-end gap-1">
          <p className="text-sm font-medium text-neutral-700">Analytics Engine</p>
          <EngineControls onStatusChange={onContainerStatus} />
        </div>
      </div>
      {!projectValid ? (
        <p className="mt-2 text-xs text-red-700">
          Project ID must match {PROJECT_ID_PATTERN.toString()}
        </p>
      ) : null}
      {effectiveOutput ? (
        <p className="mt-2 font-mono text-xs text-neutral-500">
          Jobs save to: {effectiveOutput}
        </p>
      ) : null}
      {selected ? (
        <p className="mt-1 text-xs text-neutral-500">{selected.description}</p>
      ) : null}
    </section>
  );
}
