"use client";

import { useState } from "react";

import { Button } from "@/components/ui/button";
import { PathBrowser } from "@/features/intake/path-browser";
import {
  CONTAINER_PATH_HINT,
  type MountConfig,
  translatePaths,
} from "@/lib/container-paths";

export function IntakePanel({
  disabled,
  busy,
  mountConfig,
  onIntake,
}: {
  disabled: boolean;
  busy: boolean;
  mountConfig: MountConfig | null;
  onIntake: (paths: string[]) => void | Promise<void>;
}) {
  const [pickerOpen, setPickerOpen] = useState(false);
  const [pathWarning, setPathWarning] = useState<string | null>(null);

  async function handleSelect(hostPaths: string[]) {
    if (hostPaths.length === 0 || !mountConfig) {
      return;
    }
    const { paths, untranslated } = translatePaths(hostPaths, mountConfig.mounts);
    if (untranslated.length > 0) {
      setPathWarning(
        `${CONTAINER_PATH_HINT} Unmapped: ${untranslated[0]}`,
      );
    } else {
      setPathWarning(null);
    }
    await onIntake(paths);
  }

  return (
    <section className="rounded-lg border border-neutral-200 bg-white p-4">
      <h2 className="text-sm font-semibold tracking-wide text-neutral-500 uppercase">
        Video intake
      </h2>
      <p className="mt-1 text-xs text-neutral-500">
        Pick videos under <code className="font-mono">./data</code> (container{" "}
        <code className="font-mono">/data</code>). Paths are translated automatically
        for the orchestrator.
      </p>
      <div className="mt-3">
        <Button
          type="button"
          size="sm"
          disabled={disabled || busy || !mountConfig}
          onClick={() => setPickerOpen(true)}
        >
          {busy ? "Adding…" : "Select file(s) or dir"}
        </Button>
      </div>
      {pathWarning ? (
        <p className="mt-3 rounded border border-amber-200 bg-amber-50 px-3 py-2 text-xs text-amber-900">
          {pathWarning}
        </p>
      ) : null}
      {pickerOpen ? (
        <PathBrowser
          purpose="intake"
          open
          mountConfig={mountConfig}
          onClose={() => setPickerOpen(false)}
          onSelect={(paths) => void handleSelect(paths)}
        />
      ) : null}
    </section>
  );
}
