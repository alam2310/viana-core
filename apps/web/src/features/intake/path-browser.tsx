"use client";

import { useCallback, useEffect, useState } from "react";

import { Button } from "@/components/ui/button";
import type { FsBrowseResponse } from "@/app/api/fs/browse/route";
import type { MountConfig } from "@/lib/container-paths";
import { readBrowsePath, writeBrowsePath } from "@/lib/prefs";
import { cn } from "@/lib/utils";

const DARK_GHOST_BTN =
  "dark:bg-zinc-100 dark:text-zinc-900 dark:hover:bg-zinc-200";
const DARK_OUTLINE_BTN =
  "dark:border-zinc-300 dark:bg-zinc-100 dark:text-zinc-900 dark:hover:bg-zinc-200";

export type PickerPurpose = "intake" | "output_dir";

async function fetchBrowse(path?: string): Promise<FsBrowseResponse> {
  const query = path ? `?path=${encodeURIComponent(path)}` : "";
  const response = await fetch(`/api/fs/browse${query}`);
  const data = (await response.json()) as FsBrowseResponse & { detail?: string };
  if (!response.ok) {
    throw new Error(data.detail ?? `Browse failed (${response.status})`);
  }
  return data;
}

async function createDirectory(parent: string, name: string): Promise<string> {
  const response = await fetch("/api/fs/mkdir", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ parent, name }),
  });
  const data = (await response.json()) as { path?: string; detail?: string };
  if (!response.ok || !data.path) {
    throw new Error(data.detail ?? `Create folder failed (${response.status})`);
  }
  return data.path;
}

export function PathBrowser({
  purpose,
  open,
  mountConfig,
  initialPath,
  viewOnly = false,
  onClose,
  onSelect,
}: {
  purpose: PickerPurpose;
  open: boolean;
  mountConfig: MountConfig | null;
  initialPath?: string;
  viewOnly?: boolean;
  onClose: () => void;
  onSelect: (paths: string[]) => void;
}) {
  const [current, setCurrent] = useState<FsBrowseResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [selected, setSelected] = useState<Set<string>>(new Set());
  const [newDirName, setNewDirName] = useState("");
  const [showNewDir, setShowNewDir] = useState(false);

  const load = useCallback(
    async (path?: string) => {
      setBusy(true);
      setError(null);
      try {
        const data = await fetchBrowse(path);
        setCurrent(data);
        writeBrowsePath(purpose, data.path);
      } catch (err) {
        setError(err instanceof Error ? err.message : String(err));
      } finally {
        setBusy(false);
      }
    },
    [purpose],
  );

  useEffect(() => {
    if (!open) {
      return;
    }
    setSelected(new Set());
    setShowNewDir(false);
    setNewDirName("");
    const defaultPath =
      purpose === "intake"
        ? mountConfig?.defaultBrowsePath
        : mountConfig?.mounts.find((m) => m.container === "/data")?.host;
    const start = initialPath ?? readBrowsePath(purpose) ?? defaultPath ?? undefined;
    void load(start);
  }, [initialPath, load, mountConfig, open, purpose]);

  if (!open) {
    return null;
  }

  const videoEntries =
    current?.entries.filter((entry) => entry.type === "file" && entry.isVideo) ??
    [];

  function toggleFile(path: string) {
    setSelected((prev) => {
      const next = new Set(prev);
      if (next.has(path)) {
        next.delete(path);
      } else {
        next.add(path);
      }
      return next;
    });
  }

  function addAndClose(paths: string[]) {
    if (paths.length === 0) {
      return;
    }
    if (current?.path) {
      writeBrowsePath(purpose, current.path);
    }
    onSelect(paths);
    onClose();
  }

  function handleEntryActivate(entry: FsBrowseResponse["entries"][number]) {
    if (entry.type === "directory") {
      void load(entry.path);
      return;
    }
    if (!entry.isVideo || purpose !== "intake") {
      return;
    }
    addAndClose([entry.path]);
  }

  async function handleCreateDirectory() {
    if (!current?.path || !newDirName.trim()) {
      return;
    }
    setBusy(true);
    setError(null);
    try {
      const created = await createDirectory(current.path, newDirName.trim());
      setShowNewDir(false);
      setNewDirName("");
      await load(created);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setBusy(false);
    }
  }

  const title = viewOnly
    ? "Job output"
    : purpose === "output_dir"
      ? "Select output directory"
      : "Select file(s) or folder";

  return (
    <div className="fixed inset-0 z-50 flex items-start justify-center overflow-y-auto bg-black/50 p-4">
      <div className="my-8 flex w-full max-w-2xl flex-col rounded-lg bg-card text-foreground shadow-xl">
        <div className="border-b border-border px-5 py-4">
          <div className="flex items-start justify-between gap-4">
            <div>
              <h2 className="text-lg font-semibold">{title}</h2>
              <p className="mt-1 font-mono text-xs text-muted">
                {current?.path ?? "…"}
              </p>
              {purpose === "intake" ? (
                <p className="mt-1 text-xs text-muted">
                  Double-click a video to add it. Double-click a folder to open it.
                </p>
              ) : null}
            </div>
            <Button
              type="button"
              size="sm"
              variant="ghost"
              className={DARK_GHOST_BTN}
              onClick={onClose}
            >
              Cancel
            </Button>
          </div>
        </div>

        {error ? (
          <p className="px-5 pt-3 text-sm text-red-600 dark:text-red-400">{error}</p>
        ) : null}

        <div className="flex flex-wrap items-center gap-2 px-5 pt-3">
          <Button
            type="button"
            size="sm"
            variant="outline"
            disabled={!current?.parent || busy}
            onClick={() => void load(current?.parent ?? undefined)}
          >
            ↑ Parent folder
          </Button>
          {purpose === "output_dir" ? (
            <Button
              type="button"
              size="sm"
              variant="outline"
              disabled={!current?.path || busy}
              onClick={() => setShowNewDir((prev) => !prev)}
            >
              New folder
            </Button>
          ) : null}
        </div>

        {showNewDir && purpose === "output_dir" ? (
          <div className="mx-5 mt-3 flex gap-2">
            <input
              className="min-w-0 flex-1 rounded border border-input bg-card px-2 py-1 font-mono text-xs text-foreground"
              placeholder="folder-name"
              value={newDirName}
              onChange={(event) => setNewDirName(event.target.value)}
            />
            <Button
              type="button"
              size="sm"
              disabled={!newDirName.trim() || busy}
              onClick={() => void handleCreateDirectory()}
            >
              Create
            </Button>
          </div>
        ) : null}

        <ul className="mx-5 mt-3 max-h-80 flex-1 divide-y divide-border overflow-y-auto rounded border border-border">
          {current?.entries.map((entry) => {
            const isSelected = entry.type === "file" && selected.has(entry.path);
            return (
              <li key={entry.path}>
                <button
                  type="button"
                  className={cn(
                    "flex w-full items-center gap-2 px-3 py-2 text-left text-sm",
                    entry.type === "directory" || entry.isVideo
                      ? "hover:bg-accent"
                      : "opacity-40",
                    isSelected && "bg-sky-100 ring-1 ring-inset ring-sky-300 dark:bg-sky-950 dark:ring-sky-700",
                  )}
                  disabled={entry.type === "file" && !entry.isVideo && purpose === "intake"}
                  onClick={() => {
                    if (entry.type === "file" && entry.isVideo && purpose === "intake") {
                      toggleFile(entry.path);
                    }
                  }}
                  onDoubleClick={() => handleEntryActivate(entry)}
                >
                  <span className="text-muted">
                    {entry.type === "directory" ? "📁" : entry.isVideo ? "🎬" : "📄"}
                  </span>
                  <span className="truncate font-mono text-xs">{entry.name}</span>
                  {isSelected ? (
                    <span className="ml-auto text-xs text-sky-700 dark:text-sky-300">
                      selected
                    </span>
                  ) : null}
                </button>
              </li>
            );
          })}
          {!current?.entries.length ? (
            <li className="px-3 py-4 text-sm text-muted">Empty directory</li>
          ) : null}
        </ul>

        <div className="flex flex-wrap items-center justify-end gap-2 border-t border-border px-5 py-4">
          {viewOnly ? (
            <Button
              type="button"
              variant="outline"
              className={DARK_OUTLINE_BTN}
              onClick={onClose}
            >
              Close
            </Button>
          ) : purpose === "output_dir" ? (
            <Button
              type="button"
              disabled={!current?.path || busy}
              onClick={() => current?.path && addAndClose([current.path])}
            >
              Use this folder
            </Button>
          ) : (
            <>
              <Button
                type="button"
                variant="outline"
                disabled={videoEntries.length === 0 || busy}
                onClick={() =>
                  addAndClose(videoEntries.map((entry) => entry.path))
                }
              >
                Add all videos here ({videoEntries.length})
              </Button>
              <Button
                type="button"
                disabled={selected.size === 0 || busy}
                onClick={() => addAndClose(Array.from(selected))}
              >
                Add selected ({selected.size})
              </Button>
            </>
          )}
        </div>
      </div>
    </div>
  );
}
