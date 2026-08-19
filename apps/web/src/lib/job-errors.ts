/** Operator-friendly text for orchestrator / engine error_message values. */

export function formatJobErrorMessage(
  message: string | null | undefined,
): string | null {
  if (!message?.trim()) {
    return null;
  }
  const raw = message.trim();

  const exitMatch = /^viana exited (\d+)$/i.exec(raw);
  if (exitMatch) {
    const code = exitMatch[1];
    if (code === "1") {
      return (
        "Processing failed. If you re-ran the same video into the same output folder, " +
        "remove existing output files (events CSV, processed MP4) or choose a different " +
        "output directory, then use Start fresh or intake again."
      );
    }
    return `Processing failed (engine exit code ${code}). Check container logs for details.`;
  }

  if (/processed video not found|no such file/i.test(raw)) {
    return "Output files are missing or incomplete. Try Start fresh or re-intake the video.";
  }

  if (/video not found/i.test(raw)) {
    return "Source video not found inside the container. Re-select the file under ./data or fix the mount path.";
  }

  if (/prescan.*failed/i.test(raw)) {
    return `Prescan failed: ${raw}`;
  }

  return raw;
}

/** Extract GPU index from cuda:N for table display. */
export function gpuIdFromDevice(gpuDevice: string | null | undefined): string {
  if (!gpuDevice) {
    return "—";
  }
  const match = /^cuda:(\d+)$/.exec(gpuDevice);
  return match ? match[1] : gpuDevice;
}
