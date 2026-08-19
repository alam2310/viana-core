export async function openPathInFileManager(hostPath: string): Promise<void> {
  const response = await fetch("/api/fs/open", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ path: hostPath }),
  });
  const data = (await response.json()) as { detail?: string };
  if (!response.ok) {
    throw new Error(data.detail ?? `Open failed (${response.status})`);
  }
}
