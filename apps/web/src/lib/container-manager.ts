/**
 * Host-side Docker lifecycle for the GPU orchestrator container.
 * Used only from Next.js `/api/container/*` routes — never from the browser.
 */

import { execFile } from "node:child_process";
import { existsSync, readFileSync } from "node:fs";
import path from "node:path";
import { promisify } from "node:util";
import { parse as parseYaml } from "yaml";

import type {
  ContainerStatus,
  OrchestratorHostConfig,
} from "./container-types";

export type { ContainerStatus, OrchestratorHostConfig };

const execFileAsync = promisify(execFile);

const DEFAULT_CONFIG: OrchestratorHostConfig = {
  container: {
    image_name: "itva-base:stable",
    container_name: "viana_core",
    runtime: "nvidia",
  },
  api: { base_url: "http://localhost:8000" },
};

function repoRootFromCwd(): string {
  return path.resolve(process.cwd(), "../..");
}

export function resolveOrchestratorConfigPath(): string {
  const fromEnv = process.env.ORCHESTRATOR_CONFIG_PATH;
  if (fromEnv) {
    return path.isAbsolute(fromEnv)
      ? fromEnv
      : path.resolve(process.cwd(), fromEnv);
  }
  return path.resolve(
    process.cwd(),
    "../../docker/orchestrator_config.yaml",
  );
}

function loadYamlConfig(filePath: string): OrchestratorHostConfig | null {
  if (!existsSync(filePath)) {
    return null;
  }
  const raw = readFileSync(filePath, "utf8");
  const parsed = parseYaml(raw) as Partial<OrchestratorHostConfig> | null;
  if (!parsed || typeof parsed !== "object") {
    return null;
  }
  return {
    container: {
      image_name:
        parsed.container?.image_name ?? DEFAULT_CONFIG.container.image_name,
      container_name:
        parsed.container?.container_name ??
        DEFAULT_CONFIG.container.container_name,
      runtime: parsed.container?.runtime,
    },
    output: parsed.output,
    api: parsed.api,
  };
}

export function loadOrchestratorConfig(): {
  config: OrchestratorHostConfig;
  config_path: string;
  config_found: boolean;
} {
  const primary = resolveOrchestratorConfigPath();
  const loaded = loadYamlConfig(primary);
  if (loaded) {
    return { config: loaded, config_path: primary, config_found: true };
  }

  const examplePath = path.resolve(
    repoRootFromCwd(),
    "docker/orchestrator_config.yaml.example",
  );
  const example = loadYamlConfig(examplePath);
  return {
    config: example ?? DEFAULT_CONFIG,
    config_path: primary,
    config_found: false,
  };
}

async function dockerInspectRunning(containerName: string): Promise<{
  running: boolean;
  error?: string;
}> {
  try {
    const { stdout } = await execFileAsync(
      "docker",
      ["inspect", "-f", "{{.State.Running}}", containerName],
      { timeout: 8_000 },
    );
    return { running: stdout.trim().toLowerCase() === "true" };
  } catch (err) {
    const message = err instanceof Error ? err.message : String(err);
    if (message.includes("No such object")) {
      return { running: false, error: "container_not_found" };
    }
    if ((err as NodeJS.ErrnoException).code === "ENOENT") {
      return { running: false, error: "docker_not_installed" };
    }
    return { running: false, error: "docker_inspect_failed" };
  }
}

export async function getContainerStatus(): Promise<ContainerStatus> {
  const { config, config_path, config_found } = loadOrchestratorConfig();
  const { container_name, image_name } = config.container;
  const inspect = await dockerInspectRunning(container_name);
  return {
    running: inspect.running,
    container_name,
    image_name,
    config_path,
    config_found,
    error: inspect.error,
  };
}

export async function startContainer(): Promise<ContainerStatus> {
  const { config, config_path, config_found } = loadOrchestratorConfig();
  const { container_name } = config.container;

  try {
    await execFileAsync("docker", ["start", container_name], {
      timeout: 30_000,
    });
  } catch (startErr) {
    try {
      await execFileAsync("docker", ["compose", "up", "-d"], {
        cwd: repoRootFromCwd(),
        timeout: 120_000,
      });
    } catch (composeErr) {
      const startMsg =
        startErr instanceof Error ? startErr.message : String(startErr);
      const composeMsg =
        composeErr instanceof Error ? composeErr.message : String(composeErr);
      const status = await getContainerStatus();
      return {
        ...status,
        config_path,
        config_found,
        error: `docker start failed (${startMsg}); compose up failed (${composeMsg})`,
      };
    }
  }

  return getContainerStatus();
}
