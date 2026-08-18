/** Host container manager types (safe to import from client components). */

export interface OrchestratorContainerConfig {
  image_name: string;
  container_name: string;
  runtime?: string;
}

export interface OrchestratorHostConfig {
  container: OrchestratorContainerConfig;
  output?: { parent_dir?: string };
  api?: { base_url?: string };
}

export interface ContainerStatus {
  running: boolean;
  container_name: string;
  image_name: string;
  config_path: string;
  config_found: boolean;
  error?: string;
}
