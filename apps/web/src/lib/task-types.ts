import type { TaskTypePref } from "@/lib/prefs";

export interface TaskTypeOption {
  id: TaskTypePref;
  label: string;
  description: string;
  enabled: boolean;
}

/** UI-facing task type catalog (contract values unchanged). */
export const TASK_TYPE_OPTIONS: TaskTypeOption[] = [
  {
    id: "ViAna_Moving",
    label: "Moving traffic count",
    description: "Vehicle detection, tracking, and line crossing counts",
    enabled: true,
  },
  {
    id: "ViAnaNP",
    label: "Non-motorized / parked (NP)",
    description: "Parked zone analytics — coming in a future release",
    enabled: false,
  },
  {
    id: "ViAnaJunction",
    label: "Junction turning counts",
    description: "Polygon gates and turning movements — coming soon",
    enabled: false,
  },
];

export function taskTypeLabel(id: TaskTypePref): string {
  return TASK_TYPE_OPTIONS.find((opt) => opt.id === id)?.label ?? id;
}
