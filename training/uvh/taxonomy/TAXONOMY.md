# UVH-26 label taxonomy (training only)

Maps raw UVH-26 detector class names to a 3-level ITVA hierarchy used when **building the training dataset**. Production inference uses `configs/classes.yaml` instead.

## Hierarchy

1. **Category** — Passenger vs Goods  
2. **Class type** — Heavy Fast, Light Fast, Slow  
3. **Sub-class** — ITVA target type (Car, MTW, LCV, …)

## Key mappings

| Raw UVH label | Sub-class | Notes |
|---------------|-----------|--------|
| Hatchback, Sedan, MUV | Car | Aggregated |
| SUV | Jeep | Rugged profile proxy |
| Van | Van | |
| Tempo-traveller, Mini-bus | Mini Bus | Merged |
| Bus | City Bus | |
| Three-wheeler | Auto | |
| Two-wheeler, bike, scooter | MTW | |
| Bicycle | Cycle | |
| Truck | Truck | Training id 7 → production Heavy Truck |
| LCV, tata-ace | LCV | |
| Others | Others | Catch-all |

Full machine-readable map: `vehicle_taxonomy.json` in this directory.

## Pedestrians

UVH-26 does not train pedestrian class. Production uses a separate YOLO11-L model (`models/pretrained/yolo11l.pt`) per `configs/engine_defaults.yaml`.
