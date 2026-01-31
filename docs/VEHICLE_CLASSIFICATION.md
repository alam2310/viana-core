# Vehicle Classification Logic (v1.0)

This document defines the 3-level hierarchy used to classify vehicles detected by the UVH-26 model.

## Hierarchy Structure
1. **Category:** Primary use (Passenger vs. Goods)
2. **Class:** Speed and Size profile (Heavy Fast, Light Fast, Slow)
3. **Sub-Class:** Specific vehicle type (Aggregated from raw detection labels)

## Mapping Table

| Raw Input (UVH-26) | Target Category | Target Class | Target Sub-Class | Logic Notes |
| :--- | :--- | :--- | :--- | :--- |
| **Hatchback** | Passenger | Light Fast | **Car** | Aggregated into generic Car. |
| **Sedan** | Passenger | Light Fast | **Car** | Aggregated into generic Car. |
| **MUV** | Passenger | Light Fast | **Car** | Aggregated into generic Car. |
| **SUV** | Passenger | Light Fast | **Jeep** | Mapped to Jeep/Rugged profile. |
| **Van** | Passenger | Light Fast | **Van** | - |
| **Tempo-traveller**| Passenger | Light Fast | **Mini Bus** | **Merged** to avoid separate class. |
| **Mini-bus** | Passenger | Light Fast | **Mini Bus** | - |
| **Bus** | Passenger | Heavy Fast | **City Bus** | Default assumption for all buses. |
| **Three-wheeler** | Passenger | Light Fast | **Auto** | Assumed passenger auto. |
| **Two-wheeler** | Passenger | Light Fast | **MTW** | Motorized Two-Wheelers. |
| **Cycle** | Passenger | Slow | **Cycle** | - |
| **Truck** | Goods | Heavy Fast | **Truck** | - |
| **LCV** | Goods | Light Fast | **LCV** | - |
| **Other** | Goods | Light Fast | **Others** | Catch-all bucket. |

> **Note:** "Pedestrian", "Intercity Bus", and "Rickshaw Trolley" are defined in our target requirements but are currently not detected by UVH-26.

## Target Classification

Category,Class Type,Target Sub-Class,Strategy Source,Implementation Logic (Phase 2)
Passenger,Heavy Fast,City Bus,🤖 Direct AI,Default mapping for detection class Bus (ID 6).
Passenger,Heavy Fast,Intercity Bus,📐 Logic (Future),"Phase 1: Map to City Bus.  Phase 2: If Color != Green/Blue, re-tag."
Passenger,Light Fast,Mini Bus,🤖 Direct AI,Merged classes Mini-bus + Tempo-traveller → Mini Bus (ID 3).
Passenger,Light Fast,Van,🤖 Direct AI,Direct detection of class Van (ID 2).
Passenger,Light Fast,Car,🤖 Direct AI,"Merged classes Sedan, Hatchback, MUV → Car (ID 0)."
Passenger,Light Fast,Jeep,🔄 Proxy,Map class SUV → Jeep (ID 1).
Passenger,Light Fast,Taxi,📐 Logic,Detect Car. Extract crop. If Yellow Plate Region > Threshold → Re-tag Taxi.
Passenger,Light Fast,MTW,🤖 Direct AI,"Merged classes Bike, Scooter → MTW (ID 4)."
Passenger,Light Fast,Auto,🤖 Direct AI,Direct detection of Auto (ID 5).
Passenger,Slow,Cycle,🤖 Direct AI,Direct detection of Cycle (ID 9).
Passenger,Slow,Cycle Rickshaw,❌ Defer,No training data. Map to Others (ID 10) for Phase 1.
Passenger,Slow,Pedestrian,👯 Sidecar,Run YOLO11-Nano (COCO). Filter Class 0 (Person). Merge into stream.
Passenger,Slow,Others,🤖 Direct AI,Direct detection of Others (ID 10).
Goods,Heavy Fast,Truck,🤖 Direct AI,Direct detection of Truck (ID 7).
Goods,Heavy Fast,MCV,📐 Logic,Detect Truck. If Box Area < ThresholdMCV​ → Re-tag MCV.
Goods,Heavy Fast,Trailers,📐 Logic,Detect Truck. If Aspect Ratio (W/H) > 2.5 → Re-tag Trailer.
Goods,Light Fast,LCV,🤖 Direct AI,Direct detection of LCV (ID 8) (Tata Ace / Dost).
Goods,Slow,Carts,❌ Defer,No training data. Map to Others (ID 10) or ignore.