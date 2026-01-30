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