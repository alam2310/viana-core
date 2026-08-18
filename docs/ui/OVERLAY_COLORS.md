# Processed-video overlay colors

Canonical legend for `{stem}_processed.mp4`. Code: `src/viana/stages/render.py` (`OVERLAY_BGR`). Pixel origin is top-left; OpenCV stores **BGR**.

This file lives only under `docs/ui/`. Do not copy it into job output directories.

## Calibration lines (all jobs)

| Line | On video | Hex (approx.) | Purpose |
|------|----------|---------------|---------|
| Horizon | Red | `#ff0000` | Drop detections whose center is above this line |
| Counting | Green | `#00ff00` | Crossing trigger (once per track) |

## Box + label color by class

Label format: `{class_name} #{track_id}` using `classes.yaml` **name** (not `sub_class`).

| id | `name` | `category` | `class_type` | `sub_class` | On video | Hex (approx.) |
|----|--------|------------|--------------|-------------|----------|---------------|
| 0 | Car | Passenger | Light Fast | Car | blue | `#1e90ff` |
| 1 | Jeep | Passenger | Light Fast | Jeep | orange | `#ff8c00` |
| 2 | Van | Passenger | Light Fast | Van | magenta | `#ff00ff` |
| 3 | MiniBus | Passenger | Light Fast | Mini Bus | pink | `#ff69b4` |
| 4 | MTW | Passenger | Light Fast | MTW | yellow | `#ffff00` |
| 5 | Auto | Passenger | Light Fast | Auto | plum | `#ffc0cb` |
| 6 | Bus | Passenger | Heavy Fast | Bus | gold | `#ffd700` |
| 7 | Heavy Truck | Goods | Heavy Fast | Truck | brown | `#a52a2a` |
| 8 | LCV | Goods | Light Fast | LCV | saddle | `#8b4513` |
| 9 | Cycle | Passenger | Slow | Cycle | cyan | `#00ffff` |
| 10 | Other | Goods | Light Fast | Others | gray | `#a0a0a0` |
| 11 | Pedestrian | Passenger | Slow | Pedestrian | violet | `#7f00ff` |
| 12 | MCV | Goods | Heavy Fast | MCV | coral | `#ff7f50` |
| 13 | Trailer | Goods | Heavy Fast | Trailers | indigo | `#4b0082` |
| 14 | Taxi | Passenger | Light Fast | Taxi | amber | `#ffc800` |

Unknown class id: light gray `#dcdcdc`.

`category`, `class_type`, and `sub_class` are **not drawn** on the video. They are filled on `{stem}_events.csv` by looking up the same `classes.yaml` row as `class_id`.
