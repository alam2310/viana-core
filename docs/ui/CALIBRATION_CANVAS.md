# Calibration Canvas Specification

## Coordinate system

- **Pixel space** matching `video_meta.width` × `video_meta.height`
- Origin top-left; x right, y down
- All endpoints clamped: `0 <= x < width`, `0 <= y < height`

## Lines

| Line | Color | Purpose |
|------|-------|---------|
| Horizon | Red | Filter distant detections |
| Counting | Green | Crossing trigger (segment, not infinite) |

## Interaction

1. Prescan returns `proposed_lines` → render on canvas over `preview_url` image
2. User drags either endpoint of each line
3. On drag end: clamp to frame bounds
4. Submit blocked until both lines have valid endpoints

## Resolution scaling (batch / profile)

When applying a profile or lines from another video:

```
scale_x = target_width / profile.reference_resolution[0]
scale_y = target_height / profile.reference_resolution[1]
new_point = (round(x * scale_x), round(y * scale_y))
```

Then clamp to target frame bounds.

## API payload

Send as `task_parameters.horizon_line` and `task_parameters.counting_line`:

```json
{
  "start": [120, 400],
  "end": [1800, 520]
}
```

## Validation errors (show in UI)

- Point out of bounds
- Line degenerate (start == end)
- Missing line before submit
