# Character rules (default for all future videos)

**Character:** Mecmation (grey beanie, red shirt, stick arms), `projects/rs5/cut/character.png`, rigged by `tools/stick_rig.py`.
The cap / blue-shirt character (`toon_rig.py`) is retired.

## Stance and expression: confident
- **Default pose: hands on hips** (`hips`) for the whole video, not arms hanging.
- Chin slightly up (head_dy -4), head steady: tilt at most ±1.5°, slow drift at most ±5 px.
- Calm, confident closed smile at rest; no surprised faces, no eyebrow flicks, rare blinks (every 6-9 s).
- Body and legs never move.

## Arms: only a few times
- **3 to 5 gestures per video**, only on the biggest moments: the hook, the car reveal, the final horsepower number, the final question.
- Gesture = one arm (`point_R` / `raise_R`), held about 2 s, then back to hands on hips. Smooth 5-frame ease, no shaking.
- **No arm movement just because he is talking.**

## Mouth: calm
- Small mouth shapes (about 75% size) and calmer lip-sync: each shape held at least 3 frames, wide-open shapes rare.
- Closed confident smile in every pause.
