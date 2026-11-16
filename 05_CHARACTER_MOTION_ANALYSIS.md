# Character Motion Analysis: stick-arm talking character reference

Source: 576×562, **24 fps** (1,835 frames, 76.5 s). It's a talking character on a flat grey background, with one B-roll cut (engine footage, 20–26 s) and a TikTok outro.
Method: frame-by-frame differencing per body region, head tracking (beanie position and tilt), hand tracking, mouth crops stabilised against head movement, and speech recognition with word timestamps to link movements to words.

## 1. Global timing
| Metric | Value |
|---|---|
| Frames identical to the previous frame | **44%** (partly animated "on twos": each drawing held 2 frames) |
| Body parts that move | mouth 34% of frames, head 49%, torso 37%, arms 15–18%, **legs ~4% (static)** |
| Pace | ~3.4 words/s; a gesture every **2.0 s** (~1 per 6–7 words) |

## 2. Mouth (lip-sync)
- **Phoneme-based** (visemes), not volume-based. About **7 mouth drawings**:

| # | Drawing | Used for |
|---|---|---|
| 1 | Closed curved line | rest, pauses, **M / B / P** |
| 2 | Teeth smile (white band) | **E, S, T, D, N**, general consonants |
| 3 | Open: top teeth, dark mouth, red tongue | **A / I / open vowels** |
| 4 | Small dark "o" dot | **O / U / W / OO** |
| 5 | Two separated teeth rows | **EE, CH, SH, J** |
| 6 | Jagged bottom teeth | **F / V** |
| 7 | Small rounded open "O" with tongue | **OH / AW** |

- **Hold per shape:** 1–9 frames, mostly **2–3 frames** (mean 2.6 frames = 109 ms, ~9 shapes/s).
- Back to the closed line (shape 1) in every pause, even short ones.

## 3. Head
- **Sideways sway:** smooth drift, x range **~32 px (±16) on a ~215 px head**, frame-to-frame jitter only 0.85 px. It drifts slowly instead of shaking.
- **Vertical:** almost none (std 0.7 px). Nods are rare (~1 per 19 s).
- **Tilt:** upright most of the time (|tilt| > 1° on 30% of frames, > 3° on ~9%). It tilts a little **together with gestures** and stays within about ±3° in character shots.
- The **torso follows the head slightly** (body sway ±4 px). Legs and feet stay planted.

## 4. Arms and hands
- The arms are **thin bendable lines** with 4-finger line hands. The elbows bend as smooth curves, with no rigid joints.
- **Gesture move:** median **8–9 frames (0.35 s)**, range 4–18 frames. The move **eases out with a small overshoot**: the arms go slightly past the target (frames 8–9) and settle back over 2–3 frames.
- **Hold:** about **1.1–1.2 s**, then the next gesture or back to rest. 54 arm moves in 70 s of character shots.
- **Pose library:**
  1. **Relaxed rest:** arms hang slightly away from the body, hands loose.
  2. **Arms tight at the sides:** the default talking pose between gestures.
  3. **Both arms open:** a shrug or "check this out".
  4. **Presenting:** one arm extended to the side, the other forearm bent across the stomach (**the most frequent gesture**).
  5. **Pointing down or forward** with one arm.
- **When gestures happen:** on the **key word of each phrase**, starting about 0–0.2 s before it. Examples from the transcript:
  - "Hellcat **performance**", "price **tag**", "the **craziest**", "the **foundation** is"
  - each **year**, "under the **hood**", "here's where things get **interesting**"
  - "**three main upgrades**", "**six hundred** horsepower", "**absolutely** embarrasses"
  - "**muscle cars, sports cars**" (one gesture per item), "it's **fast, stealthy**", the final **question**

## 5. Eyes and brows
- **Blinks:** one every **~3.5 s**, lasting 2–6 frames (mostly 3–4).
- **Eyebrow raises:** one every **~5 s**, about 0.5 s long, on emphasis and questions.

## 6. Rules for our version (30 fps output)
- **Lip-sync:** map each word's letters to the 7 visemes. Hold each shape 2–4 frames (70–130 ms at 30 fps) and close the mouth in pauses.
- **Arms:** draw them as curved strokes with a smooth elbow bend. A move takes 10–11 frames (0.35 s) with an ease-out and ~8% overshoot, then holds 1.2 s and returns to the tight-at-sides pose.
- **Gesture timing:** one gesture on each phrase's key word (~every 2 s). Use the presenting pose most often, both arms open for big claims and questions, and one gesture per item in lists.
- **Head:** slow sideways drift ±16 px (scaled to our head size), tilt ≤ 3° along with gestures, a nod every ~15–20 s.
- **Torso:** sways ±4 px with the head. Legs and feet don't move.
- **Blinks and brows:** a blink every 3–4 s (3–4 frames), and a brow raise every ~5 s, timed to emphasis words and questions.
