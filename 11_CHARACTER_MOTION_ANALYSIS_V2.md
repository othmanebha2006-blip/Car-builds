# Character Motion Analysis V2: cap-and-blue-shirt reference

Source: 576×550, **30 fps** (1,804 frames, 60 s); the file is 30 fps, not 60. It's a talking character on a flat grey background, with the parts corner mode at 29–38 s and a TikTok outro at 56 s+. 1,398 full-character frames analysed.
Method: per-region frame differencing, cap tracking (position and tilt), mouth-change intervals, pose-change detection, and speech recognition to tie each pose change to the words being said.

## 1. Overall
| Metric | Value |
|---|---|
| Frames identical to the previous frame | **75%** (very calm, mostly held drawings) |
| Mouth changing | 22% of frames |
| Head changing | 25% |
| Torso / legs / arms changing | **2–3%** (practically static) |
| Torso sideways sway | **0 px** (the body never moves) |

## 2. Arms: pose swaps, not animation
- Every arm change takes **2 frames (0.07 s)**, range 2–4. The pose **snaps** from one drawing to the next, with no smooth in-between.
- **Hold:** median **3.3 s** per pose. That's 13 changes in about 47 s of talking, roughly **one every 3.5 s** (calmer than the previous reference's 2 s).
- **Pose library** (6 drawings):

| # | Pose | Seen at (words) |
|---|---|---|
| 1 | Arms down, relaxed (rest) | start, "eight, you can find one", "simple" |
| 2 | Elbows bent, hands at hip height, palms in | "street truck that can gap" |
| 3 | **One finger pointing up** (right hand), other arm down | "this is the…", "which is crazy", "power and five hundred" |
| 4 | **Both palms up**, shrug / presenting | "foundation is", "Chevy Silverado", "roast tires" |
| 5 | One hand out to the side, palm up; other hand at the hip | "it's the perfect…", "you're basically looking" |
| 6 | Hands slightly open at the sides | transitions |

- **Timing:** the change lands on the **start of a key phrase** ("this is the one", "the foundation is", "which is crazy", "you're basically…").

## 3. Head
- Sideways range 27 px (std 7 px), vertical 13 px (std 2 px), frame-to-frame jitter only 0.35 px, so movement is smooth.
- **Tilt:** std 3°, and |tilt| > 3° on 26% of frames. The head **tilts more than before** and holds the tilt for a while.
- The head moves; the body doesn't.

## 4. Mouth
- **Hold per shape:** mostly 1–5 frames, mean 3.6 frames (**120 ms**).
- **Shapes** follow the sounds of the words: wide teeth grin (E/S/T), open with tongue (A/O), round small "o" (OO/W), closed smile.
- In pauses the mouth returns to the **teeth grin / closed smile**, never fully neutral.

## 5. Eyes and expression
- **Blinks are rare:** about one every **8 s**, lasting 3–6 frames.
- **Expressions:** happy squint (eyes closed into arcs) on jokes; wide eyes and a big open mouth on surprising numbers ("eighteen", "five hundred").

## 6. Edit
- The same parts corner mode as before: the character shrinks to the bottom-right while parts appear (29–38 s).

## 7. Rules for our videos (30 fps)
1. **Body and legs:** completely static, with no sway and no breathing.
2. **Arms:** snap to a new pose drawing in 2 frames, on the first word of a key phrase, about **every 3.5 s**. No smooth in-between, no overshoot.
3. **Head:** slow drift of about ±13 px and tilt of up to ±4°, changing between phrases; never jittery.
4. **Mouth:** phoneme shapes, 3–4 frames each, returning to the grin in pauses.
5. **Eyes:** a blink every 6–9 s; a happy squint on jokes; wide eyes on big numbers.

## 8. What the new character needs
Because the arms **swap between drawings**, the most faithful result comes from **pose images of the same character**. I can rig a single image, but snapping a rigged arm looks less natural than swapping real drawings.
