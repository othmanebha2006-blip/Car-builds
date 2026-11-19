# Car Image Style (default for all new videos)

The style is taken from the 3 reference pictures (Mercedes C-Class, Mitsubishi Galant, Mitsubishi 3000GT). It works on any car: only `[CAR]` changes.

## Style elements
| Element | Value |
|---|---|
| Paint | Gloss black, deep reflections |
| Wheels | **Gloss black** multi-spoke, **red brake calipers** |
| Stance | **Natural street stance**: only a slight drop, wheels not tucked (ref: `assets/style_refs/ref_silverado_street.jpg`) |
| Aero | Aggressive street kit: wide fender flares, front splitter with red tow hooks, hood scoop, side skirts |
| Windows | Dark tint |
| Art style | **Digital illustration with photoreal shading**: clean thin dark outlines around the body, smooth gradients, slightly cartoon-like (not a raw photo) |
| Background | Plain pure white (I remove it and put the grey background and logo behind it) |

## Style line (paste at the end of every car prompt)
`, gloss black paint, gloss black multi-spoke wheels with red brake calipers, aggressive street body kit with wide fender flares, front splitter with red tow hooks, hood scoop and side skirts, sporty natural street stance (only a slight drop, wheels not tucked into the fenders), dark tinted windows, digital car illustration with photorealistic shading and clean thin dark outlines, plain pure white background, no text, no logo watermark, no shadow, entire car visible and centered, landscape 3:2`

## The 8 car prompts (one ChatGPT chat per car)
Replace `[CAR]` with the year and model, for example `2012 Mercedes-Benz C-Class C63 AMG sedan` or `1999 Mitsubishi Galant VR-4`.

1. **`car_01_front34.png`**: `[CAR]`, front three-quarter view + style line
2. **`car_02_rear34.png`**: The exact same car and style, rear three-quarter view + style line
3. **`car_03_side.png`**: The exact same car and style, perfect side profile view + style line
4. **`car_04_front.png`**: The exact same car and style, straight-on front view + style line
5. **`car_05_rear.png`**: The exact same car and style, straight-on rear view + style line
6. **`car_06_hoodopen.png`**: The exact same car and style, front view with the hood open showing the engine bay + style line
7. **`car_07_engine.png`**: The engine of `[CAR]` on its own, removed from the car, isolated, same digital illustration style with photorealistic shading and clean thin dark outlines, plain pure white background, no text, centered, square 1:1
8. **`car_08_front34_low.png`**: The exact same car and style, low aggressive front three-quarter angle + style line

**Full example (#1, Mercedes):**
> 2012 Mercedes-Benz C63 AMG sedan, front three-quarter view, gloss black paint, gloss black multi-spoke wheels with red brake calipers, aggressive street body kit with wide fender flares, front splitter with red tow hooks, hood scoop and side skirts, sporty natural street stance (only a slight drop, wheels not tucked into the fenders), dark tinted windows, digital car illustration with photorealistic shading and clean thin dark outlines, plain pure white background, no text, no logo watermark, no shadow, entire car visible and centered, landscape 3:2

## Parts in the same style
`Single [PART], isolated object, digital illustration with photorealistic shading and clean thin dark outlines, plain pure white background, no text, no logo, centered, square 1:1`

## Tips
- If ChatGPT changes the wheels or slams the car, reply: *"same car, keep the black wheels, red calipers, body kit and natural stance exactly as image 1"*.
- Keep all 8 images in **one chat** so the car stays identical.
