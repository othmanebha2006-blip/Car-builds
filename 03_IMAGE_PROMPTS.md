# Image Prompts for ChatGPT

## How to generate (read once)
1. **One ChatGPT chat per car.** Paste the prompts in order in the same chat so the car stays the same colour and style.
2. Every image should have a **plain white background with no text**. I remove the background and add all the text myself.
3. If ChatGPT puts text, a road, or a cut-off car in an image, reply: *"Same image but plain white background, no text, whole car visible."*
4. **Download each image and name it exactly** as in the list (e.g. `stinger_01_front34.png`), then upload them all here.
5. **Logos:** search Google Images for e.g. "Kia logo png transparent" and upload one per brand (`logo_kia.png`).

### Style line (already included in every prompt)
`photorealistic studio render, plain pure white background, no text, no watermark, no shadow, entire car visible and centered, landscape 3:2`

---

## Shared parts: make ONCE, reused in every video
Use a new chat. These work for all 5 cars.

| File name | Prompt |
|---|---|
| `part_downpipe.png` | Photorealistic studio photo of a single stainless steel aftermarket turbo downpipe exhaust pipe, isolated object, plain pure white background, no text, no watermark, centered, square 1:1 |
| `part_intake.png` | Photorealistic studio photo of a single black cone performance air intake filter, isolated object, plain pure white background, no text, no watermark, centered, square 1:1 |
| `part_intercooler.png` | Photorealistic studio photo of a single black aluminium front-mount intercooler core, isolated object, plain pure white background, no text, no watermark, centered, square 1:1 |
| `part_chargepipe.png` | Photorealistic studio photo of a single polished aluminium turbo charge pipe with silicone couplers, isolated object, plain pure white background, no text, no watermark, centered, square 1:1 |
| `part_headers.png` | Photorealistic studio photo of a single set of stainless steel long tube V8 exhaust headers, isolated object, plain pure white background, no text, no watermark, centered, square 1:1 |
| `part_camshaft.png` | Photorealistic studio photo of a single performance engine camshaft, isolated object, plain pure white background, no text, no watermark, centered, square 1:1 |
| `part_heatexchanger.png` | Photorealistic studio photo of a single aluminium car heat exchanger radiator, isolated object, plain pure white background, no text, no watermark, centered, square 1:1 |
| `part_pulley.png` | Photorealistic studio photo of a single black aluminium supercharger pulley, isolated object, plain pure white background, no text, no watermark, centered, square 1:1 |
| `part_tune.png` | Simple flat black line icon of a car engine control unit (ECU chip) with a small lightning bolt, minimal, plain pure white background, no text, centered, square 1:1 |
| `part_e85.png` | Flat icon: a white circle with a thick black ring border, empty in the middle, plain pure white background, no text, square 1:1 *(I'll add "E85"/"E30" on top)* |

---

## Per-car images (8 per car)

Paste **#1 first**. For #2 onwards, ChatGPT already knows the car, so the prompts say "the same car".

### Template
| # | File | Prompt |
|---|---|---|
| 1 | `CAR_01_front34.png` | Photorealistic studio render of a **[COLOUR] [YEAR] [MAKE MODEL]**, **[STYLE]**, front three-quarter view, plain pure white background, no text, no watermark, no shadow, entire car visible and centered, landscape 3:2 |
| 2 | `CAR_02_rear34.png` | The exact same car, rear three-quarter view, same style, plain pure white background, no text, entire car visible, landscape 3:2 |
| 3 | `CAR_03_side.png` | The exact same car, perfect side profile view, plain pure white background, no text, entire car visible, landscape 3:2 |
| 4 | `CAR_04_front.png` | The exact same car, straight-on front view, plain pure white background, no text, entire car visible, landscape 3:2 |
| 5 | `CAR_05_rear.png` | The exact same car, straight-on rear view, plain pure white background, no text, entire car visible, landscape 3:2 |
| 6 | `CAR_06_hoodopen.png` | The exact same car, front view with the hood open showing the engine bay, plain pure white background, no text, entire car visible, landscape 3:2 |
| 7 | `CAR_07_engine.png` | Photorealistic studio photo of the **[ENGINE]** engine on its own, removed from the car, isolated, plain pure white background, no text, centered, square 1:1 |
| 8 | `CAR_08_front34_low.png` | The exact same car, low aggressive front three-quarter angle, plain pure white background, no text, entire car visible, landscape 3:2 |

### Fill-ins for each video
**A/B/C = the 3 production videos (`04_PRODUCTION_SCRIPTS.md`): make these first.** CTS-V and V-Sport also need `part_pulley.png` and `part_headers.png`.
| Video | CAR prefix | COLOUR | YEAR MAKE MODEL | STYLE | ENGINE | Logo file |
|---|---|---|---|---|---|---|
| **A** | `ctsv` | gloss black | 2017 Cadillac CTS-V sedan | lowered, aggressive body kit, black wheels, red brake calipers | Cadillac LT4 6.2 litre supercharged V8 | `logo_cadillac.png` |
| **B** | `vsport` | gloss black | 2016 Cadillac CTS V-Sport sedan | lowered, rear wing spoiler, black wheels, red brake calipers | Cadillac 3.6 litre twin-turbo V6 | `logo_cadillac.png` |
| **C** | `arteon` | deep metallic green | 2021 Volkswagen Arteon R-Line | lowered, black wheels, red brake calipers | Volkswagen EA888 2.0 litre turbo four-cylinder | `logo_vw.png` |
| **D** | `g70` | Havana red | 2022 Genesis G70 3.3T sedan | lowered, aggressive body kit, gloss black wheels, red brake calipers | Genesis 3.3 litre twin-turbo V6 | `logo_genesis.png` |
| 1 | `stinger` | deep red | 2022 Kia Stinger GT | lowered, aggressive body kit, gloss black wheels, red brake calipers | Kia 3.3 litre twin-turbo V6 | `logo_kia.png` |
| 2 | `q50` | pearl white | 2019 Infiniti Q50 Red Sport 400 | lowered, carbon lip, bronze wheels | Infiniti VR30DDTT 3.0 litre twin-turbo V6 | `logo_infiniti.png` |
| 3 | `bmw340` | Estoril blue | 2018 BMW 340i F30 sedan | lowered, M Performance kit, black wheels | BMW B58 3.0 litre turbo inline-six | `logo_bmw.png` |
| 4 | `sho` | gloss black | 2017 Ford Taurus SHO | lowered, blacked-out trim, black wheels, police-car look | Ford 3.5 EcoBoost twin-turbo V6 | `logo_ford.png` |
| 5 | `chevyss` | bright red | 2017 Chevrolet SS sedan | lowered, black wheels, subtle lip spoiler | Chevrolet LS3 6.2 litre V8 | `logo_chevrolet.png` |

**Example, Video 1, image #1 (ready to paste):**
> Photorealistic studio render of a deep red 2022 Kia Stinger GT, lowered, aggressive body kit, gloss black wheels, red brake calipers, front three-quarter view, plain pure white background, no text, no watermark, no shadow, entire car visible and centered, landscape 3:2

**Total:** 9 shared parts (once) + 8 images per car + 1 logo per car.
