# Sound-effect rules (apply to every video)

| Moment | Sound | File |
|---|---|---|
| A car shot appears | Reference "appear" sound, **never change it** | `assets/sfx/appear_reference.wav` (level: 1.139 x active speech RMS, starts 15 ms before the frame) |
| Shine flash on the car | Reference shine sound | `assets/sfx/shine_reference.wav` |
| Parts, upgrades, numbers, prices, labels | The pop-up from the user's clip (rising pop, 47 ms), on the frame the item appears, 1.18 x speech; never the car sound | `assets/sfx/popup_reference.wav` (fixed) |
| Typewriter text (years, engine) | The reference typing recording, from the first letter until the typing stops (30 ms fade), 0.48 x speech, sped up x1.25 | `assets/sfx/typing_reference.wav` (from the references, fixed) |
| Engine break | Sound of the clip the user sends, never its video; shake only while the sound plays | `projects/<car>/raw/engine_sound.wav` |
| Not allowed | Any noise-based whoosh / riser / hiss | -- |
