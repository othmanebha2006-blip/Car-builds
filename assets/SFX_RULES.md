# Sound-effect rules (apply to every video)

| Moment | Sound | File |
|---|---|---|
| A car shot appears | Reference "appear" sound, **never change it** | `assets/sfx/appear_reference.wav` (level: 1.139 x active speech RMS, starts 15 ms before the frame) |
| Shine flash on the car | Reference shine sound | `assets/sfx/shine_reference.wav` |
| Parts, numbers, prices, labels | Own tonal pop + tick (`sfx_part` in the renderer), never the car sound | generated, pure tones |
| Typewriter text (years, engine) | The reference typing recording, from the first letter until the typing stops (30 ms fade), 0.40 x speech | `assets/sfx/typing_reference.wav` (from the references, fixed) |
| Engine break | Sound of the clip the user sends, never its video; shake only while the sound plays | `projects/<car>/raw/engine_sound.wav` |
| Not allowed | Any noise-based whoosh / riser / hiss | -- |
