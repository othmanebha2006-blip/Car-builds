# Voiceover rules (from the user, default for all future videos)

"Cut the spaces" means: remove **only the clear pauses** where the speaker stops. Never touch the parts he reads smoothly.

- Tool: `tools/cut_pauses.py IN.wav OUT.wav` (defaults: -47 dB, pauses >= 0.15 s, shortened to 0.10 s).
- A pause = true silence for at least 0.15 s. Shorter gaps between words are the natural flow: left as they are.
- Each pause keeps 50 ms after the last word and 50 ms before the next, joined with an 8 ms crossfade inside the silence,
  so no word start or end is clipped (no "swallowed" words).
- Do not change the voice: no de-esser, no click repair, no filtering, no loudness change (`"keep_voice": true`).
- The old aggressive `tighten.py` (-40 dB, gaps > 90 ms -> 60 ms) and `clean_vo.py` are not used any more unless asked.
