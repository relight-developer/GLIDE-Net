# Web fonts bundled with the workstation

`app/static/fonts/` ships local TTFs so the UI renders identically without
network access (the demo machine may be offline during the class presentation).
All of them are licensed under the SIL Open Font License 1.1, which permits
bundling and redistribution:

| File | Family | Upstream | License |
|------|--------|----------|---------|
| `BarlowCondensed-SemiBold.ttf` | Barlow Condensed | https://github.com/jpt/barlow | OFL-1.1 |
| `Inter-Regular.ttf`, `Inter-Medium.ttf` | Inter | https://github.com/rsms/inter | OFL-1.1 |
| `JetBrainsMono-Regular.ttf` | JetBrains Mono | https://github.com/JetBrains/JetBrainsMono | OFL-1.1 |
| `SpaceGrotesk-Regular.ttf`, `SpaceGrotesk-Medium.ttf` | Space Grotesk | https://github.com/floriankarsten/space-grotesk | OFL-1.1 |

Fonts are referenced from `app/ui/styles.py` via `@font-face` and served through
Gradio's `allowed_paths` (see `app/main.py`).
