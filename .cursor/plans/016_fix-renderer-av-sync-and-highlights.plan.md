---
name: fix-renderer-av-sync-and-highlights
overview: Investigate and fix the renderer’s audio-video sync bug for concatenated clips, and outline improvements so generated highlights better match the requested person and duration.
todos:
  - id: locate-renderer
    content: Locate the Python renderer/preview export pipeline that generates preview_*.mp4 files and identify how it uses ffmpeg or moviepy.
    status: completed
  - id: analyze-sync-pattern
    content: Map the user-reported sync drift to specific concat/timing behaviors in the current implementation and shortlist the most plausible root causes.
    status: completed
  - id: design-sync-fix
    content: Design precise ffmpeg or moviepy concat/trim logic (including timestamps and options) that guarantees aligned audio/video durations.
    status: completed
  - id: person-focus-logic
    content: Review or propose segment scoring logic that prioritizes segments containing Purvik when the prompt requests it.
    status: completed
  - id: validation-plan
    content: Define concrete tests and duration checks to ensure the sync issue does not regress and that person-focused prompts behave as expected.
    status: completed
isProject: false
---

### Goals

- **Primary**: Find and fix why concatenated clips drift out of sync so that a ~60s output actually has tightly synced audio/video at the target duration.
- **Secondary**: Ensure the pipeline can preferentially select segments showing `Purvik` when the prompt asks for highlights focused on him.

### High-level approach

- **Understand current pipeline**: Locate the Python rendering/assembly code that takes selected segments and produces `preview_*.mp4` outputs, including where it invokes `ffmpeg` (CLI or via a library like moviepy/ffmpeg-python).
- **Reproduce & reason about sync drift**: Use the existing `misc/preview_ccc49af968ee.mp4` as a reference to infer likely causes (timestamp handling, rounding, audio vs video timelines, or re-encoding behavior).
- **Design concrete fixes** across three likely classes of issues:
  - **Segment-boundary timing math** (off-by-frames / non-monotonic timestamps / gaps).
  - **ffmpeg filters and concat options** (e.g., using `-vsync`, `-async`, `concat` filter vs. concat demuxer, mismatched sample rates).
  - **Audio resampling or trimming** (ensuring the final muxed timeline ends at the same duration for both streams).
- **Plan improvements to segment selection** so the clips respect the “focused on Purvik” constraint, likely by incorporating face detection/recognition scores into the ranking.

### Investigation steps

- **Locate renderer code**
  - Search for Python files that:
    - Call `ffmpeg` (e.g., `subprocess.run(["ffmpeg", ...])`).
    - Use `moviepy.editor.VideoFileClip`, `concatenate_videoclips`, or `ffmpeg`-related utilities.
  - Likely targets: a `renderer`, `export`, or `preview` module that writes `preview_*.mp4` into a `misc` or cache directory.
- **Map the rendering flow**
  - Identify how segments are represented (start/end times, frame indices, or durations).
  - Trace how those segments are converted into ffmpeg commands/filters or moviepy clip lists.
  - Note whether the pipeline:
    - Uses the **concat demuxer** (text file of parts).
    - Uses the **concat filter** (`[0:v][0:a][1:v][1:a]concat=n=...`).
    - Uses a **high-level library** that hides concat details.
- **Infer error pattern from the preview file**
  - Treat the user-reported behavior as ground truth:
    - Each segment boundary introduces ~0.5–1s delay in audio vs video.
    - The final audio runs ~5s longer than video for a ~50s visual duration.
  - Map this pattern to common ffmpeg pitfalls, e.g.:
    - Using `asetpts`/`setpts` incorrectly.
    - Dropping or duplicating audio frames due to `-vsync` / `-af apad` / `-shortest` / missing `-shortest`.
    - Concatenating mismatched sample rates or channel layouts without explicit resampling, causing drift.

### Likely root-cause hypotheses to check

- **Hypothesis 1 – Timestamp drift at concat**
  - The concat method does not rebase timestamps correctly between segments, causing audio PTS to lag and accumulate offset.
  - Plan to:
    - Inspect how `-avoid_negative_ts`, `-copyts`, `-fflags +genpts`, or `setpts` are used (if at all).
    - Ensure concat is done with a single `ffmpeg` invocation using proper `concat` semantics, not manual offset math.
- **Hypothesis 2 – Using `-async`, `-vsync`, or resampling that stretches audio**
  - Over-aggressive sync options can stretch audio to fit video or vice versa.
  - Plan to:
    - Remove or adjust `-async`, `-vsync`, or `-af` filters that change length.
    - Standardize audio to a single sample rate and channel layout before concat (e.g., `-ar 48000 -ac 2`).
- **Hypothesis 3 – Per-segment trim logic off by up to ~1s**
  - If each segment is trimmed with slightly wrong start/end times (e.g., rounding to whole seconds or frame indices at 24/25/30 fps), error can accumulate.
  - Plan to:
    - Check whether segment boundaries are computed in seconds vs frames.
    - Ensure start/end times passed to ffmpeg’s `-ss`/`-to` (or `trim` filter) match the model’s segment boundaries precisely (float seconds), and are consistently applied to both audio and video.

### Concrete fixes for the renderer (once we see the code)

- **Standardize segment preparation**
  - For each source clip:
    - Re-encode to a **uniform video fps** (e.g., 30 fps) and **uniform audio format** (e.g., 48kHz stereo) in a pre-processing step (or on-the-fly in the concat graph).
    - Trim both audio and video together using a single filter chain per segment (`trim`/`atrim` with matching durations).
- **Use robust concat strategy**
  - Prefer one of the following patterns (chosen after seeing the current code):
    - **Concat demuxer**: pre-generate a `segments.txt` file with `file 'segmentN.mp4'` and consistent codecs/params, then run `ffmpeg -f concat -safe 0 -i segments.txt -c copy ...`.
    - **Concat filter**: keep work in one ffmpeg command with `[v0][a0][v1][a1]concat=n=N:v=1:a=1` after trimming / scaling.
  - Avoid multiple re-encode/concat passes that can introduce rounding error.
- **Clamp final duration & ensure A/V alignment**
  - Use `-shortest` if needed so that if any minor drift remains, the longer stream is cropped to the shorter.
  - Optionally add a **final pass** that:
    - Rewrites timestamps with `setpts=PTS-STARTPTS` for video and `asetpts=PTS-STARTPTS` for audio.
    - Verifies output duration equals the sum of segment durations within a small epsilon.

### Improving “focus on Purvik” in highlights

- **Understand current segment selection**
  - Locate where segments are scored/selected based on transcript, embeddings, or vision signals.
  - Identify whether face detection/identification is already available, or if selection is purely text-based.
- **Integrate face/person-focus scoring**
  - If you already compute face tracks or person IDs, incorporate a `purvik_presence_score` into the segment ranking, for example:
    - Probability that Purvik’s face appears in the segment.
    - Proportion of frames in the segment where he is present.
  - If there is no face pipeline yet, plan a minimal viable addition (e.g., a lightweight face detector/recognizer run over candidate segments only).
- **Update highlight ranking logic**
  - When the prompt asks for “focused on Purvik”:
    - Increase the weight of `purvik_presence_score` relative to general semantic relevance.
    - Optionally hard-filter out segments where Purvik is not present at all, unless you need B-roll.

### Validation & regression checks

- **Unit-level checks**
  - For a small set of short segments (2–3 clips), verify:
    - Sum of per-segment durations == output duration within a tiny epsilon (e.g., < 50ms).
    - Audio waveform visibly aligns with lip movement at each boundary.
- **End-to-end preview test**
  - Re-run the preview generation for the same asset/prompt:
    - Confirm the final output is ~60s and that audio ends exactly when video does.
    - Visually inspect that segments with Purvik are dominant in the timeline.
- **Guardrails for future changes**
  - Add simple assertions/logs in the renderer:
    - Log segment start/end times and cumulative expected duration.
    - Log final ffmpeg-reported duration and A/V stream lengths.
    - Warn if drift exceeds a small threshold.

### Next information needed from you

- **Confirm renderer entrypoint**: Once we locate the Python module or function that creates the preview (`renderer`, `export_preview`, etc.), we will focus the detailed fix plan on that concrete code.
- **Tolerance for re-encoding**: Since this is a PoC, we’ll favor correctness over speed and are willing to add an extra encode pass if that simplifies robust sync.

