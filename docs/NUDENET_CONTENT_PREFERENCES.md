# NudeNet content preferences

## Current menu

The content-preferences menu uses **only NudeNet 3.4.2's actual 18 detector labels**, not inferred garment/activity categories or nonfunctional disabled controls.

Separate opt-in [local contextual-nudity discovery](LOCAL_CONTEXTUAL_NUDITY.md) now runs alongside this detector when enabled. It is not presented as an additional NudeNet label and does not change these 18 controls.

Authoritative upstream label list: [NudeNet v3 source](https://github.com/notAI-tech/NudeNet/blob/v3/nudenet/nudenet.py). The exact IDs and readable/grouped names live in `app/vision/nudenet_labels.py`.

| Menu group | Exact detector labels |
| --- | --- |
| Exposed anatomy (6) | `BUTTOCKS_EXPOSED`, `FEMALE_BREAST_EXPOSED`, `FEMALE_GENITALIA_EXPOSED`, `MALE_BREAST_EXPOSED`, `ANUS_EXPOSED`, `MALE_GENITALIA_EXPOSED` |
| Covered anatomy (5) | `FEMALE_GENITALIA_COVERED`, `BELLY_COVERED`, `ANUS_COVERED`, `FEMALE_BREAST_COVERED`, `BUTTOCKS_COVERED` |
| Other detector labels (7) | `FACE_FEMALE`, `FEET_EXPOSED`, `FEET_COVERED`, `ARMPITS_COVERED`, `ARMPITS_EXPOSED`, `FACE_MALE`, `BELLY_EXPOSED` |

The layout provides grouped disclosure panels, aligned label/action rows, readable names with exact model IDs, per-group actions, a separate audio-default control, and responsive desktop/mobile styling. Other labels are initially collapsed because most users should leave faces/feet/armpits allowed. There are no room, bikini, low-cut-shirt, violence or medical options in this menu: NudeNet cannot distinguish those semantic categories.

`Allow` does not propose windows for that label. `Blur` uses the global audio default. `Blur + mute` requests both. Individual scene overrides still take precedence. Defaults retain the six previously blocked exposure labels; covered and other labels are allowed.

## Functional behavior

- Selected labels drive coarse scoring, dense refinement and raw-evidence action resolution. These are not cosmetic dropdowns over the old exposure-only score.
- Selection is scoped with a context variable, so concurrent scans cannot leak settings into one another.
- Remote v2 scores are recomputed from raw detections using the current selection. Legacy scalar-only remote scan results fall back to extraction/per-frame raw classification rather than claiming coverage of covered/facial/etc. labels.
- Resume identities include the selected proposal-label set. Raw detection evidence survives checkpoint-prefix rewrites and is rescored on resume. Legacy scalar-only checkpoints cannot resume as label-aware scans.
- Each scene retains raw per-label confidence, sampling threshold and detector provenance. Covered anatomy is never reinterpreted as a bikini, underwear or cleavage.
- Enabling a label that did not previously propose windows requires a fresh scan. Disabling an existing matched label can update existing effective actions without model calls. A title's coverage records the labels that actually proposed windows, not every class the model could theoretically return.
- A late scan cannot publish if the set of enabled labels changed during its run. The activation transaction checks again while holding the database writer reservation.
- The older semantic yes/no verifier must not veto a selected NudeNet class merely because it considers covered swimwear acceptable. NudeNet scans now use raw label evidence directly.

## Compatibility and privacy

Policy schema 3 retains schema 2's namespaced `nudenet.*` IDs and adds a separate `nudity.contextual` action. Schema-1/2 policies upgrade on validation/read, preserving existing semantic rules, NudeNet preferences and reviewed evidence for compatibility. Existing exposed-anatomy preferences map to the equivalent raw labels; clothing preferences are **not** guessed into covered-anatomy rules. No destructive DB or media migration is performed.

The redesigned form selects **local NudeNet scanning** when saved. If the previous external semantic backend was enabled, a visible notice explains the switch; its old rules/evidence remain stored, but semantic controls are no longer offered in this menu. The server retains compatibility with earlier settings imports and existing evidence. NudeNet-only scans do not need a vision provider or send frames to one. Enabling independent contextual discovery additionally sends sampled frames to the explicitly configured local vision server. Existing explicitly configured remote NudeNet workers remain supported.

Saving never regenerates a library. Review/re-scan the intended title as indicated, then explicitly regenerate. Previously published edits and source-preservation/clean-audio/mux safeguards remain unchanged.

## Original NudeNet rollout verification (historical)

- `python -m pytest tests/ -q`: **432 passed, 12 skipped**, 5 dependency warnings.
- `SPF_TEST_RENDER_CONTAINERS=vulgarr-worker-worker-0-1,vulgarr-worker-worker-1-1 python -m pytest tests/ -q`: **456 passed**, 5 dependency warnings.
- Chrome checks at a 1,074px desktop viewport and an emulated **390px** mobile viewport found no overlapping visible rows; group and individual controls correctly updated the submitted policy. All 18 exact labels were present.
- New tests cover all 18 scoring/action paths, concurrent selection isolation, schema-1 compatibility, covered-label scene proposals without a semantic verifier, threshold handling and raw/legacy checkpoint behavior.

These checks verify wiring and rendering, not model accuracy on user media. NudeNet still makes mistakes and temporal sampling can miss short shots. No new dependencies, worker deployment, service restart, or user-media regeneration were needed.
