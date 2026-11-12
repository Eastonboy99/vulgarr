# Independent local contextual-nudity discovery

## What changes

NudeNet detects body parts, not the general concept of a person being nude with
intimate anatomy hidden. The new optional pass examines the whole title through
an explicitly configured **local vision model**, even when NudeNet finds zero
candidates. It supplements, rather than replaces, the 18 real NudeNet controls.

Two separate functions are now named clearly in Settings:

- **Discover contextual / obscured nudity across the whole video**: independent
  proposal generation, implemented in `app/vision/contextual.py`.
- **Verify existing legacy candidates**: the older yes/no recheck of existing
  windows. Its historical `claude_vision_*` configuration names remain compatible.
  It cannot discover missed windows and does not veto exact NudeNet or contextual
  evidence. A local LM Studio endpoint is local; the old name did not imply that
  an Anthropic/cloud service was necessarily being used. Its key is now optional
  for local servers that do not require authentication.

## Setup

After updating the application through the normal operator-approved process:

1. Run an image-capable model in LM Studio on the trusted local network.
   `qwen3.5-4b` is the default starting identifier and was already downloaded in
   the operator's installation. Qwen3-VL is another candidate; model accuracy is
   not guaranteed. Match the model's actual API identifier, including any custom
   identifier assigned when loading it.
2. Configure model context length and device placement in LM Studio. Start with
   a bounded context such as 8,192 tokens, not the model's maximum. Benchmark GPU
   throughput and memory alongside other workloads before scanning a library.
   CPU-only scans can take many hours.
3. In Settings > Video Processing > Contextual nudity discovery, enter a base URL
   such as `http://DESKTOP_LAN_IP:1234/v1`, the model identifier and an optional
   local API key. The URL initially offers the existing verification URL when
   the new field is empty; keys are **not** implicitly copied across endpoints.
4. Enable discovery and save. Merely saving does not scan or regenerate anything.
5. Run an ordinary fresh scan on the intended movie/episode. The explicit
   **Clear scenes & restart scan** action remains available when immediate scene
   deletion is desired instead of atomic successful replacement.
6. Review previews and candidate boundaries, then approve/reject/adjust and Apply.
   An uncertain candidate with no positive nudity observation has no automatic
   blocking action; choose a manual Blur override if you want it filtered.

`localhost` refers to the application/container making the request, **not** a
separate desktop. LM Studio must permit access from the application host, with
appropriate firewall/authentication restrictions. Do not expose it to the WAN.
The local server must support image inputs, OpenAI-compatible chat completions,
JSON-schema structured output and `reasoning_effort=none`. Unsupported protocols,
missing image projectors or invalid responses fail safely; there is no external
fallback or silent downgrade to unstructured guessing.

### Defaults

| Setting | Default |
| --- | --- |
| `scene_contextual_enabled` | `false` |
| `scene_contextual_base_url` | empty |
| `scene_contextual_api_key` | empty; optional |
| `scene_contextual_model` | `qwen3.5-4b` |
| `scene_contextual_window_seconds` | 8; allowed 4–30 |
| `scene_contextual_timeout_seconds` | 300; allowed 30–900 |

The separate `nudity.contextual` action defaults to Blur + mute; existing audio
and manual action overrides still apply. Discovery must be explicitly enabled.
Turning discovery off stops future contextual scans, not previously approved
scene decisions. Use the action control/review/Apply to change existing edits.

## Sampling and decision semantics

- Four ordered stills per window, at fractions `.125`, `.375`, `.625`, `.875`,
  scaled to a 768-pixel long edge. An eight-second window samples at +1,+3,+5,+7
  seconds. The last window is clipped to actual source duration.
- Every window across the title is visited, independent of NudeNet candidates.
  This is **sampling**, not exhaustive frame-by-frame video understanding. Short
  shots between samples can still be missed.
- The model distinguishes exposed/contextual nudity, towel/sheet coverage,
  swimwear/underwear, clothing, ambiguous bare skin, no people and uncertainty.
  Location or bare shoulders alone are not sufficient evidence by policy, but
  models can still misinterpret them. Inspect the actual preview.
- Input/output frame IDs are validated for exact count and uniqueness. The model
  never supplies trusted timestamps, approval, filtering commands or confidence.
  Binary presence is shown as an **AI flag**, not a misleading 100% confidence.
- Uncertainty is derived from per-frame states, never a model's own review flag.
  Every new contextual candidate requires human approval. It has no invented
  NudeNet verification fraction and cannot enter high-confidence auto-approval.
- Adjacent equivalent contextual windows coalesce. Already-blocking detector
  intervals are subtracted, preserving newly discovered portions without
  duplicate overlaps or changing NudeNet's evidence/approval. Contextual windows
  are not passed through NudeNet's dense verifier: lack of visible anatomy must
  not erase an obscured-nudity proposal.
- Candidate boundaries conservatively cover sampled windows, **not** exact
  observed onset/offset. Review and adjust them. No frame-accurate contextual
  boundary detector is claimed.
- Any pending review blocks automatic Apply, even if some detector scenes are
  auto-approved. Explicit manual Apply continues to operate on approved decisions.

## Privacy, failure and compatibility

Discovery accepts only HTTP(S) loopback, RFC1918 IPv4 or IPv6 unique-local
addresses. Credentials in URLs, query strings, fragments, public/mixed DNS
answers and non-HTTP protocols are rejected. DNS is resolved and validated once
for each request and the connection is pinned to the private address with the
original Host/TLS server name. Redirects and environment proxies are disabled.
A private endpoint could itself proxy to a cloud service: use a trusted local
inference server, not a forwarding proxy, when local-only processing is required.

Calls are sequential and timeout-bounded. Temporary sample images are deleted.
Job-scoped checkpoints contain only normalized evidence and a hashed identity,
not credentials, endpoint URLs or raw images. Resume accepts only a contiguous
compatible prefix; corrupt tails and source/model-identifier/config/prompt/schema
changes invalidate reuse. The server's model identifier is not a weights hash:
start a fresh scan (or use a new model identifier) after replacing weights behind
an unchanged alias. Provider errors are redacted and never interpreted as safe content.

Both passes stage their results before one atomic activation. A failed local
window, cancellation, superseded generation or changed discovery configuration
retains the previous successful scene set. The existing explicit reset's
intentional immediate-deletion semantics remain unchanged. Neither path edits
original media, clean audio masters or the last published edit during scanning.

Policy schema 3 adds `nudity.contextual`; schema-1/2 policies retain existing
rules and raw detector preferences during upgrade. Existing evidence coverage
sets remain unchanged. Settings imports validate the merged local configuration
before writing; old form clients without the new presence marker preserve the
new settings. Coverage/config changes are surfaced as requiring a fresh scan.

## Qualification

- `python -m pytest tests/ -q`: **499 passed, 12 skipped**, 15 dependency warnings.
- With both local diagnostic render containers enabled: **523 passed**, 15 warnings.
- Chrome offline checks at **1,100px** and a real emulated **390px** viewport:
  no horizontal overflow or overlapping discovery controls; no JavaScript errors;
  all 18 NudeNet controls remain present. Changing the contextual action updates
  only its policy key, leaving every NudeNet action unchanged.
- Actual candidate classifier code, including frame extraction, private address
  pinning, HTTPX **0.27.2**, strict output parsing and policy resolution, was tested
  in an isolated worker directory against existing local LM Studio Qwen3.5-4B
  weights on a short operator-authorized *Breaking Dawn – Part 1* source clip.
  The **36:54–37:02** window produced a contextual-nudity candidate with mandatory
  review. With `--gpu off` requested, that call took approximately **33 seconds**.
  Temporary dependencies/code were isolated, removed afterward, and the dedicated
  model instance was unloaded; daemon code and services were not changed.

The real clip result is a limited wiring/candidate-detection check, **not** a
false-positive/recall benchmark. No full movie rescan, frame-accurate scene-boundary
qualification, GPU throughput qualification or broad clothed/swimwear evaluation
was performed. Published playback/HDR acceptance is not newly claimed. No model
weights were downloaded, no movie frames were submitted to cloud vision APIs,
and no application deployment, restart or user-media regeneration was performed.
