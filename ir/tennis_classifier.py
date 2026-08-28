"""
Zero-shot "is this actually tennis footage" gate (2026-08-12), CLIP-based
(open_clip_torch, ViT-B-32 / laion2b_s34b_b79k -- a standard, well-tested
pretrained combo, no fine-tuning). Runs server-side in backend/app.py, right
after decode_and_resample_video() produces its resampled frames and BEFORE
the expensive MediaPipe pose extraction / StrokeGRU / LLM coaching stages --
reject cheaply before paying for the rest of the pipeline on content that
was never going to be a valid stroke analysis to begin with.

Standard CLIP zero-shot classification: encode a fixed set of text prompts
(a positive "tennis" group plus several negative groups covering the
confusable near-neighbors, not just generic scenes), encode a handful of
sampled video frames, and for each frame compute a softmax over ALL prompts'
cosine similarities -- the "tennis_score" is the summed probability mass
landing on the positive group, averaged across sampled frames.

THRESHOLD_STATUS: LOCKED at 0.5 (2026-08-12). See PROJECT_LOG.md's
2026-08-12 entries for the full validation writeup and history --
summarized here so the gap stays visible without having to cross-reference
the log:

  Validated against (ir/run_tennis_classifier_validation.py, real footage,
  run through the real decode_and_resample_video() pipeline path):
    - 2 real tennis clips (clip1_serve_practice 0.99, clip2_teen_practice
      0.91) -- positive side. A 3rd (clip3_group_distant) couldn't be
      tested at all: it's rejected by the PRE-EXISTING MAX_RAW_FRAMES check
      before reaching this classifier.
    - 5 real non-tennis clips from Penn Action (a multi-sport action
      dataset already in this repo), chosen to include the closest
      plausible confusable case -- a golf swing (0.21, the closest real
      negative found so far) -- plus baseball_swing (0.001), bowl (0.073),
      strum_guitar (0.37), jumping_jacks (0.063). 7/7 correct at
      threshold=0.5, with real margins: ~0.29 headroom below threshold on
      the closest negative, ~0.41 headroom above threshold on the weakest
      positive.

  Explicitly NOT validated, and this gap is real, not hidden:
    - No adversarial/edited video content (e.g. a movie premiere, red-carpet
      event footage, or other produced/non-amateur-sport footage) was
      tested -- only real, unedited physical-activity footage (tennis vs.
      other real sports/exercise). A movie premiere clip (crowds, camera
      flashes, produced lighting) is a materially different visual
      distribution from anything in this validation set, and this
      classifier's real-world performance against that KIND of content is
      unknown, not just untested-on-this-one-clip.
    - A specific "ball going over a net with no player visible" case was
      considered and deliberately excluded from this classifier's scope,
      not tested and passed -- that's a different, ALREADY-HANDLED failure
      mode (backend/app.py's existing no_player_detected check), not
      something this tennis-content gate needs to catch too.
    - No systematic sweep across the threshold range was done -- 0.5 is
      the single value validated above, not the result of an ROC-style
      search for an optimal cutoff.
"""
import numpy as np
import open_clip
import torch

MODEL_NAME = "ViT-B-32"
PRETRAINED = "laion2b_s34b_b79k"
N_SAMPLE_FRAMES = 4  # cheap on GPU; no need to run all 60 resampled frames for a confirming sample

# LOCKED 2026-08-12 -- see module docstring THRESHOLD_STATUS for what this
# was and wasn't validated against before locking it.
DEFAULT_THRESHOLD = 0.5

POSITIVE_PROMPTS = [
    "a photo of a person playing tennis",
    "a tennis player swinging a racket on a tennis court",
    "a person practicing a tennis stroke",
]

# Deliberately includes the closest visual/semantic neighbors (other racket
# sports), not just generic non-sport scenes -- a classifier that only has to
# beat "an empty hallway" isn't proving much.
NEGATIVE_PROMPTS = [
    "a photo of a person playing badminton",
    "a photo of a person playing table tennis",
    "a photo of a person playing squash",
    "a photo of people playing basketball",
    "a photo of people playing soccer",
    "a photo of an empty room or hallway",
    "a person walking or standing, not playing any sport",
]

_state = {}


def _load():
    """Lazy singleton -- loads the model + text embeddings once per process,
    not per request. Uses CUDA if available (already the case for this
    project's torch install), falls back to CPU otherwise."""
    if "model" in _state:
        return
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _, preprocess = open_clip.create_model_and_transforms(MODEL_NAME, pretrained=PRETRAINED)
    tokenizer = open_clip.get_tokenizer(MODEL_NAME)
    model = model.to(device).eval()

    all_prompts = POSITIVE_PROMPTS + NEGATIVE_PROMPTS
    with torch.no_grad():
        text_tokens = tokenizer(all_prompts).to(device)
        text_features = model.encode_text(text_tokens)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    _state["model"] = model
    _state["preprocess"] = preprocess
    _state["device"] = device
    _state["text_features"] = text_features
    _state["n_positive"] = len(POSITIVE_PROMPTS)


def _sample_frames(frames: list[np.ndarray], n: int) -> list[np.ndarray]:
    if len(frames) <= n:
        return frames
    idx = np.linspace(0, len(frames) - 1, n).round().astype(int)
    return [frames[i] for i in idx]


def classify_frames(frames: list[np.ndarray], threshold: float = DEFAULT_THRESHOLD) -> dict:
    """frames: list of RGB uint8 arrays (same format decode_and_resample_video()
    produces). Returns {"is_tennis": bool, "tennis_score": float, "threshold":
    float, "n_frames_sampled": int} -- tennis_score is the positive-group's
    average softmax probability mass across the sampled frames, in [0, 1]."""
    _load()
    model, preprocess, device = _state["model"], _state["preprocess"], _state["device"]
    text_features = _state["text_features"]
    n_positive = _state["n_positive"]

    sampled = _sample_frames(frames, N_SAMPLE_FRAMES)
    from PIL import Image
    images = torch.stack([preprocess(Image.fromarray(f)) for f in sampled]).to(device)

    with torch.no_grad():
        image_features = model.encode_image(images)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        logits = 100.0 * image_features @ text_features.T  # CLIP's standard temperature scaling
        probs = logits.softmax(dim=-1)
        positive_mass = probs[:, :n_positive].sum(dim=-1)

    tennis_score = float(positive_mass.mean().item())
    return {
        "is_tennis": tennis_score >= threshold,
        "tennis_score": tennis_score,
        "threshold": threshold,
        "n_frames_sampled": len(sampled),
    }
