"""
PLACEHOLDER knowledge base for STROKE-LEVEL biomechanics coaching -- see
rag/kb_placeholder.py for the session-level equivalent and the same
"authored, not sourced" caveat, which applies identically here: these
snippets are HAND-AUTHORED BY US, NOT sourced from the ITF Coaching
Manual, USTA guidelines, or any other copyrighted publication.

Scoped ONLY to the 7 fields ir/DESIGN.md's final usability table marked
"validated at scale and percentile-ranked" or "percentile-ranked":
wrist_height_ratio, weight_transfer_ratio, knee_flex (left/right, one
topic), stance_width_ratio, follow_through_extension_ratio,
elbow_angle_deg, shoulder_hip_separation_deg. Deliberately NO entries for
shoulder_rotation_deg / hip_rotation_deg / forearm_orientation_deg -- those
3 fields are excluded from percentile ranking entirely (camera-facing-
direction artifact, see ir/biomechanics_percentiles.py), so there is
nothing groundable to say about them yet; adding KB content for a field
that can never be cited would just be dead retrieval bait.

IMPORTANT -- elbow_angle_deg and shoulder_hip_separation_deg carry a
FALSIFIED small-sample pattern (see ir/DESIGN.md's full-dataset run: the
5-example forehand/backhand elbow-angle contrast and the "serve has low
X-factor" impression were both contradicted by the real population). Their
entries below are written GENERICALLY about what the feature measures and
stay deliberately honest that no fixed technique claim can be made for
either -- because we now know the specific-pattern claims would be wrong.
Do NOT add a "try X" fix to these two topics; say what was measured and
that it's normal variation, not something to act on from the number alone.

Percentile framing dropped from player-facing text (2026-08-01 rewrite,
see below) -- the underlying data is still percentile-ranked against our
own sampled population (ir/biomechanics_percentiles.py), unchanged, but
that framing is no longer stated to the player using the words
"ratio"/"percentile" themselves.

REWRITTEN (2026-08-01) for player-facing tone, per real-user testing
feedback -- same rules and same verification as rag/kb_placeholder.py's
rewrite (see that file's docstring and PROJECT_LOG.md for the full
rationale, the grounding-mechanics safety check, and the re-run
verification). The one biomechanics-specific rule carried over from the
original file: elbow_angle_deg and shoulder_hip_separation_deg stay
non-committal by design, not by oversight -- see the IMPORTANT note above.
"""
from dataclasses import dataclass

from rag.kb_placeholder import KBSnippet


BIOMECHANICS_KB = [
    # --- wrist_height_ratio ---
    KBSnippet(
        "wrist_height_ratio",
        "Where you make contact with the ball naturally changes by shot, and "
        "yours looks unusual for this one -- that's not automatically wrong, "
        "some players do it on purpose. If it feels rushed or uncomfortable "
        "though, try meeting the ball a touch earlier or later. (contact height: "
        "unusual for this shot)",
    ),
    KBSnippet(
        "wrist_height_ratio",
        "A lower contact point than usual for this shot could mean you're taking "
        "the ball later than normal, or it could just be your natural, flatter "
        "swing style -- this measurement alone can't tell which. If it doesn't "
        "feel like a problem to you, there's nothing here that says it is.",
    ),
    # --- weight_transfer_ratio ---
    KBSnippet(
        "weight_transfer_ratio",
        "How much your weight shifts forward into the shot -- from your back "
        "foot toward the ball -- shows up here. If yours looks light, try "
        "driving more off your back foot as you swing, so your weight is moving "
        "into the shot at contact rather than staying back. One caveat: this "
        "reading's direction can vary with how your camera was angled, so treat "
        "it as a starting point, not a certainty.",
    ),
    KBSnippet(
        "weight_transfer_ratio",
        "An unusually low weight shift into the shot isn't automatically a flaw "
        "-- it can just mean a more neutral, reactive stance, which is normal on "
        "defensive shots. Treat this as a talking point rather than something to "
        "fix outright.",
    ),
    # --- knee_flex (left + right) ---
    KBSnippet(
        "knee_flex",
        "How bent your knees are right at the moment of contact -- most "
        "players' legs are close to straight at that exact instant across all "
        "shot types, so this number on its own doesn't tell us much that's "
        "specific to you. There's nothing to change based on this measurement "
        "alone.",
    ),
    KBSnippet(
        "knee_flex",
        "If your knees are noticeably more bent than usual at the moment of "
        "contact, that can be a sign of a later or more compact leg drive. Try "
        "starting your leg drive a touch earlier, so your legs are straightening "
        "as you make contact rather than still loading.",
    ),
    # --- stance_width_ratio ---
    KBSnippet(
        "stance_width_ratio",
        "A narrower stance than usual for you at contact can mean a less stable "
        "base -- try setting your feet a little wider before you strike the "
        "ball. A wider stance than usual isn't automatically a problem though; "
        "that can just be a deliberate wide split-step or lunge.",
    ),
    KBSnippet(
        "stance_width_ratio",
        "This is measured at just one instant of your swing, not your footwork "
        "through the whole stroke. Don't read too much into it on its own -- "
        "look at it alongside your other results, not by itself.",
    ),
    # --- follow_through_extension_ratio ---
    KBSnippet(
        "follow_through_extension_ratio",
        "If your swing stops moving soon after contact, that's a short "
        "follow-through -- normal on compact shots like volleys, so not "
        "automatically a problem there. On groundstrokes though, try letting "
        "your racket keep travelling out toward the target after you hit the "
        "ball, instead of pulling up short.",
    ),
    KBSnippet(
        "follow_through_extension_ratio",
        "A much shorter follow-through than usual for you can mean you're "
        "checking your swing early -- try letting your arm continue through the "
        "shot instead of stopping right at contact. One caveat: this is only "
        "measured to the end of the filmed clip, so a swing still in motion "
        "when the video cuts off can look shorter than it really is.",
    ),
    # --- elbow_angle_deg ---
    # NOTE: do NOT add a "try X" technique fix here (see module docstring) --
    # the one striking pattern seen on 5 examples (forehand vs. backhand
    # contrast) was contradicted by the full 1980-clip population.
    KBSnippet(
        "elbow_angle_deg",
        "How bent your arm is at the moment you hit the ball varies a lot "
        "between players even on the same shot -- there's no single 'correct' "
        "angle to aim for. If this looks unusual for you, it's worth a coach's "
        "eye on video rather than something to fix from a number alone.",
    ),
    KBSnippet(
        "elbow_angle_deg",
        "This number alone can't tell you if your arm bend is a problem -- it "
        "varies too much between players to have one right answer. Treat an "
        "unusual reading as a cue to get a coach's eyes on your video, not proof "
        "something's wrong.",
    ),
    # --- shoulder_hip_separation_deg ---
    # NOTE: do NOT claim serve (or any class) has a distinctively low/high
    # separation here -- the 5-example impression that serve was lowest was
    # contradicted by the full population (serve is actually one of the
    # highest class means).
    KBSnippet(
        "shoulder_hip_separation_deg",
        "How much your upper body has turned relative to your hips at contact "
        "is what this measures -- sometimes called your 'coil.' An unusual "
        "reading here is worth a conversation about timing with a coach, not "
        "something to grade as pass or fail.",
    ),
    KBSnippet(
        "shoulder_hip_separation_deg",
        "This is measured from a single estimated instant in your swing, not "
        "tracked continuously -- so treat the number as a rough signal, not an "
        "exact reading. It's not something to chase precision on by itself.",
    ),
]
