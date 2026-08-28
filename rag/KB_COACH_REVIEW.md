# VisioTennis Coaching Knowledge Base — Coach Review

**Requesting:** a tennis coach's review of every piece of content our app is
currently allowed to draw on when generating coaching feedback for a player.

**What we need from you:** for each entry below, three short judgements
(explained under "How to fill this out"), plus a few whole-set questions at
the very end. There's no need to touch the code or the app itself — this is
a pure content review, on paper or in this document.

---

## Four things to know before you start

**1. This content is authored by us, not sourced from a coaching manual.**
Every sentence below was written by the project team, informed by general
tennis knowledge, not drawn from the ITF Coaching Manual, USTA guidelines,
or any other published coaching text. Nothing here should be read as
carrying outside coaching authority — that's exactly what we're asking you
to check for. If something is wrong, incomplete, or would sound naive
coming from a real coach, that's precisely the kind of finding this review
is for.

**2. There are two different kinds of content here, and the second kind
needs your particular judgement.**
- **Section A** is built from session-level match statistics (serve
  percentages, unforced errors, movement speed, etc.) — the kind of thing
  a coach already tracks informally.
- **Section B** is built from stroke-level *biomechanical measurements* —
  e.g. how high the player made contact on a given shot, measured against
  our own sampled population of clips. This is a newer, more experimental
  kind of framing, and it's the area we most want scrutinized: is a raw
  body-pose measurement, expressed as "unusual for this player" or "bottom
  of the range," useful and honest, or does it risk implying more
  precision or authority than a single-camera, geometry-derived estimate
  actually has? Your read on this framing matters more than any individual
  entry's wording. (The underlying numbers are still computed as
  percentiles against our sampled population — unchanged — we've just
  stopped saying "percentile" to the player; see point 4.)

**3. Two biomechanics entries are deliberately non-committal, on purpose —
not an oversight.**
Early in development, a small 5-example check suggested some specific
patterns (e.g. "servers show unusually low shoulder-hip separation").
Running the same measurement across the full dataset (1,980 clips)
*contradicted* both of those specific patterns. Rather than leave wrong
claims in place, the `elbow_angle_deg` and `shoulder_hip_separation_deg`
entries (E35–E38) were written to explain what the measurement means
without asserting any specific stroke-type pattern, and deliberately do
NOT tell the player to try anything based on the number alone. If those
two entries read as vaguer than the others, that's why — flag if you
think they're *too* vague to be useful even so.

**4. Every entry below was rewritten on 2026-08-01 for tone, after real
player-facing testing showed the previous wording read as measurement
reporting, not coaching** (e.g. "your follow-through extension ratio
(0.216) is at the 0.0th percentile" — a real number, correctly cited, but
not something a club player knows what to do with). The rewrite rules
were: lead with what it means in plain language (no "ratio"/"percentile"
jargon in the main sentence), state findings directly instead of hedging
on something already measured as unusual, end with something the player
can actually try next time, and keep it short enough that several results
stacked on one screen stay skimmable. **This did not change which fields
can be cited or how a citation is checked against real data** — only the
prose. Where the rewrite introduced a concrete new "try this" that wasn't
in the prior wording, the entry is marked **⚠ NEW ADVICE** below — that
flag is exactly what we'd like your review to weigh most carefully,
since it's content nobody has reviewed in this form before.

---

## What you're actually reviewing

Each entry below is a snippet the app's AI coaching layer can retrieve and
cite when generating feedback for a real player. The final sentence a
player sees is grounded in — but not always verbatim identical to — the
text below; the AI combines a snippet's guidance with the player's own
specific measured number (e.g. "your first-serve percentage" + this
snippet's interpretation of what a drop in that number tends to mean).
**What you're reviewing is the substance and direction of the advice the
AI is allowed to give, not necessarily the exact final UI sentence.** If
the substance is wrong, everything downstream built on it is wrong too,
regardless of exact phrasing.

---

## How to fill this out

Each entry has:
1. **Is this technically correct?** (Yes / No / Partly — and why, if not
   straightforwardly yes)
2. **Would you say this to a player?** (Yes / No / With changes — and what
   you'd change)
3. **What's wrong or missing?** (free text — anything from wording to a
   missing caveat to "this is fine")

Leave any entry blank if you have nothing to add — a blank entry with no
notes is a useful signal on its own (nothing stood out).

---

## Section A — Session-level match-stat feedback

12 metrics, 2 differently-framed entries each (24 entries total). These
come from `rag/kb_placeholder.py`.

### A1. `first_serve_pct` — % of first serves that land in play

**E1.**
> Your first serve isn't landing as often as usual for you. Before changing
> anything in your motion, try playing it safer for a bit -- a little less
> pace, a little more margin -- and see if it comes back on its own.

1. Technically correct? ______________________________________________
2. Would you say this to a player? ____________________________________
3. Wrong or missing? __________________________________________________

**E2.**
> You're missing more first serves than usual, even though your toss looks
> steady on video -- that points to over-swinging for extra pace, not a
> motion breakdown. Try taking 10-15% off your target serve speed and see
> if your percentage comes back.

1. Technically correct? ______________________________________________
2. Would you say this to a player? ____________________________________
3. Wrong or missing? __________________________________________________

### A2. `first_serve_points_won_pct` — % of points won when the first serve landed in

**E3.**
> Your first serve is landing fine, but you're not winning as many points
> behind it as usual -- that points to what you do with the point right
> after the serve, not the serve itself. Look at your shot choice on that
> first ball, not your service motion.

1. Technically correct? ______________________________________________
2. Would you say this to a player? ____________________________________
3. Wrong or missing? __________________________________________________

**E4.**
> You're winning fewer points behind your first serve than usual this
> session. Before assuming something's wrong with your game, consider that
> the returner may just have read your serve well today -- this kind of
> dip is often about them, not you.

1. Technically correct? ______________________________________________
2. Would you say this to a player? ____________________________________
3. Wrong or missing? __________________________________________________

### A3. `second_serve_points_won_pct` — % of points won when the player had to hit a second serve

**E5.**
> Your second serve isn't winning points the way it usually does, and your
> double faults are creeping up too -- that combination usually points to
> overhitting or pressing on the second serve, not playing it too safe.
> Try taking a little pace off and aiming for more margin over the net,
> rather than going for more on that ball.

1. Technically correct? ______________________________________________
2. Would you say this to a player? ____________________________________
3. Wrong or missing? __________________________________________________

**E6.**
> Your second serve is losing points while your first serve is winning
> them normally -- that points to the second serve specifically, not your
> overall service motion or footwork. Any changes are better spent there
> than on reworking your whole serve.

1. Technically correct? ______________________________________________
2. Would you say this to a player? ____________________________________
3. Wrong or missing? __________________________________________________

### A4. `ace_rate` — aces as a share of service points played

**E7.**
> Your ace count is down today, but if your first-serve points won is
> holding steady, this probably isn't about your serve -- the returner
> likely just read you well. Only worry about it if the pattern keeps
> showing up over several sessions.

1. Technically correct? ______________________________________________
2. Would you say this to a player? ____________________________________
3. Wrong or missing? __________________________________________________

**E8.**
> Your aces dropped along with a slower reaction time today -- that
> combination usually means fatigue is taking a bit off your racket speed
> on serve, not a targeting problem. If you're feeling tired, this is one
> to let go rather than trying to fix your placement.

1. Technically correct? ______________________________________________
2. Would you say this to a player? ____________________________________
3. Wrong or missing? __________________________________________________

### A5. `double_fault_rate` — double faults as a share of service points played

**E9.**
> You double-faulted more than usual today, alongside a lower first-serve
> percentage -- that combination usually means you pushed too hard on the
> second serve under pressure, not a toss problem. Check whether they
> clustered on break points or late in games; if so, that's pressure, not
> technique.

1. Technically correct? ______________________________________________
2. Would you say this to a player? ____________________________________
3. Wrong or missing? __________________________________________________

**E10.** ⚠ **NEW ADVICE** *(previously stated the fatigue-clustering
pattern as an observation only; now adds a concrete suggestion)*
> Your double faults bunched up in the second half of the session instead
> of spreading out evenly -- that's usually fatigue catching up with your
> toss, not a technical flaw. Consider a shorter warm-up or more rest
> between sets before changing your serve motion.

1. Technically correct? ______________________________________________
2. Would you say this to a player? ____________________________________
3. Wrong or missing? __________________________________________________

### A6. `break_points_saved_pct` — % of the opponent's break-point chances the player defended

**E11.**
> Your break-point save rate was low this session, but a handful of break
> points isn't enough to draw real conclusions from. Don't change
> anything based on this alone -- just watch this number over your next
> few sessions before treating it as a pattern.

1. Technically correct? ______________________________________________
2. Would you say this to a player? ____________________________________
3. Wrong or missing? __________________________________________________

**E12.** ⚠ **NEW ADVICE** *(previously isolated the issue to "big-point
shot selection" without saying what to do about it; now recommends
dialing back risk specifically)*
> Your break-point save rate is low, but your overall unforced-error rate
> looks normal -- that points to how you're playing the biggest points
> specifically, not your game as a whole. On break points, try dialing
> back your risk a notch rather than going bigger.

1. Technically correct? ______________________________________________
2. Would you say this to a player? ____________________________________
3. Wrong or missing? __________________________________________________

### A7. `unforced_error_rate` — unforced errors as a share of total points played

**E13.** ⚠ **NEW ADVICE** *(previously described the fatigue-vs-technique
split as an observation only; now tells the player what to actually do
in each case)*
> You made more unforced errors than usual today. Check when they
> happened -- a cluster late in the session points to fatigue, so rest a
> bit more between reps; errors on one specific shot point to that shot's
> technique instead.

1. Technically correct? ______________________________________________
2. Would you say this to a player? ____________________________________
3. Wrong or missing? __________________________________________________

**E14.** ⚠ **NEW ADVICE** *(previously said to "compare against winners
to tell the two apart"; now recommends taking pace off directly -- this
is the entry the original real-testing feedback used as its own example
of what coaching should sound like)*
> Your errors went up while your points got shorter -- you were probably
> going for bigger shots than usual. Check how many of those bigger
> swings actually landed as winners; if not many did, try taking a little
> pace off until your errors settle back down.

1. Technically correct? ______________________________________________
2. Would you say this to a player? ____________________________________
3. Wrong or missing? __________________________________________________

### A8. `winners_rate` — clean winners as a share of total points played

**E15.**
> Your winners were down today, but your unforced errors didn't go up --
> that usually means you played more conservatively, not that your shots
> got worse. If that was a deliberate choice against a tough opponent, no
> need to change anything; if not, it's worth noticing.

1. Technically correct? ______________________________________________
2. Would you say this to a player? ____________________________________
3. Wrong or missing? __________________________________________________

**E16.**
> Your winners and unforced errors moved in the same direction today --
> both up, or both down. Read that as one shift in how aggressively you
> were playing, not two separate problems to fix.

1. Technically correct? ______________________________________________
2. Would you say this to a player? ____________________________________
3. Wrong or missing? __________________________________________________

### A9. `avg_rally_length` — average number of shots per point

**E17.**
> Your points got shorter than usual today. If your winners were also up,
> that's you taking control earlier in points -- a good sign; if your
> unforced errors were up instead, points are ending before you can build
> them, which is worth addressing.

1. Technically correct? ______________________________________________
2. Would you say this to a player? ____________________________________
3. Wrong or missing? __________________________________________________

**E18.**
> Your points ran longer than usual while your winners were down -- that's
> a more passive, defensive session. That's fine against a tough opponent
> on the day, but worth noticing if it keeps happening rather than being
> a one-off.

1. Technically correct? ______________________________________________
2. Would you say this to a player? ____________________________________
3. Wrong or missing? __________________________________________________

### A10. `court_coverage_speed_mps` — average on-court movement speed while playing, in m/s (from video tracking)

**E19.**
> You covered the court more slowly than usual today, and your reaction
> time was down too -- together that's a fatigue signal, not a footwork
> problem. Consider more recovery time or a lighter session before your
> next one.

1. Technically correct? ______________________________________________
2. Would you say this to a player? ____________________________________
3. Wrong or missing? __________________________________________________

**E20.**
> You moved slower than usual today, but your total distance covered was
> normal -- that points to less urgency on individual points, not less
> overall effort. Worth checking your split-step timing on video to see
> if you're getting a late start on points.

1. Technically correct? ______________________________________________
2. Would you say this to a player? ____________________________________
3. Wrong or missing? __________________________________________________

### A11. `avg_reaction_time_ms` — average time from the opponent's contact to the player's first responsive movement, in ms

**E21.**
> Your reaction time was slower than usual today, especially later in the
> session -- that's a classic fatigue marker, not a technical issue.
> Check how long you'd been playing and what time of day it was before
> reading too much into it.

1. Technically correct? ______________________________________________
2. Would you say this to a player? ____________________________________
3. Wrong or missing? __________________________________________________

**E22.** ⚠ **NEW ADVICE** *(previously said to "consider anticipation
instead of physical conditioning"; now tells the player specifically to
watch the ball earlier off the opponent's racket)*
> Slower reaction time along with slower court coverage points to overall
> fatigue. If it's reaction time alone though, with normal movement
> speed, try watching the ball earlier out of your opponent's racket
> rather than working on fitness.

1. Technically correct? ______________________________________________
2. Would you say this to a player? ____________________________________
3. Wrong or missing? __________________________________________________

### A12. `total_distance_covered_m` — total distance the player moved on court over the session, in meters

**E23.**
> You covered less ground than usual today. Before treating that as a
> fitness or effort issue, check whether the session was shorter or
> points were ending quicker than usual -- both would explain it without
> anything being wrong.

1. Technically correct? ______________________________________________
2. Would you say this to a player? ____________________________________
3. Wrong or missing? __________________________________________________

**E24.**
> Distance covered, court-speed, and reaction time all dropping together
> is a fairly complete fatigue picture. If only one of the three is down,
> treat it as something narrower and specific, not overall fatigue.

1. Technically correct? ______________________________________________
2. Would you say this to a player? ____________________________________
3. Wrong or missing? __________________________________________________

---

## Section B — Stroke-level biomechanics feedback

7 metrics, 2 entries each (14 entries total). These come from
`rag/kb_biomechanics.py`. **Reminder: every number here is computed as a
percentile within our own small sampled dataset, not an absolute or
externally validated "correct" value** — the player-facing text below
deliberately avoids the word "percentile," but the underlying number is
one; please keep that in mind throughout, and say directly if you think
that framing itself is a problem, independent of any individual entry's
wording.

### B1. `wrist_height_ratio` — contact height of the hitting wrist, relative to shoulder width, at the moment of contact

**E25.**
> Where you make contact with the ball naturally changes by shot, and
> yours looks unusual for this one -- that's not automatically wrong,
> some players do it on purpose. If it feels rushed or uncomfortable
> though, try meeting the ball a touch earlier or later. (contact height:
> unusual for this shot)

1. Technically correct? ______________________________________________
2. Would you say this to a player? ____________________________________
3. Wrong or missing? __________________________________________________

**E26.**
> A lower contact point than usual for this shot could mean you're taking
> the ball later than normal, or it could just be your natural, flatter
> swing style -- this measurement alone can't tell which. If it doesn't
> feel like a problem to you, there's nothing here that says it is.

1. Technically correct? ______________________________________________
2. Would you say this to a player? ____________________________________
3. Wrong or missing? __________________________________________________

### B2. `weight_transfer_ratio` — how far the hips shift horizontally (in shoulder-widths) from the start of the swing to the contact frame

**E27.** ⚠ **NEW ADVICE** *(previously this entry was a pure data-quality
caveat -- that the measurement's sign depends on camera framing and
hasn't been independently verified -- with zero player advice. Now
recommends driving off the back foot, with the sign/calibration caveat
restored alongside it in plain language -- see E34 for the same pattern
of advice + a real measurement caveat, both present.)*
> How much your weight shifts forward into the shot -- from your back
> foot toward the ball -- shows up here. If yours looks light, try
> driving more off your back foot as you swing, so your weight is moving
> into the shot at contact rather than staying back. One caveat: this
> reading's direction can vary with how your camera was angled, so treat
> it as a starting point, not a certainty.

1. Technically correct? ______________________________________________
2. Would you say this to a player? ____________________________________
3. Wrong or missing? __________________________________________________

**E28.**
> An unusually low weight shift into the shot isn't automatically a flaw
> -- it can just mean a more neutral, reactive stance, which is normal on
> defensive shots. Treat this as a talking point rather than something to
> fix outright.

1. Technically correct? ______________________________________________
2. Would you say this to a player? ____________________________________
3. Wrong or missing? __________________________________________________

### B3. `knee_flex` (left + right) — how bent or extended the knees are at the contact instant (hip-knee-ankle angle)

**E29.**
> How bent your knees are right at the moment of contact -- most
> players' legs are close to straight at that exact instant across all
> shot types, so this number on its own doesn't tell us much that's
> specific to you. There's nothing to change based on this measurement
> alone.

1. Technically correct? ______________________________________________
2. Would you say this to a player? ____________________________________
3. Wrong or missing? __________________________________________________

**E30.** ⚠ **NEW ADVICE** *(previously framed as a percentile-relative
observation only, explicitly not a claimed-ideal value; now recommends
starting the leg drive earlier)*
> If your knees are noticeably more bent than usual at the moment of
> contact, that can be a sign of a later or more compact leg drive. Try
> starting your leg drive a touch earlier, so your legs are straightening
> as you make contact rather than still loading.

1. Technically correct? ______________________________________________
2. Would you say this to a player? ____________________________________
3. Wrong or missing? __________________________________________________

### B4. `stance_width_ratio` — ankle-to-ankle distance relative to shoulder width, at the contact frame

**E31.** ⚠ **NEW ADVICE** *(previously described what a low/high
percentile might indicate without recommending a fix; now tells the
player to set their feet wider)*
> A narrower stance than usual for you at contact can mean a less stable
> base -- try setting your feet a little wider before you strike the
> ball. A wider stance than usual isn't automatically a problem though;
> that can just be a deliberate wide split-step or lunge.

1. Technically correct? ______________________________________________
2. Would you say this to a player? ____________________________________
3. Wrong or missing? __________________________________________________

**E32.**
> This is measured at just one instant of your swing, not your footwork
> through the whole stroke. Don't read too much into it on its own --
> look at it alongside your other results, not by itself.

1. Technically correct? ______________________________________________
2. Would you say this to a player? ____________________________________
3. Wrong or missing? __________________________________________________

### B5. `follow_through_extension_ratio` — how far the hitting wrist is from the shoulder at the end of the clip vs. at contact

**E33.** ⚠ **NEW ADVICE** *(previously this entry only defined what the
ratio measures mechanically, with no advice content at all; now
recommends letting the racket travel out toward the target)*
> If your swing stops moving soon after contact, that's a short
> follow-through -- normal on compact shots like volleys, so not
> automatically a problem there. On groundstrokes though, try letting
> your racket keep travelling out toward the target after you hit the
> ball, instead of pulling up short.

1. Technically correct? ______________________________________________
2. Would you say this to a player? ____________________________________
3. Wrong or missing? __________________________________________________

**E34.** ⚠ **NEW ADVICE** *(previously flagged a short follow-through as
"worth noting" with a clip-boundary caveat, no fix; now recommends
letting the arm continue through the shot -- the clip-boundary caveat is
kept, since it's a real measurement limitation, not just tone)*
> A much shorter follow-through than usual for you can mean you're
> checking your swing early -- try letting your arm continue through the
> shot instead of stopping right at contact. One caveat: this is only
> measured to the end of the filmed clip, so a swing still in motion
> when the video cuts off can look shorter than it really is.

1. Technically correct? ______________________________________________
2. Would you say this to a player? ____________________________________
3. Wrong or missing? __________________________________________________

### B6. `elbow_angle_deg` — bend of the hitting arm's elbow at the moment of contact

**Note:** this metric's entries are deliberately generic — see "Four
things to know" item 3 above. A specific forehand/backhand pattern seen on
5 examples did not hold up on the full 1,980-clip population, so no
specific stroke-type pattern is claimed here, and no "try X" fix is given
-- these two entries are NOT flagged as new advice, because they
deliberately still contain none. Please flag if these read as too vague
to be useful even with that context.

**E35.**
> How bent your arm is at the moment you hit the ball varies a lot
> between players even on the same shot -- there's no single 'correct'
> angle to aim for. If this looks unusual for you, it's worth a coach's
> eye on video rather than something to fix from a number alone.

1. Technically correct? ______________________________________________
2. Would you say this to a player? ____________________________________
3. Wrong or missing? __________________________________________________

**E36.**
> This number alone can't tell you if your arm bend is a problem -- it
> varies too much between players to have one right answer. Treat an
> unusual reading as a cue to get a coach's eyes on your video, not proof
> something's wrong.

1. Technically correct? ______________________________________________
2. Would you say this to a player? ____________________________________
3. Wrong or missing? __________________________________________________

### B7. `shoulder_hip_separation_deg` — angle between shoulder line and hip line at contact (upper-body rotation relative to lower body, sometimes called "X-factor")

**Note:** same situation as `elbow_angle_deg` above — an early
"serve has low separation" impression was contradicted by the full
population (serve is actually one of the *highest*-separation classes),
so no specific stroke-type claim is made here either, and these two
entries are likewise NOT flagged as new advice.

**E37.**
> How much your upper body has turned relative to your hips at contact
> is what this measures -- sometimes called your 'coil.' An unusual
> reading here is worth a conversation about timing with a coach, not
> something to grade as pass or fail.

1. Technically correct? ______________________________________________
2. Would you say this to a player? ____________________________________
3. Wrong or missing? __________________________________________________

**E38.**
> This is measured from a single estimated instant in your swing, not
> tracked continuously -- so treat the number as a rough signal, not an
> exact reading. It's not something to chase precision on by itself.

1. Technically correct? ______________________________________________
2. Would you say this to a player? ____________________________________
3. Wrong or missing? __________________________________________________

---

## Whole-set questions

Please answer these after reviewing all 38 entries above.

**1. What common tennis faults or observations do you think are missing
entirely from this knowledge base?** (Either metrics we're not measuring
at all, or common coaching observations within a metric we do measure that
none of the entries above capture.)

______________________________________________________________________

______________________________________________________________________

______________________________________________________________________

**2. Is there anything in this document that you'd consider actively bad
advice** — something you would not want a student to read, even out of
context, regardless of how it's caveated? **Please look especially closely
at the 10 entries marked ⚠ NEW ADVICE above** — those are the only places
in this document where we're now telling a player to specifically try
something, rather than just interpreting a number for them, and none of
that phrasing has been reviewed by a coach before.

______________________________________________________________________

______________________________________________________________________

______________________________________________________________________

**3. Anything else?** General impressions, tone, things that surprised
you, or anything the two questions above didn't give you room to say.

______________________________________________________________________

______________________________________________________________________

______________________________________________________________________

---

*Thank you for reviewing this. If you're able to go through it with a
second coach independently before comparing notes, that's genuinely more
useful to us than a single combined pass — coaching philosophy varies, and
disagreement between two reviewers is itself a useful signal about which
entries are more subjective vs. more clearly right or wrong.*
