// Stroke chat panel (2026-08-08) -- attached to the Results screen and the
// History detail screen (both render one stroke's ResultView), scoped to
// asking follow-up questions about THAT stroke only, per the approved chat
// grounding design (PROJECT_LOG.md). Conversation memory is deliberately
// ephemeral -- plain React state here, lost when this component unmounts,
// never persisted -- see the design's memory decision.
//
// Renders 3 distinct coach-bubble styles matching the backend's 3 chat
// routes (never guessed client-side -- the route comes straight from
// rag/chat_pipeline.py's response):
//   - existing_citation / new_grounded_claim -> grounded bubble (teal
//     accent + citation caption, same visual language as ResultView's
//     citationText)
//   - general_note -> plain bubble, no citation
//   - unavailable_comparison -> amber caveat bubble (same colors.caveat*
//     tokens ResultView already uses for confidence/pose caveats) -- the
//     disclaimer text itself is prepended server-side (chat_pipeline.py),
//     never generated client-side, so this bubble is just styling.
//
// Loading state deliberately does NOT try to predict the route/tier before
// the response arrives (that would need a second round-trip) -- instead
// mirrors app/processing.js's own pattern: show a quick "Thinking..." state,
// then swap to a "this can take a bit longer" message once enough time has
// passed that a cheap (tier 1/3) reply would already have returned.
import { useEffect, useMemo, useRef, useState } from "react";
import { Keyboard, Pressable, StyleSheet, Text, TextInput, View } from "react-native";
import Svg, { Ellipse, Path } from "react-native-svg";

import { useAuth } from "../lib/AuthContext";
import { sendChatMessage } from "../lib/chatRepository";
import { fonts, fontSizes, radii, spacing, useTheme } from "../lib/theme";

// Coach avatar naming (2026-08-23) -- the fallback shown wherever a user
// hasn't set coach_name (backend/db.py), and the literal string every
// display site must use verbatim rather than re-typing "Coach" in multiple
// places and risking drift. Exported so settings.js's placeholder/hint text
// matches this exact value, not a second hardcoded copy of it.
export const DEFAULT_COACH_NAME = "Coach";

// Suggested-prompt chip (2026-08-12, item 7's cross-session digest) --
// sends this EXACT phrase through the normal chat path when tapped, same as
// if the player had typed and sent it themselves. Shown unconditionally,
// not gated on whether a real digest exists for this player: the pipeline
// already handles "no digest yet" gracefully with a fixed fallback message
// (rag/chat_pipeline.py:NO_DIGEST_AVAILABLE_MESSAGE) via the same
// classify_turn() -> route="progress_digest" path a typed version of this
// question would take, so there's no separate "does this player have
// enough history" check to build just to gate a button -- the existing
// fallback already covers it.
const PROGRESS_PROMPT_TEXT = "How am I trending?";

const THINKING_MESSAGE = "Thinking...";
// "Checking your data" (not "double-checking", the analyze-stroke retry-gate's own wording) --
// a chat follow-up's tier-2 path is a fresh grounded claim, not a re-verification of feedback
// already shown, so the wording is deliberately different even though the mechanism is the same.
const RETRY_GATE_MESSAGE = "Checking your data -- this can take a bit longer...";
// Calibrated against real measured tier latencies (rag/run_chat_pipeline_trials.py,
// PROJECT_LOG.md): tier 1 is instant, tier 3 is ~8-17s even with its own 1-retry parse
// budget, tier 2's retry-gate is the slow path (~20s-200s+). 12s sits comfortably above
// typical tier 1/3 completion and below where a tier-2 turn would still be running.
const RETRY_GATE_THRESHOLD_MS = 12000;

function citationCaption(citation) {
  if (!citation) return null;
  const path = citation.ir_field_path;
  // SessionDigest citations (item 7, 2026-08-12) -- both prefixes are
  // already percentile-scale numbers (see ir/schema.py:SessionDigest),
  // captioned distinctly from a single-stroke biomechanics citation rather
  // than falling through to the generic "field: value" format below, which
  // would show the raw "trend_deltas.foo" path verbatim and lose the sign/
  // "percentile points" framing that makes a trend number legible.
  if (path.startsWith("trend_deltas.")) {
    const field = path.replace("trend_deltas.", "");
    const sign = citation.cited_value >= 0 ? "+" : "";
    return `${field} trend: ${sign}${citation.cited_value} percentile pts`;
  }
  if (path.startsWith("trailing_averages.")) {
    const field = path.replace("trailing_averages.", "");
    return `${field}: ${citation.cited_value}th percentile avg`;
  }
  const field = path.replace("biomechanics_percentile.", "").replace("biomechanics.", "");
  const suffix = path.startsWith("biomechanics_percentile.") ? "th percentile" : "";
  return `${field}: ${citation.cited_value}${suffix}`;
}

// Coach avatar (2026-08-23) -- 3 expression states matching the approved
// design mockup exactly (same SVG shapes, translated to react-native-svg).
// Duotone: navy linework on the theme's own surface tokens, no new colors
// introduced -- `skin`/`skinShade` below reuse colors.surfaceAlt/colors.border
// (already close to the mockup's dedicated skin tones in both themes, no new
// tokens needed) so the character survives the light/dark swap for free,
// same as everything else in this app.
function CoachAvatar({ state, colors, size = 26 }) {
  const navy = colors.navy;
  const skin = colors.surfaceAlt;
  const skinShade = colors.border;
  const teal = colors.teal;

  const eyesAndBrows =
    state === "listening" ? (
      <>
        <Ellipse cx="47" cy="59" rx="3.6" ry="5.6" fill={navy} />
        <Ellipse cx="73" cy="59" rx="3.6" ry="5.6" fill={navy} />
        <Path d="M40 47 Q47 43 55 46" stroke={navy} strokeWidth="2.4" fill="none" strokeLinecap="round" />
        <Path d="M65 46 Q73 43 80 47" stroke={navy} strokeWidth="2.4" fill="none" strokeLinecap="round" />
      </>
    ) : (
      <>
        <Ellipse cx="47" cy="58" rx="4.2" ry="5.2" fill={navy} />
        <Ellipse cx="73" cy="58" rx="4.2" ry="5.2" fill={navy} />
        <Path d="M40 49 Q47 45.5 54 49" stroke={navy} strokeWidth="2.4" fill="none" strokeLinecap="round" />
        <Path d="M66 49 Q73 45.5 80 49" stroke={navy} strokeWidth="2.4" fill="none" strokeLinecap="round" />
      </>
    );

  // Unified calm/attentive mouth-curve family (2026-08-25 redesign, approved
  // mockup) -- idle and listening now share the same gentle closed-mouth
  // curve (listening is just a shallower version of it), instead of idle's
  // curve and listening's unrelated flat line. Keeps the baseline demeanor
  // legible underneath the functional state change. Speaking's open-mouth
  // shape stays anchored at the same x-center (60) for visual continuity.
  const mouth =
    state === "speaking" ? (
      <>
        <Ellipse cx="60" cy="79" rx="9" ry="6.5" fill={navy} />
        <Ellipse cx="60" cy="78" rx="5.4" ry="3.4" fill={skin} />
      </>
    ) : state === "listening" ? (
      <Path d="M52 77.3 Q60 78.6 68 77.3" stroke={navy} strokeWidth="3" fill="none" strokeLinecap="round" />
    ) : (
      <Path d="M50 76.5 Q60 80 70 76.5" stroke={navy} strokeWidth="3" fill="none" strokeLinecap="round" />
    );

  return (
    <Svg viewBox="0 0 120 120" width={size} height={size}>
      {/* Tapered jaw (2026-08-25 redesign) -- chin point pulled to y95 with
          tighter control points vs. the original's uniformly round y92 arc. */}
      <Path
        d="M60 20 C40 20 30 36 30 54 C30 73 41 91 60 95 C79 91 90 73 90 54 C90 36 80 20 60 20 Z"
        fill={skin} stroke={navy} strokeWidth="4"
      />
      {/* Asymmetric side-swept hair (2026-08-25 redesign) -- off-center part,
          longer sweep down the left side, uneven sideburns -- replaces the
          old symmetric "dome" so the silhouette reads as one particular
          head, not a generic short-crop shape, even at avatar size. */}
      <Path
        d="M28 48 C27 30 38 14 58 13 C74 13 87 23 90 38 C91 43 90 47 87 44 C80 32 66 25 54 26 C44 27 36 32 32 41 C29 44 28 46 28 48 Z"
        fill={navy}
      />
      <Path d="M52 16 Q48 22 44 29" stroke={skin} strokeWidth="1.6" fill="none" strokeLinecap="round" opacity="0.55" />
      <Path d="M30 46 L32 60 L37 58 L35 45 Z" fill={navy} />
      <Path d="M86 40 L88 50 L92 48 L90 39 Z" fill={navy} />
      {/* Collar shown in all 3 states, including idle (2026-08-25 redesign,
          approved) -- reads as part of the character's settled, attentive
          posture rather than only a functional-state marker. */}
      <Path
        d="M46 83 L41 98 L53 102 L60 93 L67 102 L79 98 L74 83 L60 90 Z"
        fill={skinShade} stroke={navy} strokeWidth="2"
      />
      {eyesAndBrows}
      {state !== "listening" ? <Path d="M58 62 Q60 68 62 62" stroke={navy} strokeWidth="2" fill="none" strokeLinecap="round" /> : null}
      {mouth}
      {state === "listening" ? (
        <>
          <Path d="M94 30 m-3,0 a3,3 0 1,0 6,0 a3,3 0 1,0 -6,0" fill={teal} opacity="0.55" />
          <Path d="M101 38 m-2.2,0 a2.2,2.2 0 1,0 4.4,0 a2.2,2.2 0 1,0 -4.4,0" fill={teal} opacity="0.4" />
        </>
      ) : null}
    </Svg>
  );
}

// Bubble content only, byte-identical to before the avatar was added --
// deliberately unchanged, per instruction not to touch bubble styling.
// CoachBubble below wraps this with the avatar + name label.
function CoachBubbleContent({ message, styles }) {
  if (message.error) {
    return (
      <View style={[styles.bubble, styles.coachBubble, styles.errorBubble]}>
        <Text style={styles.errorBubbleText}>{message.text}</Text>
      </View>
    );
  }
  if (message.route === "unavailable_comparison") {
    return (
      <View style={[styles.bubble, styles.coachBubble, styles.caveatBubble]}>
        <Text style={styles.caveatBubbleText}>{message.text}</Text>
      </View>
    );
  }
  // "progress_digest" (item 7, 2026-08-12) included here too -- a
  // successful digest response is just as genuinely grounded (real
  // SessionDigest citation) as "new_grounded_claim", it just cites a
  // different record type. A digest turn that got downgraded to
  // "unavailable_comparison" by the skill-tier backstop is handled by the
  // check above already, before reaching this line. Requires an actual
  // citation for "progress_digest" specifically -- the digest=None fallback
  // (NO_DIGEST_AVAILABLE_MESSAGE) is also route="progress_digest" but has
  // no citation at all, and shouldn't get the same teal "grounded"
  // treatment as a real, cited trend claim -- it falls through to the
  // plain default bubble instead, same as a general_note.
  const grounded =
    message.route === "existing_citation" || message.route === "new_grounded_claim" ||
    (message.route === "progress_digest" && message.citation != null);
  return (
    <View style={[styles.bubble, styles.coachBubble, grounded && styles.groundedBubble]}>
      <Text style={styles.coachBubbleText}>{message.text}</Text>
      {grounded && message.citation ? (
        <Text style={styles.citationText}>{citationCaption(message.citation)}</Text>
      ) : null}
    </View>
  );
}

// Adds the avatar + name label around CoachBubbleContent -- `showLabel` is
// false for every coach message except the first in a consecutive run (set
// by the caller, which has the full messages array and knows what preceded
// this one), same convention a messaging app uses for a sender name: shown
// once per run, not repeated on every bubble. Every delivered coach message
// uses the "speaking" expression, permanently (not reverted to idle after a
// delay) -- a historical message legitimately stays in the state it was
// delivered in; only the loading indicator and the panel-title avatar are
// live/transient (see StrokeChat below).
function CoachBubble({ message, styles, colors, coachName, showLabel }) {
  return (
    <View style={styles.coachBubbleRow}>
      <View style={styles.miniAvatar}>
        <CoachAvatar state="speaking" colors={colors} size={26} />
      </View>
      <View style={styles.coachCol}>
        {showLabel ? <Text style={styles.coachNameLabel}>{coachName}</Text> : null}
        <CoachBubbleContent message={message} styles={styles} />
      </View>
    </View>
  );
}

export default function StrokeChat({ sessionId }) {
  const { token, user } = useAuth();
  const { colors } = useTheme();
  const styles = useMemo(() => makeStyles(colors), [colors]);
  const coachName = user?.coach_name?.trim() || DEFAULT_COACH_NAME;
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [sending, setSending] = useState(false);
  const [loadingText, setLoadingText] = useState(THINKING_MESSAGE);
  const [keyboardHeight, setKeyboardHeight] = useState(0);
  const thresholdTimerRef = useRef(null);

  // This panel sits at the bottom of ResultView's ScrollView, not pinned to the screen
  // like a typical chat input -- the screen-level KeyboardAvoidingView (results.js,
  // history/[sessionId].js) shrinks the scrollable area when the keyboard opens, but
  // since the input row is the LAST thing in that ScrollView, there's otherwise no
  // scrollable slack below it to bring it fully above the keyboard. This spacer, sized
  // to the real keyboard height, supplies exactly that slack only while the keyboard
  // is actually showing.
  useEffect(() => {
    const showSub = Keyboard.addListener("keyboardDidShow", (e) => setKeyboardHeight(e.endCoordinates.height));
    const hideSub = Keyboard.addListener("keyboardDidHide", () => setKeyboardHeight(0));
    return () => {
      showSub.remove();
      hideSub.remove();
    };
  }, []);

  async function handleSend(textOverride) {
    const text = (textOverride ?? input).trim();
    if (!text || sending) return;
    if (!textOverride) setInput("");
    const conversationHistory = messages.map((m) => ({ role: m.role, text: m.text }));
    setMessages((prev) => [...prev, { id: `p-${Date.now()}`, role: "player", text }]);
    setSending(true);
    setLoadingText(THINKING_MESSAGE);
    thresholdTimerRef.current = setTimeout(() => setLoadingText(RETRY_GATE_MESSAGE), RETRY_GATE_THRESHOLD_MS);

    try {
      const result = await sendChatMessage(token, sessionId, text, conversationHistory);
      setMessages((prev) => [
        ...prev,
        {
          id: `c-${Date.now()}`,
          role: "coach",
          text: result.text,
          route: result.route,
          citation: result.citation,
          error: !result.success,
        },
      ]);
    } catch (e) {
      setMessages((prev) => [
        ...prev,
        {
          id: `c-${Date.now()}`,
          role: "coach",
          text: "Something went wrong sending that -- check your connection and try again.",
          error: true,
        },
      ]);
    } finally {
      clearTimeout(thresholdTimerRef.current);
      setSending(false);
    }
  }

  // First-in-run label logic (shared by the message loop and the loading
  // row below) -- a name label shows once per consecutive coach turn, not
  // repeated on every bubble, same convention a messaging app uses.
  const loadingShowsLabel = messages.length === 0 || messages[messages.length - 1].role !== "coach";

  return (
    <View style={styles.container}>
      <View style={styles.titleRow}>
        <View style={styles.titleAvatar}>
          <CoachAvatar state={sending ? "listening" : "idle"} colors={colors} size={22} />
        </View>
        <Text style={styles.sectionTitle}>Ask about this stroke</Text>
      </View>
      <Pressable
        onPress={() => handleSend(PROGRESS_PROMPT_TEXT)}
        disabled={sending}
        style={({ pressed }) => [styles.suggestionChip, (sending || pressed) && styles.btnPressed]}
      >
        <Text style={styles.suggestionChipText}>{PROGRESS_PROMPT_TEXT}</Text>
      </Pressable>
      {messages.length === 0 && !sending ? (
        <Text style={styles.emptyText}>
          Ask a follow-up question about your results -- e.g. "why was my knee flagged?"
        </Text>
      ) : (
        <View style={styles.messages}>
          {messages.map((m, i) => (
            <View key={m.id} style={m.role === "player" ? styles.playerRow : styles.coachRow}>
              {m.role === "player" ? (
                <View style={[styles.bubble, styles.playerBubble]}>
                  <Text style={styles.playerBubbleText}>{m.text}</Text>
                </View>
              ) : (
                <CoachBubble
                  message={m}
                  styles={styles}
                  colors={colors}
                  coachName={coachName}
                  showLabel={i === 0 || messages[i - 1].role !== "coach"}
                />
              )}
            </View>
          ))}
          {sending ? (
            <View style={styles.coachRow}>
              <View style={styles.coachBubbleRow}>
                <View style={styles.miniAvatar}>
                  <CoachAvatar state="listening" colors={colors} size={26} />
                </View>
                <View style={styles.coachCol}>
                  {loadingShowsLabel ? <Text style={styles.coachNameLabel}>{coachName}</Text> : null}
                  {/* De-genericizing pass (section 04, approved): dropped the
                      generic OS spinner here -- the avatar two pixels to the
                      left is already rendering its own "listening" expression
                      plus a teal status-dot pulse, so a second stock
                      loading icon in the same row was redundant. */}
                  <View style={[styles.bubble, styles.coachBubble, styles.loadingBubble]}>
                    <Text style={styles.loadingText}>{loadingText}</Text>
                  </View>
                </View>
              </View>
            </View>
          ) : null}
        </View>
      )}
      <View style={styles.inputRow}>
        <TextInput
          style={styles.input}
          value={input}
          onChangeText={setInput}
          placeholder="Ask a question..."
          placeholderTextColor={colors.textFaint}
          editable={!sending}
          onSubmitEditing={() => handleSend()}
          returnKeyType="send"
          multiline
        />
        <Pressable
          onPress={() => handleSend()}
          disabled={sending || !input.trim()}
          style={({ pressed }) => [
            styles.sendButton,
            (sending || !input.trim()) && styles.sendButtonDisabled,
            pressed && styles.btnPressed,
          ]}
        >
          <Text style={styles.sendButtonText}>Send</Text>
        </Pressable>
      </View>
      {keyboardHeight > 0 ? <View style={{ height: keyboardHeight }} /> : null}
    </View>
  );
}

const makeStyles = (colors) => StyleSheet.create({
  container: { marginTop: spacing.lg },
  titleRow: { flexDirection: "row", alignItems: "center", gap: spacing.sm, marginBottom: spacing.sm },
  titleAvatar: {
    width: 34, height: 34, borderRadius: 17, backgroundColor: colors.surfaceAlt,
    borderWidth: 1, borderColor: colors.border, alignItems: "center", justifyContent: "center",
  },
  sectionTitle: {
    fontFamily: fonts.serif,
    fontSize: fontSizes.h1,
    color: colors.navyInk,
  },
  emptyText: { fontSize: fontSizes.small, color: colors.textSoft, lineHeight: 19, marginBottom: spacing.sm },
  suggestionChip: {
    alignSelf: "flex-start",
    borderWidth: 1.5,
    borderColor: colors.navy,
    borderRadius: radii.pill,
    paddingVertical: 8,
    paddingHorizontal: spacing.md,
    marginBottom: spacing.sm,
  },
  suggestionChipText: { color: colors.navy, fontSize: fontSizes.small, fontWeight: "600" },
  messages: { marginBottom: spacing.sm },
  playerRow: { alignItems: "flex-end", marginBottom: spacing.sm },
  coachRow: { alignItems: "flex-start", marginBottom: spacing.sm },
  coachBubbleRow: { flexDirection: "row", alignItems: "flex-end", gap: spacing.sm, maxWidth: "100%" },
  miniAvatar: {
    width: 30, height: 30, borderRadius: 15, backgroundColor: colors.surfaceAlt,
    borderWidth: 1, borderColor: colors.border, alignItems: "center", justifyContent: "center", flexShrink: 0,
  },
  coachCol: { flexShrink: 1, gap: 4 },
  coachNameLabel: { fontSize: fontSizes.caption, fontWeight: "600", color: colors.textFaint, paddingLeft: 2 },
  bubble: { maxWidth: "85%", borderRadius: radii.md, paddingVertical: spacing.sm, paddingHorizontal: spacing.md },
  playerBubble: { backgroundColor: colors.navy },
  playerBubbleText: { color: colors.onPrimary, fontSize: fontSizes.small, lineHeight: 19 },
  coachBubble: { backgroundColor: colors.surface, borderWidth: 1, borderColor: colors.border },
  coachBubbleText: { color: colors.text, fontSize: fontSizes.small, lineHeight: 19 },
  groundedBubble: { borderColor: colors.tealTint, backgroundColor: colors.tealTint },
  citationText: { fontSize: fontSizes.caption, color: colors.tealDeep, marginTop: spacing.xs },
  caveatBubble: { backgroundColor: colors.caveatTint, borderColor: colors.caveatTint },
  caveatBubbleText: { fontSize: fontSizes.small, color: colors.caveat, lineHeight: 19 },
  errorBubble: { backgroundColor: colors.errorTint, borderColor: colors.errorTint },
  errorBubbleText: { fontSize: fontSizes.small, color: colors.text, lineHeight: 19 },
  loadingBubble: { flexDirection: "row", alignItems: "center", gap: spacing.sm },
  loadingText: { fontSize: fontSizes.small, color: colors.textSoft },
  inputRow: { flexDirection: "row", alignItems: "flex-end", gap: spacing.sm },
  input: {
    flex: 1,
    backgroundColor: colors.surface,
    borderWidth: 1,
    borderColor: colors.border,
    borderRadius: radii.md,
    paddingVertical: spacing.sm,
    paddingHorizontal: spacing.md,
    fontSize: fontSizes.small,
    color: colors.text,
    maxHeight: 100,
  },
  sendButton: {
    backgroundColor: colors.navy,
    borderRadius: radii.pill,
    paddingVertical: 11,
    paddingHorizontal: spacing.md,
  },
  sendButtonDisabled: { opacity: 0.4 },
  sendButtonText: { color: colors.onPrimary, fontSize: fontSizes.small, fontWeight: "600" },
  btnPressed: { opacity: 0.75 },
});
