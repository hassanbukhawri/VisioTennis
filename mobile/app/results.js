// Results -- reads the just-computed result from SessionContext (set by
// Processing right before router.replace('/results')), renders it via the
// shared ResultView component also used by History's detail screen.
import { useMemo } from "react";
import { useRouter } from "expo-router";
import { KeyboardAvoidingView, Platform, Pressable, StyleSheet, Text, View } from "react-native";

import FeedbackForm from "../components/FeedbackForm";
import ResultView from "../components/ResultView";
import StrokeChat from "../components/StrokeChat";
import { useSession } from "../lib/SessionContext";
import { fontSizes, radii, spacing, useTheme } from "../lib/theme";

export default function ResultsScreen() {
  const { currentResult } = useSession();
  const router = useRouter();
  const { colors } = useTheme();
  const styles = useMemo(() => makeStyles(colors), [colors]);

  if (!currentResult) {
    // Only reachable if this route is opened directly without going through
    // Processing first (e.g. a stale deep link) -- not a normal app state.
    return (
      <View style={styles.centered}>
        <Text style={styles.noResultText}>No result to show.</Text>
        <Pressable
          onPress={() => router.replace("/(tabs)")}
          style={({ pressed }) => [styles.btnPrimary, pressed && styles.btnPressed]}
        >
          <Text style={styles.btnPrimaryText}>Back to Home</Text>
        </Pressable>
      </View>
    );
  }

  return (
    <KeyboardAvoidingView
      style={styles.container}
      behavior={Platform.OS === "ios" ? "padding" : "height"}
    >
      <ResultView result={currentResult}>
        <FeedbackForm result={currentResult} />
        {currentResult.coaching.success ? (
          <StrokeChat sessionId={currentResult.stroke_event.event_id} />
        ) : null}
      </ResultView>
    </KeyboardAvoidingView>
  );
}

const makeStyles = (colors) => StyleSheet.create({
  container: { flex: 1, backgroundColor: colors.bg, paddingHorizontal: spacing.md, paddingTop: spacing.sm },
  centered: { flex: 1, backgroundColor: colors.bg, alignItems: "center", justifyContent: "center", gap: spacing.md },
  noResultText: { fontSize: fontSizes.body, color: colors.textSoft },
  btnPrimary: { backgroundColor: colors.navy, borderRadius: radii.pill, paddingVertical: 13, paddingHorizontal: spacing.lg },
  btnPrimaryText: { color: colors.onPrimary, fontSize: fontSizes.small, fontWeight: "600" },
  btnPressed: { opacity: 0.75 },
});
