// History detail -- one past, on-device-stored session, rendered with the
// same ResultView component the live Results screen uses.
import { useEffect, useMemo, useState } from "react";
import { useLocalSearchParams } from "expo-router";
import { KeyboardAvoidingView, Platform, StyleSheet, Text, View } from "react-native";

import ResultView from "../../../components/ResultView";
import StrokeChat from "../../../components/StrokeChat";
import { useAuth } from "../../../lib/AuthContext";
import { getSession } from "../../../lib/historyRepository";
import { fontSizes, spacing, useTheme } from "../../../lib/theme";

export default function HistoryDetailScreen() {
  const { sessionId } = useLocalSearchParams();
  const { token } = useAuth();
  const { colors } = useTheme();
  const styles = useMemo(() => makeStyles(colors), [colors]);
  const [session, setSession] = useState(undefined); // undefined = loading, null = not found

  useEffect(() => {
    getSession(sessionId, token).then(setSession);
  }, [sessionId, token]);

  if (session === undefined) {
    return (
      <View style={styles.centered}>
        <Text style={styles.centeredText}>Loading...</Text>
      </View>
    );
  }

  if (session === null) {
    return (
      <View style={styles.centered}>
        <Text style={styles.centeredText}>Session not found.</Text>
      </View>
    );
  }

  return (
    <KeyboardAvoidingView
      style={styles.container}
      behavior={Platform.OS === "ios" ? "padding" : "height"}
    >
      <Text style={styles.timestamp}>{new Date(session.createdAt).toLocaleString()}</Text>
      <ResultView result={session.result}>
        {session.result.coaching.success ? <StrokeChat sessionId={session.id} /> : null}
      </ResultView>
    </KeyboardAvoidingView>
  );
}

const makeStyles = (colors) => StyleSheet.create({
  container: { flex: 1, backgroundColor: colors.bg, paddingHorizontal: spacing.md, paddingTop: spacing.sm },
  centered: { flex: 1, backgroundColor: colors.bg, alignItems: "center", justifyContent: "center" },
  centeredText: { fontSize: fontSizes.small, color: colors.textSoft },
  timestamp: { fontSize: fontSizes.caption, color: colors.textFaint, marginBottom: spacing.sm },
});
