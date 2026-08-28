// History -- lists on-device sessions (see lib/historyRepository.js). Reloads
// every time this tab gains focus, so a session just saved by Processing
// shows up without needing an app restart.
import { useCallback, useMemo, useState } from "react";
import { useFocusEffect, useRouter } from "expo-router";
import { FlatList, StyleSheet, Text, TouchableOpacity, View } from "react-native";

import StrokeBreakdownChart from "../../../components/charts/StrokeBreakdownChart";
import TrendChart from "../../../components/charts/TrendChart";
import { CONFIDENCE_CAVEAT_THRESHOLD } from "../../../components/ResultView";
import { useAuth } from "../../../lib/AuthContext";
import { listSessions } from "../../../lib/historyRepository";
import {
  MIN_SAME_CLASS_FOR_TREND,
  MIN_SESSIONS_FOR_INSIGHTS,
  computeConfidenceTrend,
  computeStrokeBreakdown,
  computeWeightTransferTrend,
} from "../../../lib/historyInsights";
import { fonts, fontSizes, radii, spacing, useTheme } from "../../../lib/theme";

function InsightsSection({ sessions, colors, styles }) {
  if (sessions.length < MIN_SESSIONS_FOR_INSIGHTS) {
    return (
      <View style={styles.insightsPlaceholder}>
        <Text style={styles.insightsPlaceholderText}>
          Insights appear after a few more sessions -- come back once you've logged a couple more.
        </Text>
      </View>
    );
  }

  const breakdown = computeStrokeBreakdown(sessions);
  const confidenceValues = computeConfidenceTrend(sessions);
  const weightTrend = computeWeightTransferTrend(sessions);
  const hasWeightTrend = weightTrend && weightTrend.values.length >= MIN_SAME_CLASS_FOR_TREND;

  return (
    <View style={styles.insightsCard}>
      <Text style={styles.sectionTitle}>Insights</Text>

      <Text style={styles.cardEyebrow}>Stroke breakdown</Text>
      <StrokeBreakdownChart data={breakdown} />

      <Text style={[styles.cardEyebrow, styles.chartSpacing]}>Confidence over time</Text>
      <TrendChart values={confidenceValues} color={colors.teal} formatValue={(v) => `${(v * 100).toFixed(0)}%`} />

      <Text style={[styles.cardEyebrow, styles.chartSpacing]}>Weight transfer over time</Text>
      {hasWeightTrend ? (
        <>
          <Text style={styles.chartCaption}>
            Showing your {weightTrend.strokeClass.charAt(0).toUpperCase() + weightTrend.strokeClass.slice(1)}{" "}
            sessions
          </Text>
          <TrendChart values={weightTrend.values} color={colors.navy} formatValue={(v) => v.toFixed(2)} />
        </>
      ) : (
        <Text style={styles.insufficientText}>
          Not enough sessions of one stroke type yet -- once you've logged 3 of the same stroke, a trend
          will appear here.
        </Text>
      )}
    </View>
  );
}

export default function HistoryScreen() {
  const router = useRouter();
  const { token } = useAuth();
  const { colors } = useTheme();
  const styles = useMemo(() => makeStyles(colors), [colors]);
  const [sessions, setSessions] = useState([]);
  const [loaded, setLoaded] = useState(false);

  useFocusEffect(
    useCallback(() => {
      listSessions(token).then(setSessions).finally(() => setLoaded(true));
    }, [token])
  );

  if (loaded && sessions.length === 0) {
    return (
      <View style={styles.centered}>
        <Text style={styles.emptyText}>No sessions yet -- analyze a stroke from Home to see it here.</Text>
      </View>
    );
  }

  // De-genericizing pass (section 05, approved mockup): the most recent
  // session -- what a returning player actually came here to see -- gets a
  // headline treatment instead of the same bordered box as every other row.
  // list_history (backend/app.py) already orders by created_at DESC, so
  // sessions[0] is genuinely the latest, not an assumption.
  const [mostRecent, ...earlier] = sessions;

  return (
    <FlatList
      style={styles.list}
      contentContainerStyle={styles.listContent}
      data={earlier}
      keyExtractor={(item) => item.id}
      ListHeaderComponent={
        sessions.length > 0 ? (
          <View style={styles.headerSection}>
            <InsightsSection sessions={sessions} colors={colors} styles={styles} />
            <Text style={styles.eyebrowLabel}>Most recent</Text>
            <TouchableOpacity
              style={styles.recentRow}
              onPress={() => router.push(`/(tabs)/history/${mostRecent.id}`)}
            >
              <View style={styles.recentHeadingLine}>
                <Text style={styles.recentTitle}>{mostRecent.strokeClass}</Text>
                {mostRecent.confidence != null ? (
                  <View style={[styles.pill, mostRecent.confidence < CONFIDENCE_CAVEAT_THRESHOLD && styles.pillCaveat]}>
                    <Text
                      style={[
                        styles.pillText,
                        mostRecent.confidence < CONFIDENCE_CAVEAT_THRESHOLD && styles.pillTextCaveat,
                      ]}
                    >
                      {(mostRecent.confidence * 100).toFixed(0)}%
                    </Text>
                  </View>
                ) : null}
              </View>
              <Text style={styles.recentSubtitle}>{new Date(mostRecent.createdAt).toLocaleString()}</Text>
            </TouchableOpacity>
            {earlier.length > 0 ? <Text style={[styles.eyebrowLabel, styles.sectionSpacing]}>Earlier</Text> : null}
          </View>
        ) : null
      }
      renderItem={({ item }) => {
        const confident = item.confidence != null && item.confidence >= CONFIDENCE_CAVEAT_THRESHOLD;
        return (
          <TouchableOpacity
            style={styles.compactRow}
            onPress={() => router.push(`/(tabs)/history/${item.id}`)}
          >
            <Text style={styles.compactRowTitle}>{item.strokeClass}</Text>
            <Text style={styles.compactRowMeta}>
              {item.confidence != null ? `${(item.confidence * 100).toFixed(0)}%` : ""}
              {item.confidence != null ? " · " : ""}
              {new Date(item.createdAt).toLocaleDateString()}
            </Text>
            {!confident ? <View style={styles.compactRowCaveatDot} /> : null}
          </TouchableOpacity>
        );
      }}
    />
  );
}

const makeStyles = (colors) => StyleSheet.create({
  list: { flex: 1, backgroundColor: colors.bg },
  listContent: { padding: spacing.md },
  centered: { flex: 1, backgroundColor: colors.bg, alignItems: "center", justifyContent: "center", paddingHorizontal: spacing.lg },
  emptyText: { fontSize: fontSizes.small, color: colors.textSoft, textAlign: "center" },
  headerSection: { marginBottom: spacing.md },
  sectionTitle: {
    fontFamily: fonts.serif,
    fontSize: fontSizes.h1,
    color: colors.navyInk,
    marginBottom: spacing.sm,
  },
  insightsCard: {
    backgroundColor: colors.surface,
    borderWidth: 1,
    borderColor: colors.border,
    borderRadius: radii.md,
    padding: spacing.md,
    marginBottom: spacing.lg,
  },
  insightsPlaceholder: {
    backgroundColor: colors.surfaceAlt,
    borderRadius: radii.md,
    padding: spacing.md,
    marginBottom: spacing.lg,
  },
  insightsPlaceholderText: { fontSize: fontSizes.small, color: colors.textSoft, lineHeight: 19 },
  cardEyebrow: {
    fontSize: fontSizes.caption,
    fontWeight: "600",
    letterSpacing: 0.6,
    textTransform: "uppercase",
    color: colors.textFaint,
    marginBottom: spacing.xs,
  },
  chartSpacing: { marginTop: spacing.md },
  chartCaption: { fontSize: fontSizes.caption, color: colors.textSoft, marginBottom: spacing.xs },
  insufficientText: { fontSize: fontSizes.small, color: colors.textSoft, lineHeight: 19 },
  sectionSpacing: { marginTop: spacing.lg },
  // Same small-caps recipe as cardEyebrow, under its own name here since it
  // labels a session grouping ("Most recent"/"Earlier") rather than a
  // top-level section.
  eyebrowLabel: {
    fontSize: fontSizes.caption,
    fontWeight: "600",
    letterSpacing: 0.6,
    textTransform: "uppercase",
    color: colors.textFaint,
    marginBottom: spacing.sm,
  },
  // Most-recent session (De-genericizing pass, section 05) -- headline
  // treatment, no border box, since it's the first thing on the screen.
  recentRow: { marginBottom: spacing.md },
  recentHeadingLine: { flexDirection: "row", alignItems: "baseline", flexWrap: "wrap", gap: spacing.sm },
  recentTitle: { fontFamily: fonts.serif, fontSize: 26, color: colors.navyInk, textTransform: "capitalize" },
  recentSubtitle: { fontSize: fontSizes.small, color: colors.textSoft, marginTop: 2 },
  // Older sessions (De-genericizing pass, section 05) -- compressed into a
  // denser list, thin border-bottom instead of a full bordered card per row.
  compactRow: {
    flexDirection: "row",
    alignItems: "center",
    paddingVertical: 11,
    borderBottomWidth: 1,
    borderBottomColor: colors.border,
    gap: spacing.sm,
  },
  compactRowTitle: {
    fontFamily: fonts.serif,
    fontSize: 15,
    color: colors.navyInk,
    textTransform: "capitalize",
    flexShrink: 1,
  },
  compactRowMeta: { fontSize: fontSizes.caption, color: colors.textFaint, flex: 1, textAlign: "right" },
  compactRowCaveatDot: { width: 6, height: 6, borderRadius: 3, backgroundColor: colors.caveat },
  pill: { paddingVertical: 2, paddingHorizontal: spacing.sm, borderRadius: radii.pill, backgroundColor: colors.tealTint },
  pillText: { fontSize: fontSizes.caption, fontWeight: "600", color: colors.tealDeep },
  pillCaveat: { backgroundColor: colors.caveatTint },
  pillTextCaveat: { color: colors.caveat },
});
