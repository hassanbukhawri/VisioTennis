// Horizontal bar chart -- count of sessions per predicted stroke class.
// Plain Views, no SVG needed for bars. `data` is [{ label, count }, ...],
// already sorted/aggregated by the caller (History screen).
import { useMemo } from "react";
import { StyleSheet, Text, View } from "react-native";

import { fontSizes, radii, spacing, useTheme } from "../../lib/theme";

export default function StrokeBreakdownChart({ data }) {
  const { colors } = useTheme();
  const styles = useMemo(() => makeStyles(colors), [colors]);
  const maxCount = Math.max(...data.map((d) => d.count), 1);

  return (
    <View style={styles.container}>
      {data.map((d) => (
        <View key={d.label} style={styles.row}>
          <Text style={styles.label}>{d.label}</Text>
          <View style={styles.trackWrap}>
            <View style={[styles.bar, { flex: d.count / maxCount }]} />
            <View style={{ flex: 1 - d.count / maxCount }} />
          </View>
          <Text style={styles.count}>{d.count}</Text>
        </View>
      ))}
    </View>
  );
}

const makeStyles = (colors) => StyleSheet.create({
  container: { gap: spacing.sm },
  row: { flexDirection: "row", alignItems: "center", gap: spacing.sm },
  label: { width: 80, fontSize: fontSizes.small, color: colors.text, textTransform: "capitalize" },
  trackWrap: { flex: 1, flexDirection: "row", height: 18 },
  bar: { backgroundColor: colors.teal, borderRadius: radii.sm, minWidth: 4 },
  count: { width: 24, fontSize: fontSizes.small, color: colors.textSoft, textAlign: "right" },
});
