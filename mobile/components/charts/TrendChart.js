// Minimal line chart -- a chronological series of numeric values, connected
// by a single polyline with a dot per point. No axes/gridlines/zoom -- for
// the 3-15 points this app's trial-stage session counts realistically
// produce, a full charting-library feature set would be overbuilt; the
// min/max caption below the line gives just enough numeric context.
import { useMemo, useState } from "react";
import { StyleSheet, Text, View } from "react-native";
import Svg, { Circle, Polyline } from "react-native-svg";

import { fontSizes, spacing, useTheme } from "../../lib/theme";

const CHART_HEIGHT = 90;
const POINT_PADDING = 10; // keeps the topmost/bottommost dot from touching the edge

export default function TrendChart({ values, color, formatValue = (v) => `${v}` }) {
  const { colors } = useTheme();
  const styles = useMemo(() => makeStyles(colors), [colors]);
  const lineColor = color ?? colors.teal;
  const [width, setWidth] = useState(0);

  const min = Math.min(...values);
  const max = Math.max(...values);
  const range = max - min || 1; // avoid divide-by-zero when every value is identical

  const points = values.map((v, i) => {
    const x = values.length > 1 ? (i / (values.length - 1)) * width : width / 2;
    const y =
      CHART_HEIGHT -
      POINT_PADDING -
      ((v - min) / range) * (CHART_HEIGHT - POINT_PADDING * 2);
    return { x, y };
  });

  return (
    <View>
      <View style={styles.chartArea} onLayout={(e) => setWidth(e.nativeEvent.layout.width)}>
        {width > 0 && (
          <Svg width={width} height={CHART_HEIGHT}>
            <Polyline
              points={points.map((p) => `${p.x},${p.y}`).join(" ")}
              fill="none"
              stroke={lineColor}
              strokeWidth={2}
            />
            {points.map((p, i) => (
              <Circle key={i} cx={p.x} cy={p.y} r={4} fill={lineColor} />
            ))}
          </Svg>
        )}
      </View>
      <Text style={styles.rangeCaption}>
        {formatValue(min)} - {formatValue(max)}
      </Text>
    </View>
  );
}

const makeStyles = (colors) => StyleSheet.create({
  chartArea: { height: CHART_HEIGHT, width: "100%" },
  rangeCaption: { fontSize: fontSizes.caption, color: colors.textFaint, marginTop: spacing.xs },
});
