// Pure computation for the History screen's Insights charts (2026-08-08).
// Kept separate from the screen component so the grouping/filtering logic
// is easy to read (and re-check) on its own, without JSX in the way.
export const MIN_SESSIONS_FOR_INSIGHTS = 3;
export const MIN_SAME_CLASS_FOR_TREND = 3;

function chronological(sessions) {
  return [...sessions].sort((a, b) => new Date(a.createdAt) - new Date(b.createdAt));
}

// [{ label, count }, ...] sorted by count desc, ties broken alphabetically
// for determinism.
export function computeStrokeBreakdown(sessions) {
  const counts = {};
  for (const s of sessions) counts[s.strokeClass] = (counts[s.strokeClass] || 0) + 1;
  return Object.entries(counts)
    .map(([label, count]) => ({ label, count }))
    .sort((a, b) => b.count - a.count || a.label.localeCompare(b.label));
}

// Chronological confidence values, oldest first, so the trend line reads
// left-to-right as "over time."
export function computeConfidenceTrend(sessions) {
  return chronological(sessions.filter((s) => s.confidence != null)).map((s) => s.confidence);
}

// The stroke class with the most sessions -- ties broken by whichever class's
// most recent session is more recent, so "your most common stroke" leans
// toward what the user has been doing lately, not an old one-time streak.
export function findMostCommonClass(sessions) {
  const byClass = {};
  for (const s of sessions) {
    (byClass[s.strokeClass] ||= []).push(s);
  }
  let best = null;
  for (const [label, list] of Object.entries(byClass)) {
    const mostRecent = Math.max(...list.map((s) => new Date(s.createdAt).getTime()));
    if (!best || list.length > best.count || (list.length === best.count && mostRecent > best.mostRecent)) {
      best = { label, count: list.length, mostRecent, sessions: list };
    }
  }
  return best;
}

// Biomechanics values are percentile-ranked PER STROKE CLASS (see
// ir/biomechanics_percentiles.py) -- charting the raw weight_transfer_ratio
// across mixed stroke types would silently splice together different
// baselines into one misleading line. Filters to the user's single most
// common class instead, and returns which class that was so the screen can
// caption it (never filter silently -- explicit product decision,
// PROJECT_LOG.md 2026-08-08).
export function computeWeightTransferTrend(sessions) {
  const dominant = findMostCommonClass(sessions);
  if (!dominant) return null;
  const values = chronological(dominant.sessions.filter((s) => s.weightTransferRatio != null)).map(
    (s) => s.weightTransferRatio
  );
  return { strokeClass: dominant.label, values };
}
