// Shared design tokens for the app. `colors` below is the static LIGHT
// palette (kept for screens not yet converted to the reactive hook -- see
// the system-adaptive theme proposal, approved 2026-08-20). Values here
// already include that proposal's two corrections: `border`/`borderStrong`
// strengthened to clear real WCAG 3:1 (checked directly: surface-vs-bg on
// their own sit at ~1.1-1.15:1, so the border is the sole delimiter for
// cards/inputs, not decorative), and `teal`* renamed in spirit to "accent"
// but kept under its original key names here to avoid touching every call
// site in this pass -- ResultView.js/StrokeChat.js's `colors.tealTint`/
// `colors.tealDeep` usage is unchanged, just repointed to the new hex values.
//
// For screens that need to actually follow the OS light/dark setting, use
// `useTheme()` below instead of this static export -- see app/(tabs)/index.js
// and both root/_layout.js files for the converted pattern. Only those are
// converted so far (system-adaptive theme task 2, 2026-08-20): the rest of
// the app still renders the static light palette regardless of OS setting,
// pending the full re-skin pass. Any screen converted later must also
// rename its `colors.surfaceAlt`-as-button-text usage to `colors.onPrimary`
// -- `surfaceAlt` collides between two real, different meanings in the
// existing code (an actual alternate-surface background in
// history/index.js's `insightsPlaceholder`, vs. "text sitting on a
// colors.navy fill" everywhere else) and only `onPrimary` is theme-correct
// for the second meaning once `navy` itself becomes a dark-mode-only slate.
import { Platform, useColorScheme } from "react-native";

const lightColors = {
  navy: "#16233F",
  navyDeep: "#0E1830",
  navyInk: "#1E2430",
  onPrimary: "#FBF9F3",

  teal: "#0AA37A",
  tealDeep: "#076B4F",
  tealTint: "#DEF5EC",
  // Neutral decorative-swatch fill (e.g. the stroke-picker placeholder cards) --
  // deliberately NOT tealTint. Found during the theme proposal's accent-usage
  // audit that index.js's strokeImagePlaceholder was reusing tealTint purely as
  // decoration, with no positive/grounded meaning attached -- diluting the one
  // signal accent is meant to carry. Fixed here, not just in the mockup.
  placeholderTint: "#ECE6D6",

  bg: "#F3EFE6",
  surface: "#FFFFFF",
  surfaceAlt: "#FBF9F3",

  border: "#8A7857",
  borderStrong: "#6C5C43",

  text: "#211F1A",
  textSoft: "#6B6558",
  textFaint: "#726B5B",

  error: "#B3261E",
  errorTint: "#F7E7E5",

  caveat: "#8A5E13",
  caveatTint: "#F3E7CE",
};

// Dark theme: `navy` is repurposed as a neutral slate here, NOT the literal
// navy hex -- a navy-on-near-black button/tab-icon/spinner would nearly
// disappear, and reusing the accent (teal) color for that role would have
// made accent common instead of rare, diluting the one signal it's meant to
// carry (confidence/quality pill's positive state + the grounded chat
// bubble -- unchanged, identically narrow in both themes). `navyInk`
// (heading text) and `onPrimary` (text on the `navy`-token fill) both flip
// to keep every existing `colors.navy`-filled element readable without
// touching its call site.
const darkColors = {
  navy: "#B7C0D4",
  navyDeep: "#8F9BC2",
  navyInk: "#E9ECF5",
  onPrimary: "#16233F",

  teal: "#23E0A8",
  tealDeep: "#23E0A8",
  tealTint: "#0F2E24",
  placeholderTint: "#22304A",

  bg: "#0B0F1A",
  surface: "#141B2B",
  surfaceAlt: "#1B2436",

  border: "#5F6F99",
  borderStrong: "#7788A6",

  text: "#F3F4F8",
  textSoft: "#AEB4C4",
  textFaint: "#7C859A",

  error: "#FF6B60",
  errorTint: "#331211",

  caveat: "#E8B34A",
  caveatTint: "#33270E",
};

export const colors = lightColors;

// Reactive theme hook -- the actual system-adaptive path. `useColorScheme()`
// is documented to re-render on a live OS appearance change (not just a
// cold-start read) -- see PROJECT_LOG.md for whether that was confirmed for
// real on the emulator, rather than trusted from the docs alone.
export function useTheme() {
  const scheme = useColorScheme();
  return { colors: scheme === "dark" ? darkColors : lightColors, scheme: scheme === "dark" ? "dark" : "light" };
}

export const fonts = {
  serif: Platform.select({ ios: "Georgia", android: "serif", default: "serif" }),
  sans: undefined, // no override -- inherits platform default (SF on iOS, Roboto on Android)
};

export const fontSizes = {
  display: 30,
  h1: 22,
  h2: 18,
  body: 15,
  small: 13,
  caption: 11,
};

export const spacing = {
  xs: 4,
  sm: 8,
  md: 16,
  lg: 24,
  xl: 32,
  xxl: 48,
};

export const radii = {
  sm: 6,
  md: 12,
  lg: 20,
  pill: 999,
};
