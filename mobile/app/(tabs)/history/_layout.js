import { Stack } from "expo-router";

import { fonts, fontSizes, useTheme } from "../../../lib/theme";

export default function HistoryStackLayout() {
  const { colors } = useTheme();
  return (
    <Stack
      screenOptions={{
        headerStyle: { backgroundColor: colors.surface },
        headerShadowVisible: false,
        headerTitleStyle: { fontFamily: fonts.serif, fontSize: fontSizes.h2, color: colors.navyInk },
        headerTintColor: colors.navy,
      }}
    >
      <Stack.Screen name="index" options={{ title: "History" }} />
      <Stack.Screen name="[sessionId]" options={{ title: "Session" }} />
    </Stack>
  );
}
