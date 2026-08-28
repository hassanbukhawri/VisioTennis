import { Stack } from "expo-router";
import { ActivityIndicator, StyleSheet, View } from "react-native";

import { AuthProvider, useAuth } from "../lib/AuthContext";
import { SessionProvider } from "../lib/SessionContext";
import { initHistoryDb } from "../lib/historyRepository";
import { fonts, fontSizes, useTheme } from "../lib/theme";

initHistoryDb();

// Phase 2 (2026-08-07): (tabs)/processing/results now require a logged-in
// user (history is per-account); login/signup require the opposite.
// Stack.Protected (expo-router 6, confirmed present in the installed
// version before using it) handles the redirect itself when `guard` flips
// -- no manual router.replace-on-mount juggling needed here.
function RootNavigator() {
  const { token, loading } = useAuth();
  // System-adaptive theme (2026-08-20, task 2): reads the OS appearance
  // setting live via useColorScheme() inside useTheme(), so this re-renders
  // with the other palette the instant the OS setting changes -- verified
  // for real on the vt_test emulator, not just assumed from the hook's docs.
  const { colors } = useTheme();

  // Waiting on the stored-token check (AuthContext validates it against
  // GET /api/v1/auth/me before trusting it) -- render nothing decisive
  // yet, rather than flash the login screen for an already-logged-in user.
  if (loading) {
    return (
      <View style={[styles.loadingScreen, { backgroundColor: colors.bg }]}>
        <ActivityIndicator size="large" color={colors.navy} />
      </View>
    );
  }

  return (
    <Stack
      screenOptions={{
        headerStyle: { backgroundColor: colors.surface },
        headerShadowVisible: false,
        headerTitleStyle: { fontFamily: fonts.serif, fontSize: fontSizes.h2, color: colors.navyInk },
        headerTintColor: colors.navy,
      }}
    >
      <Stack.Protected guard={!!token}>
        <Stack.Screen name="(tabs)" options={{ headerShown: false }} />
        <Stack.Screen name="processing" options={{ title: "Processing" }} />
        <Stack.Screen name="results" options={{ title: "Results" }} />
        <Stack.Screen name="profile-setup" options={{ title: "Coaching Profile" }} />
      </Stack.Protected>
      <Stack.Protected guard={!token}>
        <Stack.Screen name="login" options={{ headerShown: false }} />
        <Stack.Screen name="signup" options={{ headerShown: false }} />
      </Stack.Protected>
    </Stack>
  );
}

export default function RootLayout() {
  return (
    <AuthProvider>
      <SessionProvider>
        <RootNavigator />
      </SessionProvider>
    </AuthProvider>
  );
}

const styles = StyleSheet.create({
  loadingScreen: { flex: 1, alignItems: "center", justifyContent: "center" }, // bg color applied inline (theme-reactive)
});
