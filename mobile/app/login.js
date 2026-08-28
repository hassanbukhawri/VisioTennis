// Login -- Phase 2 (2026-08-07). Gated behind Stack.Protected in
// app/_layout.js (redirects here automatically when logged out), so a
// successful login just needs to update AuthContext's state; also
// explicitly navigates to Home so the transition doesn't depend on timing.
import { useMemo, useState } from "react";
import { useRouter } from "expo-router";
import {
  KeyboardAvoidingView, Platform, Pressable, ScrollView, StyleSheet, Text, TextInput, View,
} from "react-native";

import { extractErrorMessage } from "../lib/api";
import { useAuth } from "../lib/AuthContext";
import { fonts, fontSizes, radii, spacing, useTheme } from "../lib/theme";

export default function LoginScreen() {
  const router = useRouter();
  const { login } = useAuth();
  const { colors } = useTheme();
  const styles = useMemo(() => makeStyles(colors), [colors]);
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState(null);
  const [submitting, setSubmitting] = useState(false);

  async function handleSubmit() {
    setError(null);
    setSubmitting(true);
    try {
      await login(email.trim(), password);
      router.replace("/(tabs)");
    } catch (e) {
      setError(extractErrorMessage(e));
    } finally {
      setSubmitting(false);
    }
  }

  const canSubmit = email.trim().length > 0 && password.length > 0 && !submitting;

  return (
    <KeyboardAvoidingView style={styles.screen} behavior={Platform.OS === "ios" ? "padding" : undefined}>
      <ScrollView contentContainerStyle={styles.container} keyboardShouldPersistTaps="handled">
        <Text style={styles.title}>VisioTennis</Text>
        <Text style={styles.subtitle}>Log in to see your stroke history</Text>

        <View style={styles.field}>
          <Text style={styles.label}>Email</Text>
          <TextInput
            style={styles.input}
            value={email}
            onChangeText={setEmail}
            autoCapitalize="none"
            autoCorrect={false}
            keyboardType="email-address"
            placeholder="you@example.com"
            placeholderTextColor={colors.textFaint}
          />
        </View>

        <View style={styles.field}>
          <Text style={styles.label}>Password</Text>
          <TextInput
            style={styles.input}
            value={password}
            onChangeText={setPassword}
            secureTextEntry
            placeholder="Password"
            placeholderTextColor={colors.textFaint}
          />
        </View>

        {error ? <Text style={styles.error}>{error}</Text> : null}

        <Pressable
          onPress={handleSubmit}
          disabled={!canSubmit}
          style={({ pressed }) => [styles.btnPrimary, (pressed || !canSubmit) && styles.btnPressed]}
        >
          <Text style={styles.btnPrimaryText}>{submitting ? "Logging in..." : "Log In"}</Text>
        </Pressable>

        <Pressable onPress={() => router.replace("/signup")} style={styles.linkRow}>
          <Text style={styles.linkText}>
            No account yet? <Text style={styles.linkTextStrong}>Sign up</Text>
          </Text>
        </Pressable>
      </ScrollView>
    </KeyboardAvoidingView>
  );
}

const makeStyles = (colors) => StyleSheet.create({
  screen: { flex: 1, backgroundColor: colors.bg },
  container: { flexGrow: 1, justifyContent: "center", padding: spacing.lg },
  title: {
    fontFamily: fonts.serif, fontSize: fontSizes.display, color: colors.navyInk,
    textAlign: "center", marginBottom: spacing.xs,
  },
  subtitle: { fontSize: fontSizes.small, color: colors.textSoft, textAlign: "center", marginBottom: spacing.xl },
  field: { marginBottom: spacing.md },
  label: { fontSize: fontSizes.small, fontWeight: "600", color: colors.textSoft, marginBottom: spacing.xs },
  input: {
    backgroundColor: colors.surface,
    borderWidth: 1,
    borderColor: colors.border,
    borderRadius: radii.md,
    paddingVertical: 12,
    paddingHorizontal: spacing.md,
    fontSize: fontSizes.body,
    color: colors.text,
  },
  error: { color: colors.error, fontSize: fontSizes.small, marginTop: spacing.xs, marginBottom: spacing.sm },
  btnPrimary: {
    backgroundColor: colors.navy,
    borderRadius: radii.pill,
    paddingVertical: 13,
    paddingHorizontal: spacing.lg,
    marginTop: spacing.md,
  },
  btnPressed: { opacity: 0.6 },
  btnPrimaryText: { color: colors.onPrimary, fontSize: fontSizes.small, fontWeight: "600", textAlign: "center" },
  linkRow: { marginTop: spacing.lg, alignItems: "center" },
  linkText: { fontSize: fontSizes.small, color: colors.textSoft },
  linkTextStrong: { color: colors.navy, fontWeight: "600" },
});
