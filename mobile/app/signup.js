// Signup -- Phase 2 (2026-08-07). Deliberately minimal: email + password +
// optional name, per the approved plan's "nothing beyond what's actually
// used today" data-model rule (PROJECT_LOG.md) -- playing_hand exists on
// the users table but has no UI yet since nothing reads it today; adding
// that editing UI is a separate, later change, not silently smuggled in here.
import { useMemo, useState } from "react";
import { useRouter } from "expo-router";
import {
  KeyboardAvoidingView, Platform, Pressable, ScrollView, StyleSheet, Text, TextInput, View,
} from "react-native";

import { extractErrorMessage } from "../lib/api";
import { useAuth } from "../lib/AuthContext";
import { fonts, fontSizes, radii, spacing, useTheme } from "../lib/theme";

export default function SignupScreen() {
  const router = useRouter();
  const { signup } = useAuth();
  const { colors } = useTheme();
  const styles = useMemo(() => makeStyles(colors), [colors]);
  const [name, setName] = useState("");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState(null);
  const [submitting, setSubmitting] = useState(false);

  async function handleSubmit() {
    setError(null);
    setSubmitting(true);
    try {
      await signup(email.trim(), password, name.trim());
      router.replace("/profile-setup");
    } catch (e) {
      setError(extractErrorMessage(e));
    } finally {
      setSubmitting(false);
    }
  }

  const canSubmit = email.trim().length > 0 && password.length >= 8 && !submitting;

  return (
    <KeyboardAvoidingView style={styles.screen} behavior={Platform.OS === "ios" ? "padding" : undefined}>
      <ScrollView contentContainerStyle={styles.container} keyboardShouldPersistTaps="handled">
        <Text style={styles.title}>VisioTennis</Text>
        <Text style={styles.subtitle}>Create an account to save your stroke history</Text>

        <View style={styles.field}>
          <Text style={styles.label}>Name (optional)</Text>
          <TextInput
            style={styles.input}
            value={name}
            onChangeText={setName}
            autoCapitalize="words"
            placeholder="Your name"
            placeholderTextColor={colors.textFaint}
          />
        </View>

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
            placeholder="At least 8 characters"
            placeholderTextColor={colors.textFaint}
          />
        </View>

        {error ? <Text style={styles.error}>{error}</Text> : null}

        <Pressable
          onPress={handleSubmit}
          disabled={!canSubmit}
          style={({ pressed }) => [styles.btnPrimary, (pressed || !canSubmit) && styles.btnPressed]}
        >
          <Text style={styles.btnPrimaryText}>{submitting ? "Creating account..." : "Sign Up"}</Text>
        </Pressable>

        <Pressable onPress={() => router.replace("/login")} style={styles.linkRow}>
          <Text style={styles.linkText}>
            Already have an account? <Text style={styles.linkTextStrong}>Log in</Text>
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
