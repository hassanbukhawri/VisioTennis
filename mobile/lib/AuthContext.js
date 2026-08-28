// Accounts (Phase 2, 2026-08-07). Session token lives in expo-secure-store
// (Keystore/Keychain-backed) -- deliberately not the existing on-device
// SQLite settings table (see historyRepository.js), since a login token
// shouldn't sit in a plain on-device file the way the anonymous `role`
// setting does. On mount, a stored token is validated against
// GET /api/v1/auth/me (not just trusted) so a revoked/expired token found
// on a stale install doesn't silently act as if logged in.
import { createContext, useContext, useEffect, useState } from "react";
import * as SecureStore from "expo-secure-store";

import { apiFetch } from "./api";

const TOKEN_KEY = "visiotennis_auth_token";

const AuthContext = createContext(null);

export function AuthProvider({ children }) {
  const [token, setToken] = useState(null);
  const [user, setUser] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    (async () => {
      const stored = await SecureStore.getItemAsync(TOKEN_KEY);
      if (stored) {
        try {
          const me = await apiFetch("/api/v1/auth/me", { token: stored });
          setToken(stored);
          setUser(me);
        } catch (e) {
          // Stored token is invalid/expired (backend already deleted it if
          // expired) -- clear it locally too rather than get stuck retrying
          // a token that will never work.
          await SecureStore.deleteItemAsync(TOKEN_KEY);
        }
      }
      setLoading(false);
    })();
  }, []);

  async function signup(email, password, name) {
    const data = await apiFetch("/api/v1/auth/signup", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ email, password, name: name || null }),
    });
    await SecureStore.setItemAsync(TOKEN_KEY, data.token);
    setToken(data.token);
    setUser(data.user);
  }

  async function login(email, password) {
    const data = await apiFetch("/api/v1/auth/login", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ email, password }),
    });
    await SecureStore.setItemAsync(TOKEN_KEY, data.token);
    setToken(data.token);
    setUser(data.user);
  }

  // Coaching-tone profile (2026-08-10, mobile/app/profile-setup.js) --
  // PATCHes the account and updates local `user` state from the response,
  // same pattern signup/login already use (server response is the source of
  // truth for what got saved, not an optimistic local merge).
  async function updateProfile(fields) {
    const updated = await apiFetch("/api/v1/auth/profile", {
      token,
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(fields),
    });
    setUser(updated);
    return updated;
  }

  // Coach avatar naming (2026-08-23, mobile/app/(tabs)/settings.js,
  // StrokeChat's CoachBubble) -- separate endpoint from updateProfile above,
  // does not require role to already be set. Same pattern: server response
  // is the source of truth for what got saved.
  async function updateCoachName(coachName) {
    const updated = await apiFetch("/api/v1/auth/coach-name", {
      token,
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ coach_name: coachName || null }),
    });
    setUser(updated);
    return updated;
  }

  async function logout() {
    if (token) {
      // Best-effort -- if the backend call fails (e.g. offline), still log
      // out locally rather than trap the user in a broken logged-in state.
      await apiFetch("/api/v1/auth/logout", { method: "POST", token }).catch(() => {});
    }
    await SecureStore.deleteItemAsync(TOKEN_KEY);
    setToken(null);
    setUser(null);
  }

  return (
    <AuthContext.Provider value={{ token, user, loading, signup, login, logout, updateProfile, updateCoachName }}>
      {children}
    </AuthContext.Provider>
  );
}

export function useAuth() {
  const ctx = useContext(AuthContext);
  if (!ctx) {
    throw new Error("useAuth must be used within an AuthProvider");
  }
  return ctx;
}
