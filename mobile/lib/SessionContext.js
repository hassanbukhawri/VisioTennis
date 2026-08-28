// Hands off the just-computed analysis result from the Processing screen to
// the Results screen without pushing the full JSON payload through router
// params -- per mvp/DESIGN.md's Phase B proposal, Expo Router's URL params
// are for small serializable values, not a full stroke+coaching+diagnostics
// response object.
import { createContext, useContext, useState } from "react";

const SessionContext = createContext(null);

export function SessionProvider({ children }) {
  const [currentResult, setCurrentResult] = useState(null);
  return (
    <SessionContext.Provider value={{ currentResult, setCurrentResult }}>
      {children}
    </SessionContext.Provider>
  );
}

export function useSession() {
  const ctx = useContext(SessionContext);
  if (!ctx) {
    throw new Error("useSession must be used within a SessionProvider");
  }
  return ctx;
}
