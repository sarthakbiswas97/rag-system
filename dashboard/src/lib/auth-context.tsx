"use client";

import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useRef,
  useState,
} from "react";
import type { ReactNode } from "react";
import { getMe, type TenantOut } from "./api";

interface AuthState {
  apiKey: string | null;
  tenant: TenantOut | null;
  isLoading: boolean;
  login: (key: string) => void;
  logout: () => void;
}

const AuthContext = createContext<AuthState>({
  apiKey: null,
  tenant: null,
  isLoading: true,
  login: () => {},
  logout: () => {},
});

function readStoredKey(): string | null {
  if (typeof window === "undefined") return null;
  return localStorage.getItem("api_key");
}

export function AuthProvider({ children }: { children: ReactNode }) {
  const [apiKey, setApiKey] = useState<string | null>(readStoredKey);
  const [tenant, setTenant] = useState<TenantOut | null>(null);
  const [isLoading, setIsLoading] = useState(() => readStoredKey() !== null);
  const hasFetched = useRef(false);

  useEffect(() => {
    if (!apiKey || hasFetched.current) return;
    hasFetched.current = true;

    let cancelled = false;
    getMe()
      .then((me) => {
        if (!cancelled) setTenant(me);
      })
      .catch(() => {
        if (!cancelled) {
          localStorage.removeItem("api_key");
          setApiKey(null);
          setTenant(null);
        }
      })
      .finally(() => {
        if (!cancelled) setIsLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [apiKey]);

  const login = useCallback((key: string) => {
    localStorage.setItem("api_key", key);
    hasFetched.current = false;
    setApiKey(key);
    setIsLoading(true);
  }, []);

  const logout = useCallback(() => {
    localStorage.removeItem("api_key");
    hasFetched.current = false;
    setApiKey(null);
    setTenant(null);
  }, []);

  return (
    <AuthContext.Provider value={{ apiKey, tenant, isLoading, login, logout }}>
      {children}
    </AuthContext.Provider>
  );
}

export function useAuth() {
  return useContext(AuthContext);
}
