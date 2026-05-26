"use client";

import {
  createContext,
  useCallback,
  useContext,
  useEffect,
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

export function AuthProvider({ children }: { children: ReactNode }) {
  const [apiKey, setApiKey] = useState<string | null>(null);
  const [tenant, setTenant] = useState<TenantOut | null>(null);
  const [isLoading, setIsLoading] = useState(true);

  const fetchTenant = useCallback(async () => {
    try {
      const me = await getMe();
      setTenant(me);
    } catch {
      localStorage.removeItem("api_key");
      setApiKey(null);
      setTenant(null);
    } finally {
      setIsLoading(false);
    }
  }, []);

  useEffect(() => {
    const stored = localStorage.getItem("api_key");
    if (stored) {
      setApiKey(stored);
    } else {
      setIsLoading(false);
    }
  }, []);

  useEffect(() => {
    if (apiKey) {
      fetchTenant();
    }
  }, [apiKey, fetchTenant]);

  const login = useCallback((key: string) => {
    localStorage.setItem("api_key", key);
    setApiKey(key);
  }, []);

  const logout = useCallback(() => {
    localStorage.removeItem("api_key");
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
