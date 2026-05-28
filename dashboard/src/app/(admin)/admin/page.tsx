"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import {
  adminListTenants,
  adminCreateTenant,
  adminDeleteTenant,
  type TenantOut,
  type RegisterResponse,
} from "@/lib/api";

const ADMIN_KEY_STORAGE = "admin_api_key";

function AdminLogin({
  onLogin,
}: {
  onLogin: (key: string) => void;
}) {
  const [key, setKey] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    const trimmed = key.trim();
    if (!trimmed) return;

    setLoading(true);
    setError(null);

    try {
      await adminListTenants(trimmed);
      onLogin(trimmed);
    } catch {
      setError("Invalid admin key");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="flex min-h-screen items-center justify-center bg-gray-50">
      <div className="w-full max-w-sm rounded-lg border bg-white p-6 shadow-sm">
        <h1 className="text-xl font-semibold text-gray-900">Admin Panel</h1>
        <p className="mt-1 text-sm text-gray-500">
          Enter the admin API key to continue.
        </p>
        <form onSubmit={handleSubmit} className="mt-4 space-y-3">
          <input
            type="password"
            value={key}
            onChange={(e) => setKey(e.target.value)}
            placeholder="Admin API key"
            className="block w-full rounded-lg border border-gray-300 bg-white px-3.5 py-2.5 text-sm text-gray-900 placeholder-gray-400 focus:border-blue-500 focus:outline-none focus:ring-2 focus:ring-blue-500"
          />
          {error && <p className="text-sm text-red-600">{error}</p>}
          <button
            type="submit"
            disabled={loading || !key.trim()}
            className="w-full rounded-lg bg-blue-600 px-4 py-2.5 text-sm font-semibold text-white hover:bg-blue-700 disabled:opacity-50 transition-colors"
          >
            {loading ? "Verifying..." : "Sign In"}
          </button>
        </form>
      </div>
    </div>
  );
}

function CreateTenantForm({
  adminKey,
  onCreated,
}: {
  adminKey: string;
  onCreated: (res: RegisterResponse) => void;
}) {
  const [name, setName] = useState("");
  const [email, setEmail] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    const trimmedName = name.trim();
    if (!trimmedName) return;

    setLoading(true);
    setError(null);

    try {
      const res = await adminCreateTenant(adminKey, {
        name: trimmedName,
        email: email.trim() || undefined,
      });
      onCreated(res);
      setName("");
      setEmail("");
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : "Failed to create tenant");
    } finally {
      setLoading(false);
    }
  };

  return (
    <form onSubmit={handleSubmit} className="flex flex-wrap items-end gap-3">
      <div>
        <label className="block text-xs font-medium text-gray-600">Name</label>
        <input
          type="text"
          value={name}
          onChange={(e) => setName(e.target.value)}
          placeholder="Tenant name"
          required
          className="mt-1 rounded-lg border border-gray-300 bg-white px-3.5 py-2 text-sm text-gray-900 placeholder-gray-400 focus:border-blue-500 focus:outline-none focus:ring-2 focus:ring-blue-500"
        />
      </div>
      <div>
        <label className="block text-xs font-medium text-gray-600">
          Email (optional)
        </label>
        <input
          type="email"
          value={email}
          onChange={(e) => setEmail(e.target.value)}
          placeholder="email@example.com"
          className="mt-1 rounded-lg border border-gray-300 bg-white px-3.5 py-2 text-sm text-gray-900 placeholder-gray-400 focus:border-blue-500 focus:outline-none focus:ring-2 focus:ring-blue-500"
        />
      </div>
      <button
        type="submit"
        disabled={loading || !name.trim()}
        className="rounded-lg bg-blue-600 px-4 py-2 text-sm font-semibold text-white hover:bg-blue-700 disabled:opacity-50 transition-colors"
      >
        {loading ? "Creating..." : "Create Tenant"}
      </button>
      {error && <p className="w-full text-sm text-red-600">{error}</p>}
    </form>
  );
}

function TenantTable({
  tenants,
  onDelete,
  deleting,
}: {
  tenants: TenantOut[];
  onDelete: (id: string) => void;
  deleting: string | null;
}) {
  if (tenants.length === 0) {
    return (
      <p className="py-8 text-center text-sm text-gray-400">
        No tenants yet.
      </p>
    );
  }

  return (
    <div className="overflow-x-auto">
      <table className="min-w-full text-sm">
        <thead>
          <tr className="border-b text-left text-xs font-medium uppercase text-gray-500">
            <th className="px-4 py-3">Name</th>
            <th className="px-4 py-3">Email</th>
            <th className="px-4 py-3">Status</th>
            <th className="px-4 py-3">Created</th>
            <th className="px-4 py-3">Model</th>
            <th className="px-4 py-3" />
          </tr>
        </thead>
        <tbody>
          {tenants.map((t) => (
            <tr key={t.id} className="border-b last:border-0">
              <td className="px-4 py-3 font-medium text-gray-900">
                {t.name}
                <span className="ml-2 font-mono text-[10px] text-gray-400">
                  {t.id.slice(0, 8)}
                </span>
              </td>
              <td className="px-4 py-3 text-gray-600">{t.email || "--"}</td>
              <td className="px-4 py-3">
                <span
                  className={`inline-block rounded-full px-2 py-0.5 text-xs font-medium ${
                    t.status === "active"
                      ? "bg-green-100 text-green-700"
                      : t.status === "suspended"
                        ? "bg-yellow-100 text-yellow-700"
                        : "bg-red-100 text-red-700"
                  }`}
                >
                  {t.status}
                </span>
              </td>
              <td className="px-4 py-3 text-gray-500">
                {new Date(t.created_at).toLocaleDateString()}
              </td>
              <td className="px-4 py-3 text-gray-500">
                {t.embedding_model_version || "default"}
              </td>
              <td className="px-4 py-3 text-right">
                {t.status !== "deleted" && (
                  <button
                    onClick={() => onDelete(t.id)}
                    disabled={deleting === t.id}
                    className="text-xs text-red-500 hover:text-red-700 disabled:opacity-50"
                  >
                    {deleting === t.id ? "Deleting..." : "Delete"}
                  </button>
                )}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function readStoredAdminKey(): string | null {
  if (typeof window === "undefined") return null;
  return localStorage.getItem(ADMIN_KEY_STORAGE);
}

export default function AdminPage() {
  const [adminKey, setAdminKey] = useState<string | null>(readStoredAdminKey);
  const [tenants, setTenants] = useState<TenantOut[]>([]);
  const [loading, setLoading] = useState(true);
  const [deleting, setDeleting] = useState<string | null>(null);
  const [newApiKey, setNewApiKey] = useState<string | null>(null);
  const [copied, setCopied] = useState(false);
  const [verified, setVerified] = useState(false);
  const verifyAttempted = useRef(false);

  // Verify stored admin key on mount
  useEffect(() => {
    if (!adminKey || verifyAttempted.current) return;
    verifyAttempted.current = true;

    let cancelled = false;
    adminListTenants(adminKey)
      .then((res) => {
        if (!cancelled) {
          setTenants(res.tenants);
          setVerified(true);
          setLoading(false);
        }
      })
      .catch(() => {
        if (!cancelled) {
          localStorage.removeItem(ADMIN_KEY_STORAGE);
          setAdminKey(null);
          setLoading(false);
        }
      });
    return () => {
      cancelled = true;
    };
  }, [adminKey]);

  const fetchTenants = useCallback(async () => {
    if (!adminKey) return;
    setLoading(true);
    try {
      const res = await adminListTenants(adminKey);
      setTenants(res.tenants);
    } finally {
      setLoading(false);
    }
  }, [adminKey]);

  const handleDelete = async (tenantId: string) => {
    if (!adminKey) return;
    if (!window.confirm("Are you sure you want to delete this tenant?")) return;

    setDeleting(tenantId);
    try {
      await adminDeleteTenant(adminKey, tenantId);
      await fetchTenants();
    } finally {
      setDeleting(null);
    }
  };

  const handleCreated = (res: RegisterResponse) => {
    setNewApiKey(res.api_key);
    setCopied(false);
    fetchTenants();
  };

  const handleLogin = (key: string) => {
    localStorage.setItem(ADMIN_KEY_STORAGE, key);
    verifyAttempted.current = false;
    setVerified(true);
    setAdminKey(key);
  };

  const handleLogout = () => {
    localStorage.removeItem(ADMIN_KEY_STORAGE);
    verifyAttempted.current = false;
    setAdminKey(null);
    setVerified(false);
    setTenants([]);
  };

  if (adminKey && !verified) {
    return (
      <div className="flex min-h-screen items-center justify-center">
        <div className="animate-pulse text-gray-400">Loading...</div>
      </div>
    );
  }

  if (!adminKey) {
    return <AdminLogin onLogin={handleLogin} />;
  }

  return (
    <div className="min-h-screen bg-gray-50">
      <header className="border-b bg-white px-6 py-4">
        <div className="mx-auto flex max-w-5xl items-center justify-between">
          <div>
            <h1 className="text-lg font-semibold text-gray-900">
              Admin Panel
            </h1>
            <p className="text-xs text-gray-500">
              Manage tenants and platform settings
            </p>
          </div>
          <button
            onClick={handleLogout}
            className="text-sm text-gray-500 hover:text-gray-900"
          >
            Sign Out
          </button>
        </div>
      </header>

      <div className="mx-auto max-w-5xl px-6 py-8">
        {/* Create tenant */}
        <div className="rounded-lg border bg-white p-6">
          <h2 className="mb-4 text-sm font-medium text-gray-900">
            Create Tenant
          </h2>
          <CreateTenantForm adminKey={adminKey} onCreated={handleCreated} />

          {newApiKey && (
            <div className="mt-4 rounded-md border border-blue-200 bg-blue-50 p-4">
              <p className="text-sm font-medium text-blue-800">
                Tenant created. Copy the API key now — it will not be shown
                again.
              </p>
              <div className="mt-2 flex items-center gap-2">
                <code className="flex-1 rounded bg-white px-3 py-1.5 font-mono text-xs text-gray-700 border">
                  {newApiKey}
                </code>
                <button
                  onClick={() => {
                    navigator.clipboard.writeText(newApiKey);
                    setCopied(true);
                  }}
                  className="rounded border px-3 py-1.5 text-xs text-gray-600 hover:bg-gray-50"
                >
                  {copied ? "Copied" : "Copy"}
                </button>
                <button
                  onClick={() => setNewApiKey(null)}
                  className="rounded border px-3 py-1.5 text-xs text-gray-600 hover:bg-gray-50"
                >
                  Dismiss
                </button>
              </div>
            </div>
          )}
        </div>

        {/* Tenant list */}
        <div className="mt-6 rounded-lg border bg-white">
          <div className="flex items-center justify-between border-b px-6 py-4">
            <h2 className="text-sm font-medium text-gray-900">
              Tenants ({tenants.length})
            </h2>
            <button
              onClick={fetchTenants}
              disabled={loading}
              className="text-xs text-blue-600 hover:text-blue-800 disabled:opacity-50"
            >
              {loading ? "Loading..." : "Refresh"}
            </button>
          </div>
          <TenantTable
            tenants={tenants}
            onDelete={handleDelete}
            deleting={deleting}
          />
        </div>
      </div>
    </div>
  );
}
