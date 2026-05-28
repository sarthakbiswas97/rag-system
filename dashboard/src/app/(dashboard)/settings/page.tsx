"use client";

import { useCallback, useState } from "react";
import { useAuth } from "@/lib/auth-context";
import { updateMe } from "@/lib/api";

export default function SettingsPage() {
  const { apiKey, tenant, logout } = useAuth();
  const [name, setName] = useState(tenant?.name ?? "");
  const [saving, setSaving] = useState(false);
  const [saved, setSaved] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [showKey, setShowKey] = useState(false);
  const [copied, setCopied] = useState(false);

  const handleSave = async (e: React.FormEvent) => {
    e.preventDefault();
    const trimmed = name.trim();
    if (!trimmed || trimmed === tenant?.name) return;

    setSaving(true);
    setError(null);
    setSaved(false);

    try {
      await updateMe({ name: trimmed });
      setSaved(true);
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : "Failed to update");
    } finally {
      setSaving(false);
    }
  };

  const copyKey = useCallback(() => {
    if (!apiKey) return;
    navigator.clipboard.writeText(apiKey);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  }, [apiKey]);

  if (!tenant || !apiKey) return null;

  const maskedKey = `${apiKey.slice(0, 7)}${"*".repeat(20)}${apiKey.slice(-4)}`;

  return (
    <div className="max-w-2xl">
      <h1 className="text-2xl font-semibold text-gray-900">Settings</h1>
      <p className="mt-1 text-sm text-gray-500">
        Manage your account settings and API key.
      </p>

      {/* Profile */}
      <form onSubmit={handleSave} className="mt-8 rounded-lg border bg-white p-6">
        <h2 className="text-lg font-medium text-gray-900">Profile</h2>
        <div className="mt-4 space-y-4">
          <div>
            <label htmlFor="name" className="block text-sm font-medium text-gray-700">
              Display Name
            </label>
            <input
              id="name"
              type="text"
              value={name}
              onChange={(e) => {
                setName(e.target.value);
                setSaved(false);
              }}
              className="mt-1 block w-full rounded-md border px-3 py-2 text-sm focus:border-blue-500 focus:outline-none focus:ring-1 focus:ring-blue-500"
            />
          </div>
          <div>
            <label className="block text-sm font-medium text-gray-700">Email</label>
            <p className="mt-1 text-sm text-gray-500">{tenant.email}</p>
          </div>
          <div className="flex items-center gap-3">
            <button
              type="submit"
              disabled={saving || !name.trim() || name.trim() === tenant.name}
              className="rounded-md bg-blue-600 px-4 py-2 text-sm font-medium text-white hover:bg-blue-700 disabled:opacity-50"
            >
              {saving ? "Saving..." : "Save Changes"}
            </button>
            {saved && <span className="text-sm text-green-600">Saved</span>}
            {error && <span className="text-sm text-red-600">{error}</span>}
          </div>
        </div>
      </form>

      {/* API Key */}
      <div className="mt-6 rounded-lg border bg-white p-6">
        <h2 className="text-lg font-medium text-gray-900">API Key</h2>
        <p className="mt-1 text-sm text-gray-500">
          Use this key to authenticate with the API directly.
        </p>
        <div className="mt-4 flex items-center gap-2">
          <code className="flex-1 rounded-md bg-gray-100 px-3 py-2 font-mono text-sm text-gray-700">
            {showKey ? apiKey : maskedKey}
          </code>
          <button
            onClick={() => setShowKey((prev) => !prev)}
            className="rounded-md border px-3 py-2 text-sm text-gray-600 hover:bg-gray-50"
          >
            {showKey ? "Hide" : "Show"}
          </button>
          <button
            onClick={copyKey}
            className="rounded-md border px-3 py-2 text-sm text-gray-600 hover:bg-gray-50"
          >
            {copied ? "Copied" : "Copy"}
          </button>
        </div>
      </div>

      {/* Danger zone */}
      <div className="mt-6 rounded-lg border border-red-200 bg-white p-6">
        <h2 className="text-lg font-medium text-red-700">Danger Zone</h2>
        <p className="mt-1 text-sm text-gray-500">
          Signing out will clear your API key from this browser.
        </p>
        <button
          onClick={logout}
          className="mt-4 rounded-md border border-red-300 px-4 py-2 text-sm font-medium text-red-600 hover:bg-red-50"
        >
          Sign Out
        </button>
      </div>
    </div>
  );
}
