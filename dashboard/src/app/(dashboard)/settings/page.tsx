"use client";

import { useCallback, useState } from "react";
import { useAuth } from "@/lib/auth-context";
import { updateMe } from "@/lib/api";
import { useToast } from "@/components/toast";

export default function SettingsPage() {
  const { apiKey, tenant, logout } = useAuth();
  const [name, setName] = useState(tenant?.name ?? "");
  const [saving, setSaving] = useState(false);
  const { toast } = useToast();
  const [showKey, setShowKey] = useState(false);
  const [copied, setCopied] = useState(false);

  const handleSave = async (e: React.FormEvent) => {
    e.preventDefault();
    const trimmed = name.trim();
    if (!trimmed || trimmed === tenant?.name) return;

    setSaving(true);

    try {
      await updateMe({ name: trimmed });
      toast.success("Settings saved");
    } catch (err: unknown) {
      toast.error(err instanceof Error ? err.message : "Failed to update");
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
      <h1 className="text-2xl font-bold text-gray-900">Settings</h1>
      <p className="mt-1 text-sm text-gray-500">
        Manage your account settings and API key.
      </p>

      {/* Profile */}
      <form onSubmit={handleSave} className="mt-8 rounded-xl border border-gray-200 bg-white p-6 shadow-sm">
        <h2 className="text-lg font-semibold text-gray-900">Profile</h2>
        <div className="mt-4 space-y-4">
          <div>
            <label htmlFor="name" className="block text-sm font-medium text-gray-700 mb-1.5">
              Display Name
            </label>
            <input
              id="name"
              type="text"
              value={name}
              onChange={(e) => setName(e.target.value)}
              className="block w-full rounded-lg border border-gray-300 bg-white px-3.5 py-2.5 text-sm text-gray-900 placeholder-gray-400 focus:border-blue-500 focus:outline-none focus:ring-2 focus:ring-blue-500"
            />
          </div>
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1.5">Email</label>
            <p className="text-sm text-gray-900">{tenant.email}</p>
          </div>
          <div className="flex items-center gap-3">
            <button
              type="submit"
              disabled={saving || !name.trim() || name.trim() === tenant.name}
              className="rounded-lg bg-blue-600 px-4 py-2.5 text-sm font-semibold text-white hover:bg-blue-700 disabled:opacity-50 transition-colors"
            >
              {saving ? "Saving..." : "Save Changes"}
            </button>
          </div>
        </div>
      </form>

      {/* API Key */}
      <div className="mt-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm">
        <h2 className="text-lg font-semibold text-gray-900">API Key</h2>
        <p className="mt-1 text-sm text-gray-500">
          Use this key to authenticate with the API directly.
        </p>
        <div className="mt-4 flex items-center gap-2">
          <code className="flex-1 rounded-lg bg-gray-50 border border-gray-200 px-3.5 py-2.5 font-mono text-sm text-gray-900">
            {showKey ? apiKey : maskedKey}
          </code>
          <button
            onClick={() => setShowKey((prev) => !prev)}
            className="rounded-lg border border-gray-300 bg-white px-3.5 py-2.5 text-sm font-medium text-gray-700 hover:bg-gray-50 transition-colors"
          >
            {showKey ? "Hide" : "Show"}
          </button>
          <button
            onClick={copyKey}
            className="rounded-lg border border-gray-300 bg-white px-3.5 py-2.5 text-sm font-medium text-gray-700 hover:bg-gray-50 transition-colors"
          >
            {copied ? "Copied" : "Copy"}
          </button>
        </div>
      </div>

      {/* Danger zone */}
      <div className="mt-6 rounded-xl border border-red-200 bg-white p-6 shadow-sm">
        <h2 className="text-lg font-semibold text-red-700">Danger Zone</h2>
        <p className="mt-1 text-sm text-gray-500">
          Signing out will clear your API key from this browser.
        </p>
        <button
          onClick={logout}
          className="mt-4 rounded-lg border border-red-300 bg-white px-4 py-2.5 text-sm font-semibold text-red-600 hover:bg-red-50 transition-colors"
        >
          Sign Out
        </button>
      </div>
    </div>
  );
}
