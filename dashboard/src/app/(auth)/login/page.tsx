"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import Link from "next/link";
import { getMe, ApiError } from "@/lib/api";
import { useAuth } from "@/lib/auth-context";

export default function LoginPage() {
  const [key, setKey] = useState("");
  const [error, setError] = useState("");
  const [isSubmitting, setIsSubmitting] = useState(false);
  const { login } = useAuth();
  const router = useRouter();

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    setError("");
    setIsSubmitting(true);

    const trimmed = key.trim();
    localStorage.setItem("api_key", trimmed);

    try {
      await getMe();
      login(trimmed);
      router.push("/home");
    } catch (err) {
      localStorage.removeItem("api_key");
      if (err instanceof ApiError) {
        setError(err.status === 401 ? "Invalid API key" : err.detail);
      } else {
        setError("Could not connect to the server.");
      }
    } finally {
      setIsSubmitting(false);
    }
  }

  return (
    <div className="rounded-lg border bg-white p-8 shadow-sm">
      <h1 className="text-2xl font-semibold mb-1">Sign In</h1>
      <p className="text-gray-500 text-sm mb-6">
        Enter your API key to access your dashboard
      </p>

      <form onSubmit={handleSubmit} className="space-y-4">
        <div>
          <label htmlFor="api_key" className="block text-sm font-medium mb-1">
            API Key
          </label>
          <input
            id="api_key"
            type="password"
            required
            value={key}
            onChange={(e) => setKey(e.target.value)}
            className="w-full rounded-md border px-3 py-2 text-sm font-mono focus:outline-none focus:ring-2 focus:ring-gray-900"
            placeholder="rk_..."
          />
        </div>

        {error && (
          <p className="text-sm text-red-600">{error}</p>
        )}

        <button
          type="submit"
          disabled={isSubmitting}
          className="w-full rounded-md bg-gray-900 px-4 py-2 text-sm text-white hover:bg-gray-800 disabled:opacity-50"
        >
          {isSubmitting ? "Signing in..." : "Sign In"}
        </button>
      </form>

      <p className="mt-4 text-center text-sm text-gray-500">
        Don&apos;t have an account?{" "}
        <Link href="/register" className="text-gray-900 underline">
          Create one
        </Link>
      </p>
    </div>
  );
}
