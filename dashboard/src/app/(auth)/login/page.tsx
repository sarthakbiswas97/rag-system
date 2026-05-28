"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import Link from "next/link";
import { getMe, ApiError } from "@/lib/api";
import { useAuth } from "@/lib/auth-context";
import { useToast } from "@/components/toast";

export default function LoginPage() {
  const [key, setKey] = useState("");
  const [isSubmitting, setIsSubmitting] = useState(false);
  const { login } = useAuth();
  const { toast } = useToast();
  const router = useRouter();

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    setIsSubmitting(true);

    const trimmed = key.trim();
    localStorage.setItem("api_key", trimmed);

    try {
      await getMe();
      login(trimmed);
      toast.success("Signed in successfully");
      router.push("/home");
    } catch (err) {
      localStorage.removeItem("api_key");
      if (err instanceof ApiError) {
        toast.error(err.status === 401 ? "Invalid API key" : err.detail);
      } else {
        toast.error("Could not connect to the server");
      }
    } finally {
      setIsSubmitting(false);
    }
  }

  return (
    <div
      className="rounded-xl border border-gray-200 bg-white p-8 shadow-md"
      style={{ animation: "fade-in 0.3s ease-out" }}
    >
      <h1 className="text-2xl font-bold text-gray-900 mb-1">Sign In</h1>
      <p className="text-gray-500 text-sm mb-6">
        Enter your API key to access your dashboard
      </p>

      <form onSubmit={handleSubmit} className="space-y-4">
        <div>
          <label
            htmlFor="api_key"
            className="block text-sm font-medium text-gray-700 mb-1.5"
          >
            API Key
          </label>
          <input
            id="api_key"
            type="password"
            required
            value={key}
            onChange={(e) => setKey(e.target.value)}
            className="w-full rounded-lg border border-gray-300 bg-white px-3.5 py-2.5 text-sm text-gray-900 font-mono placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
            placeholder="rk_..."
          />
          <p className="mt-1.5 text-xs text-gray-400">
            Your key starts with <code className="text-gray-500">rk_</code> and was shown when you created your account.
          </p>
        </div>

        <details className="text-sm">
          <summary className="cursor-pointer text-gray-500 hover:text-gray-700 font-medium">
            Where do I find my API key?
          </summary>
          <div className="mt-2 rounded-lg bg-gray-50 border border-gray-200 p-3 text-xs text-gray-600 space-y-1.5">
            <p>Your API key was displayed once when you registered. Check your password manager or notes app.</p>
            <p>If you have lost your key, you will need to <Link href="/register" className="text-blue-600 hover:underline">create a new account</Link>.</p>
          </div>
        </details>

        <button
          type="submit"
          disabled={isSubmitting}
          className="w-full rounded-lg bg-blue-600 px-4 py-2.5 text-sm font-semibold text-white hover:bg-blue-700 disabled:opacity-50 transition-colors"
        >
          {isSubmitting ? "Signing in..." : "Sign In"}
        </button>
      </form>

      <p className="mt-5 text-center text-sm text-gray-500">
        Don&apos;t have an account?{" "}
        <Link href="/register" className="text-blue-600 font-medium hover:text-blue-700">
          Create one
        </Link>
      </p>
    </div>
  );
}
