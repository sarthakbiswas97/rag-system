"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import Link from "next/link";
import { register, ApiError } from "@/lib/api";
import { useAuth } from "@/lib/auth-context";

export default function RegisterPage() {
  const [name, setName] = useState("");
  const [email, setEmail] = useState("");
  const [error, setError] = useState("");
  const [apiKey, setApiKey] = useState("");
  const [isSubmitting, setIsSubmitting] = useState(false);
  const { login } = useAuth();
  const router = useRouter();

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    setError("");
    setIsSubmitting(true);

    try {
      const res = await register({ name: name.trim(), email: email.trim() });
      setApiKey(res.api_key);
    } catch (err) {
      if (err instanceof ApiError) {
        setError(err.detail);
      } else {
        setError("Something went wrong. Please try again.");
      }
    } finally {
      setIsSubmitting(false);
    }
  }

  function handleContinue() {
    login(apiKey);
    router.push("/chat");
  }

  if (apiKey) {
    return (
      <div className="rounded-lg border bg-white p-8 shadow-sm">
        <h1 className="text-2xl font-semibold mb-2">Account Created</h1>
        <p className="text-gray-600 mb-4">
          Save your API key now. It will only be shown once.
        </p>
        <div className="bg-gray-100 rounded-md p-4 font-mono text-sm break-all mb-6">
          {apiKey}
        </div>
        <button
          onClick={() => navigator.clipboard.writeText(apiKey)}
          className="w-full mb-3 rounded-md border border-gray-300 px-4 py-2 text-sm hover:bg-gray-50"
        >
          Copy to Clipboard
        </button>
        <button
          onClick={handleContinue}
          className="w-full rounded-md bg-gray-900 px-4 py-2 text-sm text-white hover:bg-gray-800"
        >
          Continue to Dashboard
        </button>
      </div>
    );
  }

  return (
    <div className="rounded-lg border bg-white p-8 shadow-sm">
      <h1 className="text-2xl font-semibold mb-1">Create Account</h1>
      <p className="text-gray-500 text-sm mb-6">
        Get started with your RAG platform
      </p>

      <form onSubmit={handleSubmit} className="space-y-4">
        <div>
          <label htmlFor="name" className="block text-sm font-medium mb-1">
            Organization Name
          </label>
          <input
            id="name"
            type="text"
            required
            value={name}
            onChange={(e) => setName(e.target.value)}
            className="w-full rounded-md border px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-gray-900"
            placeholder="Acme Corp"
          />
        </div>

        <div>
          <label htmlFor="email" className="block text-sm font-medium mb-1">
            Email
          </label>
          <input
            id="email"
            type="email"
            required
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            className="w-full rounded-md border px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-gray-900"
            placeholder="you@example.com"
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
          {isSubmitting ? "Creating..." : "Create Account"}
        </button>
      </form>

      <p className="mt-4 text-center text-sm text-gray-500">
        Already have an API key?{" "}
        <Link href="/login" className="text-gray-900 underline">
          Sign in
        </Link>
      </p>
    </div>
  );
}
