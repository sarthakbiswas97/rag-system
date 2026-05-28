"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import Link from "next/link";
import { register, ApiError } from "@/lib/api";
import { useAuth } from "@/lib/auth-context";
import { useToast } from "@/components/toast";

export default function RegisterPage() {
  const [name, setName] = useState("");
  const [email, setEmail] = useState("");
  const [apiKey, setApiKey] = useState("");
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [hasCopied, setHasCopied] = useState(false);
  const { login } = useAuth();
  const { toast } = useToast();
  const router = useRouter();

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    setIsSubmitting(true);

    try {
      const res = await register({ name: name.trim(), email: email.trim() });
      setApiKey(res.api_key);
      toast.success("Account created successfully");
    } catch (err) {
      if (err instanceof ApiError) {
        toast.error(err.detail);
      } else {
        toast.error("Something went wrong. Please try again.");
      }
    } finally {
      setIsSubmitting(false);
    }
  }

  function handleCopy() {
    navigator.clipboard.writeText(apiKey);
    setHasCopied(true);
    toast.success("API key copied to clipboard");
  }

  function handleContinue() {
    login(apiKey);
    router.push("/home");
  }

  if (apiKey) {
    return (
      <div
        className="rounded-xl border border-gray-200 bg-white p-8 shadow-md"
        style={{ animation: "fade-in 0.3s ease-out" }}
      >
        <h1 className="text-2xl font-bold text-gray-900 mb-2">
          Account Created
        </h1>

        {/* Warning banner */}
        <div className="rounded-lg bg-amber-50 border border-amber-200 p-4 mb-5">
          <div className="flex gap-3">
            <svg className="h-5 w-5 shrink-0 text-amber-500 mt-0.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01M12 3l9.5 16.5H2.5L12 3z" />
            </svg>
            <div>
              <p className="text-sm font-semibold text-amber-800">
                Save your API key now
              </p>
              <p className="text-xs text-amber-700 mt-1">
                This is your only way to sign in. It will never be shown again.
                Copy it and store it somewhere safe (password manager, notes app, etc.)
              </p>
            </div>
          </div>
        </div>

        {/* API key display */}
        <div className="rounded-lg bg-gray-50 border border-gray-200 p-4 font-mono text-sm text-gray-900 break-all">
          {apiKey}
        </div>

        {/* Actions */}
        <button
          onClick={handleCopy}
          className={`w-full mt-4 rounded-lg px-4 py-2.5 text-sm font-semibold transition-colors ${
            hasCopied
              ? "border border-green-300 bg-green-50 text-green-700"
              : "border border-gray-300 bg-white text-gray-700 hover:bg-gray-50"
          }`}
        >
          {hasCopied ? "Copied" : "Copy to Clipboard"}
        </button>

        <button
          onClick={handleContinue}
          disabled={!hasCopied}
          className="w-full mt-3 rounded-lg bg-blue-600 px-4 py-2.5 text-sm font-semibold text-white hover:bg-blue-700 disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
        >
          Continue to Dashboard
        </button>

        {!hasCopied && (
          <p className="mt-2 text-center text-xs text-gray-400">
            Please copy your API key first
          </p>
        )}
      </div>
    );
  }

  return (
    <div
      className="rounded-xl border border-gray-200 bg-white p-8 shadow-md"
      style={{ animation: "fade-in 0.3s ease-out" }}
    >
      <h1 className="text-2xl font-bold text-gray-900 mb-1">Create Account</h1>
      <p className="text-gray-500 text-sm mb-6">
        Get started with your RAG platform
      </p>

      <form onSubmit={handleSubmit} className="space-y-4">
        <div>
          <label htmlFor="name" className="block text-sm font-medium text-gray-700 mb-1.5">
            Organization Name
          </label>
          <input
            id="name"
            type="text"
            required
            value={name}
            onChange={(e) => setName(e.target.value)}
            className="w-full rounded-lg border border-gray-300 bg-white px-3.5 py-2.5 text-sm text-gray-900 placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
            placeholder="Acme Corp"
          />
        </div>

        <div>
          <label htmlFor="email" className="block text-sm font-medium text-gray-700 mb-1.5">
            Email
          </label>
          <input
            id="email"
            type="email"
            required
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            className="w-full rounded-lg border border-gray-300 bg-white px-3.5 py-2.5 text-sm text-gray-900 placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
            placeholder="you@example.com"
          />
        </div>

        <button
          type="submit"
          disabled={isSubmitting}
          className="w-full rounded-lg bg-blue-600 px-4 py-2.5 text-sm font-semibold text-white hover:bg-blue-700 disabled:opacity-50 transition-colors"
        >
          {isSubmitting ? "Creating..." : "Create Account"}
        </button>
      </form>

      <p className="mt-5 text-center text-sm text-gray-500">
        Already have an API key?{" "}
        <Link href="/login" className="text-blue-600 font-medium hover:text-blue-700">
          Sign in
        </Link>
      </p>
    </div>
  );
}
