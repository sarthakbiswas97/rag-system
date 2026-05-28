"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import Link from "next/link";
import { useAuth } from "@/lib/auth-context";

export default function Home() {
  const { apiKey, isLoading } = useAuth();
  const router = useRouter();
  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    setMounted(true);
  }, []);

  useEffect(() => {
    if (mounted && !isLoading && apiKey) {
      router.replace("/home");
    }
  }, [apiKey, isLoading, router, mounted]);

  // Always render the same thing on first render (server + client)
  // to avoid hydration mismatch from localStorage reads
  if (!mounted || isLoading) {
    return (
      <div className="flex min-h-screen items-center justify-center bg-gray-50">
        <div className="animate-pulse text-gray-400 text-sm">Loading...</div>
      </div>
    );
  }

  if (apiKey) return null;

  return (
    <div className="flex min-h-screen items-center justify-center bg-gray-50 px-4">
      <div
        className="w-full max-w-md text-center"
        style={{ animation: "fade-in 0.3s ease-out" }}
      >
        <div className="rounded-2xl border border-gray-200 bg-white p-10 shadow-md">
          <h1 className="text-3xl font-bold text-gray-900">RAG Platform</h1>
          <p className="mt-3 text-gray-500">
            Upload documents. Ask questions. Get answers.
          </p>

          <div className="mt-8 space-y-3">
            <Link
              href="/login"
              className="block w-full rounded-lg bg-blue-600 px-4 py-3 text-sm font-semibold text-white hover:bg-blue-700 transition-colors"
            >
              Sign In
            </Link>
            <Link
              href="/register"
              className="block w-full rounded-lg border border-gray-300 bg-white px-4 py-3 text-sm font-semibold text-gray-700 hover:bg-gray-50 transition-colors"
            >
              Create Account
            </Link>
          </div>
        </div>

        <p className="mt-6 text-xs text-gray-400">
          Multi-tenant RAG platform with document ingestion and AI-powered Q&A
        </p>
      </div>
    </div>
  );
}
