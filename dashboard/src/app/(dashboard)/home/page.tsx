"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { useAuth } from "@/lib/auth-context";
import {
  getMeStats,
  getMeUsage,
  type TenantStats,
  type UsageSummary,
} from "@/lib/api";
import { SkeletonCard, SkeletonRow } from "@/components/skeleton";

function StatCard({
  label,
  value,
  subtitle,
}: {
  label: string;
  value: string | number;
  subtitle?: string;
}) {
  return (
    <div className="rounded-xl border border-gray-200 bg-white p-6 shadow-sm">
      <p className="text-sm font-medium text-gray-500">{label}</p>
      <p className="mt-2 text-3xl font-bold text-gray-900">{value}</p>
      {subtitle && (
        <p className="mt-1 text-xs text-gray-400">{subtitle}</p>
      )}
    </div>
  );
}

function OnboardingChecklist({
  hasDocuments,
  hasQueries,
}: {
  hasDocuments: boolean;
  hasQueries: boolean;
}) {
  const steps = [
    { label: "Create your account", done: true, href: "" },
    { label: "Upload your first document", done: hasDocuments, href: "/documents" },
    { label: "Ask your first question", done: hasQueries, href: "/chat" },
  ];

  const completed = steps.filter((s) => s.done).length;
  if (completed === steps.length) return null;

  return (
    <div
      className="mt-8 rounded-xl border border-blue-200 bg-blue-50 p-6"
      style={{ animation: "fade-in 0.3s ease-out" }}
    >
      <h2 className="text-sm font-bold text-blue-900">
        Getting Started ({completed}/{steps.length})
      </h2>
      <div className="mt-3 space-y-2.5">
        {steps.map((step) => (
          <div key={step.label} className="flex items-center gap-3">
            <div
              className={`flex h-5 w-5 shrink-0 items-center justify-center rounded-full text-xs ${
                step.done
                  ? "bg-blue-600 text-white"
                  : "border-2 border-blue-300"
              }`}
            >
              {step.done && (
                <svg className="h-3 w-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={3} d="M5 13l4 4L19 7" />
                </svg>
              )}
            </div>
            {step.done ? (
              <span className="text-sm text-blue-700 line-through opacity-60">
                {step.label}
              </span>
            ) : (
              <Link href={step.href} className="text-sm font-medium text-blue-700 hover:underline">
                {step.label}
              </Link>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}

export default function HomePage() {
  const { tenant } = useAuth();
  const [stats, setStats] = useState<TenantStats | null>(null);
  const [usage, setUsage] = useState<UsageSummary | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let cancelled = false;
    Promise.all([getMeStats(), getMeUsage(30)])
      .then(([statsData, usageData]) => {
        if (!cancelled) {
          setStats(statsData);
          setUsage(usageData);
        }
      })
      .catch((err) => {
        if (!cancelled) setError(err.detail || "Failed to load stats");
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, []);

  if (!tenant) return null;

  const createdDate = new Date(tenant.created_at).toLocaleDateString("en-US", {
    year: "numeric",
    month: "long",
    day: "numeric",
  });

  const queryCount = usage?.totals?.query ?? 0;
  const ingestCount = usage?.totals?.ingest ?? 0;

  return (
    <div className="max-w-4xl" style={{ animation: "fade-in 0.3s ease-out" }}>
      <h1 className="text-2xl font-bold text-gray-900">Dashboard</h1>
      <p className="mt-1 text-sm text-gray-500">
        Welcome back, {tenant.name}
      </p>

      {/* Onboarding checklist for new users */}
      {!loading && !error && (
        <OnboardingChecklist
          hasDocuments={ingestCount > 0 || (stats?.chunk_count ?? 0) > 0}
          hasQueries={queryCount > 0}
        />
      )}

      {/* Stats grid */}
      <div className="mt-8 grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
        {loading ? (
          <>
            <SkeletonCard />
            <SkeletonCard />
            <SkeletonCard />
            <SkeletonCard />
          </>
        ) : (
          <>
            <StatCard label="Status" value={tenant.status} />
            <StatCard
              label="Chunks Indexed"
              value={error ? "--" : stats?.chunk_count ?? 0}
              subtitle="In vector store"
            />
            <StatCard
              label="Queries (30d)"
              value={queryCount}
              subtitle="Total queries made"
            />
            <StatCard
              label="Ingestions (30d)"
              value={ingestCount}
              subtitle="Chunks ingested"
            />
          </>
        )}
      </div>

      {/* Recent activity */}
      <div className="mt-8 rounded-xl border border-gray-200 bg-white shadow-sm">
        <div className="border-b border-gray-200 px-6 py-4">
          <h2 className="text-sm font-semibold text-gray-900">
            Recent Activity
          </h2>
        </div>
        {loading ? (
          <div className="divide-y divide-gray-100">
            <SkeletonRow />
            <SkeletonRow />
            <SkeletonRow />
          </div>
        ) : usage && usage.recent.length > 0 ? (
          <div className="divide-y divide-gray-100">
            {usage.recent.map((event, i) => (
              <div
                key={`${event.created_at}-${i}`}
                className="flex items-center justify-between px-6 py-3"
              >
                <div className="flex items-center gap-3">
                  <span
                    className={`inline-block rounded-full px-2.5 py-0.5 text-xs font-semibold ${
                      event.event_type === "query"
                        ? "bg-blue-100 text-blue-700"
                        : "bg-green-100 text-green-700"
                    }`}
                  >
                    {event.event_type}
                  </span>
                  <span className="text-sm text-gray-700">
                    {event.event_type === "query"
                      ? "1 query"
                      : `${event.value} chunks`}
                  </span>
                </div>
                <div className="flex items-center gap-4 text-xs text-gray-400">
                  {event.elapsed_ms !== null && (
                    <span>{(event.elapsed_ms / 1000).toFixed(1)}s</span>
                  )}
                  <span>
                    {new Date(event.created_at).toLocaleString()}
                  </span>
                </div>
              </div>
            ))}
          </div>
        ) : (
          <div className="px-6 py-8 text-center text-sm text-gray-400">
            No activity yet. Upload a document to get started.
          </div>
        )}
      </div>

      {/* Account details */}
      <div className="mt-8 rounded-xl border border-gray-200 bg-white p-6 shadow-sm">
        <h2 className="text-lg font-semibold text-gray-900">Account Details</h2>
        <dl className="mt-4 grid gap-4 sm:grid-cols-2">
          <div>
            <dt className="text-sm font-medium text-gray-500">Name</dt>
            <dd className="mt-1 text-sm text-gray-900">{tenant.name}</dd>
          </div>
          <div>
            <dt className="text-sm font-medium text-gray-500">Email</dt>
            <dd className="mt-1 text-sm text-gray-900">{tenant.email}</dd>
          </div>
          <div>
            <dt className="text-sm font-medium text-gray-500">Tenant ID</dt>
            <dd className="mt-1 font-mono text-xs text-gray-600">{tenant.id}</dd>
          </div>
          <div>
            <dt className="text-sm font-medium text-gray-500">Created</dt>
            <dd className="mt-1 text-sm text-gray-900">{createdDate}</dd>
          </div>
          <div>
            <dt className="text-sm font-medium text-gray-500">Embedding Model</dt>
            <dd className="mt-1 text-sm text-gray-900">
              {tenant.embedding_model_version || "Default"}
            </dd>
          </div>
        </dl>
      </div>

      {error && (
        <div className="mt-4 rounded-lg border border-red-200 bg-red-50 px-4 py-3">
          <p className="text-sm font-medium text-red-700">
            Could not load stats: {error}
          </p>
        </div>
      )}
    </div>
  );
}
