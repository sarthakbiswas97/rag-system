"use client";

import { useEffect, useState } from "react";
import { useAuth } from "@/lib/auth-context";
import {
  getMeStats,
  getMeUsage,
  type TenantStats,
  type UsageSummary,
} from "@/lib/api";

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
    <div className="rounded-lg border bg-white p-6">
      <p className="text-sm font-medium text-gray-500">{label}</p>
      <p className="mt-2 text-3xl font-semibold text-gray-900">{value}</p>
      {subtitle && (
        <p className="mt-1 text-xs text-gray-400">{subtitle}</p>
      )}
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
    <div className="max-w-4xl">
      <h1 className="text-2xl font-semibold text-gray-900">Dashboard</h1>
      <p className="mt-1 text-sm text-gray-500">
        Welcome back, {tenant.name}
      </p>

      <div className="mt-8 grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
        <StatCard label="Status" value={tenant.status} />
        <StatCard
          label="Chunks Indexed"
          value={loading ? "..." : error ? "--" : stats?.chunk_count ?? 0}
          subtitle="In vector store"
        />
        <StatCard
          label="Queries (30d)"
          value={loading ? "..." : queryCount}
          subtitle="Total queries made"
        />
        <StatCard
          label="Ingestions (30d)"
          value={loading ? "..." : ingestCount}
          subtitle="Chunks ingested"
        />
      </div>

      {/* Recent activity */}
      {usage && usage.recent.length > 0 && (
        <div className="mt-8 rounded-lg border bg-white">
          <div className="border-b px-6 py-4">
            <h2 className="text-sm font-medium text-gray-900">
              Recent Activity
            </h2>
          </div>
          <div className="divide-y">
            {usage.recent.map((event, i) => (
              <div
                key={`${event.created_at}-${i}`}
                className="flex items-center justify-between px-6 py-3"
              >
                <div className="flex items-center gap-3">
                  <span
                    className={`inline-block rounded-full px-2 py-0.5 text-xs font-medium ${
                      event.event_type === "query"
                        ? "bg-blue-100 text-blue-700"
                        : "bg-green-100 text-green-700"
                    }`}
                  >
                    {event.event_type}
                  </span>
                  <span className="text-sm text-gray-600">
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
        </div>
      )}

      <div className="mt-8 rounded-lg border bg-white p-6">
        <h2 className="text-lg font-medium text-gray-900">Account Details</h2>
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
            <dd className="mt-1 font-mono text-xs text-gray-600">
              {tenant.id}
            </dd>
          </div>
          <div>
            <dt className="text-sm font-medium text-gray-500">Created</dt>
            <dd className="mt-1 text-sm text-gray-900">{createdDate}</dd>
          </div>
          <div>
            <dt className="text-sm font-medium text-gray-500">
              Embedding Model
            </dt>
            <dd className="mt-1 text-sm text-gray-900">
              {tenant.embedding_model_version || "Default"}
            </dd>
          </div>
        </dl>
      </div>

      {error && (
        <p className="mt-4 text-sm text-red-500">
          Could not load stats: {error}
        </p>
      )}
    </div>
  );
}
