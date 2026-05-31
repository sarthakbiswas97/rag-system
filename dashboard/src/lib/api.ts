const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export class ApiError extends Error {
  constructor(
    public status: number,
    public detail: string,
  ) {
    super(detail);
    this.name = "ApiError";
  }
}

async function request<T>(
  path: string,
  options: RequestInit = {},
): Promise<T> {
  const apiKey =
    typeof window !== "undefined" ? sessionStorage.getItem("api_key") : null;

  const headers: Record<string, string> = {
    ...(options.headers as Record<string, string>),
  };

  if (apiKey) {
    headers["X-API-Key"] = apiKey;
  }

  if (
    options.body &&
    typeof options.body === "string" &&
    !headers["Content-Type"]
  ) {
    headers["Content-Type"] = "application/json";
  }

  const res = await fetch(`${API_BASE}${path}`, {
    ...options,
    headers,
  });

  if (!res.ok) {
    const body = await res.json().catch(() => ({ detail: res.statusText }));
    throw new ApiError(res.status, body.detail || "Request failed");
  }

  return res.json();
}

// Auth
export interface RegisterPayload {
  name: string;
  email: string;
}

export interface TenantOut {
  id: string;
  name: string;
  email: string;
  status: string;
  embedding_model_version: string | null;
  created_at: string;
  updated_at: string;
}

export interface RegisterResponse {
  tenant: TenantOut;
  api_key: string;
}

export function register(payload: RegisterPayload): Promise<RegisterResponse> {
  return request("/v1/register", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export function getMe(): Promise<TenantOut> {
  return request("/v1/me");
}

export interface TenantStats {
  tenant_id: string;
  chunk_count: number;
}

export function getMeStats(): Promise<TenantStats> {
  return request("/v1/me/stats");
}

export interface UsageEvent {
  event_type: string;
  value: number;
  elapsed_ms: number | null;
  created_at: string;
}

export interface UsageSummary {
  tenant_id: string;
  period: string;
  totals: Record<string, number>;
  recent: UsageEvent[];
}

export function getMeUsage(days: number = 30): Promise<UsageSummary> {
  return request(`/v1/me/usage?days=${days}`);
}

export function updateMe(payload: { name: string }): Promise<TenantOut> {
  return request("/v1/me", {
    method: "PUT",
    body: JSON.stringify(payload),
  });
}

// Ingest
export interface IngestResponse {
  documents_processed: number;
  documents_skipped: number;
  documents_failed: number;
  chunks_created: number;
  elapsed_ms: number;
}

export function ingestFiles(files: File[]): Promise<IngestResponse> {
  const form = new FormData();
  for (const file of files) {
    form.append("files", file);
  }
  return request("/v1/ingest", {
    method: "POST",
    body: form,
  });
}

// Query
export interface Citation {
  index: number;
  chunk_id: string;
  source: string;
  snippet: string;
}

export interface QueryResponse {
  answer: string;
  citations: Citation[];
  is_abstention: boolean;
  confidence: number;
  timing: {
    retrieval_ms: number;
    generation_ms: number;
    total_ms: number;
  };
}

export function query(
  question: string,
  top_k: number = 5,
): Promise<QueryResponse> {
  return request("/v1/query", {
    method: "POST",
    body: JSON.stringify({ question, top_k }),
  });
}

// Admin
// Admin endpoints use a separate admin API key, passed explicitly.

function adminRequest<T>(
  path: string,
  adminKey: string,
  options: RequestInit = {},
): Promise<T> {
  const headers: Record<string, string> = {
    "X-API-Key": adminKey,
    ...(options.headers as Record<string, string>),
  };

  if (
    options.body &&
    typeof options.body === "string" &&
    !headers["Content-Type"]
  ) {
    headers["Content-Type"] = "application/json";
  }

  return fetch(`${API_BASE}${path}`, { ...options, headers }).then(
    async (res) => {
      if (!res.ok) {
        const body = await res.json().catch(() => ({ detail: res.statusText }));
        throw new ApiError(res.status, body.detail || "Request failed");
      }
      return res.json();
    },
  );
}

export interface TenantListResponse {
  tenants: TenantOut[];
  count: number;
}

export interface CreateTenantPayload {
  name: string;
  email?: string;
}

export function adminListTenants(
  adminKey: string,
): Promise<TenantListResponse> {
  return adminRequest("/admin/tenants", adminKey);
}

export function adminCreateTenant(
  adminKey: string,
  payload: CreateTenantPayload,
): Promise<RegisterResponse> {
  return adminRequest("/admin/tenants", adminKey, {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export function adminDeleteTenant(
  adminKey: string,
  tenantId: string,
): Promise<TenantOut> {
  return adminRequest(`/admin/tenants/${tenantId}`, adminKey, {
    method: "DELETE",
  });
}

// Health
export interface HealthResponse {
  status: string;
  qdrant: string;
  database: string;
}

export function getHealth(): Promise<HealthResponse> {
  return request("/v1/health");
}
