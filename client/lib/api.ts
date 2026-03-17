import type {
  DocumentListResponse,
  IngestResponse,
  QueryRequest,
  QueryResponse,
  URLIngestRequest,
  HealthResponse,
} from "./types";

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

async function apiFetch<T>(
  path: string,
  options?: RequestInit
): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    ...options,
    headers: {
      "Content-Type": "application/json",
      ...options?.headers,
    },
  });

  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.detail || `API error: ${res.status}`);
  }

  return res.json();
}

export async function getDocuments(): Promise<DocumentListResponse> {
  return apiFetch<DocumentListResponse>("/documents");
}

export async function deleteDocument(docId: string) {
  return apiFetch(`/documents/${docId}`, { method: "DELETE" });
}

export async function deleteAllDocuments() {
  return apiFetch("/documents/delete-all", { method: "DELETE" });
}

export async function uploadDocument(file: File): Promise<IngestResponse> {
  const formData = new FormData();
  formData.append("file", file);

  const res = await fetch(`${API_BASE}/ingest`, {
    method: "POST",
    body: formData,
  });

  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.detail || `Upload failed: ${res.status}`);
  }

  return res.json();
}

export async function ingestURL(
  request: URLIngestRequest
): Promise<IngestResponse> {
  return apiFetch<IngestResponse>("/ingest/url", {
    method: "POST",
    body: JSON.stringify(request),
  });
}

export async function queryDocuments(
  request: QueryRequest
): Promise<QueryResponse> {
  return apiFetch<QueryResponse>("/query", {
    method: "POST",
    body: JSON.stringify(request),
  });
}

export async function streamQuery(
  request: QueryRequest,
  onToken: (token: string) => void,
  onDone: () => void,
  onError: (error: string) => void
): Promise<void> {
  const res = await fetch(`${API_BASE}/query/stream`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(request),
  });

  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    onError(body.detail || `Stream failed: ${res.status}`);
    return;
  }

  const reader = res.body?.getReader();
  if (!reader) {
    onError("No response body");
    return;
  }

  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split("\n");
    buffer = lines.pop() || "";

    for (const line of lines) {
      if (!line.startsWith("data: ")) continue;
      const data = line.slice(6);

      if (data === "[DONE]") {
        onDone();
        return;
      }

      const unescaped = data.replace(/\\n/g, "\n");
      onToken(unescaped);
    }
  }

  onDone();
}

export async function checkHealth(): Promise<HealthResponse> {
  return apiFetch<HealthResponse>("/health");
}
