export type DocumentStatus = "pending" | "ingesting" | "ready" | "failed";

export interface DocumentMeta {
  doc_id: string;
  filename: string;
  file_type: string;
  file_size_bytes: number;
  page_count: number | null;
  upload_date: string;
  status: DocumentStatus;
  error: string | null;
  chunk_count: number | null;
}

export interface DocumentListResponse {
  documents: DocumentMeta[];
  total: number;
}

export interface SourceChunk {
  doc_id: string;
  filename: string;
  page: number | null;
  chunk_index: number;
  content: string;
  relevance_score: number;
}

export interface QueryRequest {
  question: string;
  doc_ids?: string[];
  top_k?: number;
}

export interface QueryResponse {
  question: string;
  answer: string;
  sources: SourceChunk[];
  has_relevant_context: boolean;
  tokens_used: number | null;
}

export interface IngestResponse {
  doc_id: string;
  filename: string;
  status: DocumentStatus;
  message: string;
}

export interface URLIngestRequest {
  url: string;
  display_name?: string;
}

export interface HealthResponse {
  status: string;
  version: string;
  environment: string;
  vector_store_chunk_count: number;
}

export interface ChatMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
  sources?: SourceChunk[];
  isStreaming?: boolean;
  timestamp: Date;
}
