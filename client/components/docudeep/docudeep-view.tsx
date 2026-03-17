"use client";

import { useState, useEffect, useCallback } from "react";
import {
  FileSearch,
  Search,
  Loader2,
  FileText,
  CheckSquare,
  Square,
  AlertCircle,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Badge } from "@/components/ui/badge";
import { SourceCard } from "./source-card";
import { getDocuments, queryDocuments } from "@/lib/api";
import type { DocumentMeta, QueryResponse } from "@/lib/types";
import { cn } from "@/lib/utils";

interface DocuDeepViewProps {
  refreshTrigger?: number;
}

export function DocuDeepView({ refreshTrigger }: DocuDeepViewProps) {
  const [documents, setDocuments] = useState<DocumentMeta[]>([]);
  const [selectedDocIds, setSelectedDocIds] = useState<string[]>([]);
  const [question, setQuestion] = useState("");
  const [loading, setLoading] = useState(false);
  const [docsLoading, setDocsLoading] = useState(true);
  const [result, setResult] = useState<QueryResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  const fetchDocs = useCallback(async () => {
    setDocsLoading(true);
    try {
      const res = await getDocuments();
      setDocuments(res.documents);
    } catch {
      // silently fail
    } finally {
      setDocsLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchDocs();
  }, [fetchDocs, refreshTrigger]);

  const toggleDoc = (docId: string) => {
    setSelectedDocIds((prev) =>
      prev.includes(docId)
        ? prev.filter((id) => id !== docId)
        : [...prev, docId]
    );
  };

  const handleQuery = async () => {
    if (!question.trim()) return;
    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const res = await queryDocuments({
        question: question.trim(),
        doc_ids: selectedDocIds.length > 0 ? selectedDocIds : undefined,
        top_k: 5,
      });
      setResult(res);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Query failed");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="flex flex-1 overflow-hidden">
      {/* Document Selector - Left */}
      <div className="flex w-[260px] flex-col border-r border-border bg-sidebar">
        <div className="flex items-center gap-2 border-b border-border px-3 py-3">
          <FileText className="size-4 text-muted-foreground" />
          <h3 className="text-sm font-medium text-foreground">Select Documents</h3>
        </div>

        <ScrollArea className="flex-1">
          {docsLoading ? (
            <div className="flex items-center justify-center py-12">
              <Loader2 className="size-5 animate-spin text-muted-foreground" />
            </div>
          ) : documents.length === 0 ? (
            <div className="px-4 py-12 text-center">
              <FileText className="mx-auto mb-2 size-8 text-muted-foreground/40" />
              <p className="text-sm text-muted-foreground">No documents</p>
            </div>
          ) : (
            <div className="space-y-0.5 p-1.5">
              {documents.map((doc) => {
                const isSelected = selectedDocIds.includes(doc.doc_id);
                return (
                  <div
                    key={doc.doc_id}
                    onClick={() => toggleDoc(doc.doc_id)}
                    className={cn(
                      "flex cursor-pointer items-center gap-2.5 rounded-lg px-2.5 py-2 transition-colors",
                      isSelected
                        ? "bg-accent/80"
                        : "hover:bg-accent/40"
                    )}
                  >
                    {isSelected ? (
                      <CheckSquare className="size-3.5 shrink-0 text-foreground" />
                    ) : (
                      <Square className="size-3.5 shrink-0 text-muted-foreground" />
                    )}
                    <div className="min-w-0 flex-1">
                      <p className="truncate text-sm text-foreground">
                        {doc.filename}
                      </p>
                      {doc.chunk_count && (
                        <p className="text-xs text-muted-foreground">
                          {doc.chunk_count} chunks
                        </p>
                      )}
                    </div>
                  </div>
                );
              })}
            </div>
          )}
        </ScrollArea>

        <div className="border-t border-border px-3 py-2">
          <p className="text-xs text-muted-foreground/60">
            {selectedDocIds.length === 0
              ? "Querying all documents"
              : `${selectedDocIds.length} selected`}
          </p>
        </div>
      </div>

      {/* Main Query Area */}
      <div className="flex flex-1 flex-col">
        <ScrollArea className="flex-1">
          <div className="mx-auto max-w-3xl px-6 py-8">
            {/* Header */}
            <div className="mb-8 flex items-center gap-3">
              <div className="flex size-10 items-center justify-center rounded-xl bg-accent">
                <FileSearch className="size-5 text-foreground" />
              </div>
              <div>
                <h1 className="text-xl font-semibold text-foreground">DocuDeep</h1>
                <p className="text-sm text-muted-foreground">
                  Deep document analysis with source citations
                </p>
              </div>
            </div>

            {/* Query Input */}
            <div className="mb-6 space-y-3">
              <Textarea
                value={question}
                onChange={(e) => setQuestion(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === "Enter" && !e.shiftKey) {
                    e.preventDefault();
                    handleQuery();
                  }
                }}
                placeholder="Ask a detailed question about your documents..."
                className="min-h-[100px] resize-none bg-secondary/30 text-[15px]"
                disabled={loading}
              />
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  {selectedDocIds.length > 0 && (
                    <Badge variant="secondary">
                      {selectedDocIds.length} doc{selectedDocIds.length > 1 ? "s" : ""} selected
                    </Badge>
                  )}
                </div>
                <Button onClick={handleQuery} disabled={!question.trim() || loading}>
                  {loading ? (
                    <>
                      <Loader2 className="size-4 animate-spin" />
                      Analyzing...
                    </>
                  ) : (
                    <>
                      <Search className="size-4" />
                      Analyze
                    </>
                  )}
                </Button>
              </div>
            </div>

            {/* Error */}
            {error && (
              <div className="mb-6 flex items-center gap-2 rounded-lg border border-destructive/30 bg-destructive/10 px-4 py-3">
                <AlertCircle className="size-4 shrink-0 text-destructive" />
                <p className="text-sm text-destructive">{error}</p>
              </div>
            )}

            {/* Result */}
            {result && (
              <div className="space-y-6">
                {/* Answer */}
                <div className="rounded-xl border border-border bg-card p-5">
                  <div className="mb-3 flex items-center gap-2">
                    <h3 className="text-sm font-medium text-foreground">Answer</h3>
                    {!result.has_relevant_context && (
                      <Badge variant="secondary" className="text-amber-400">
                        Low confidence
                      </Badge>
                    )}
                  </div>
                  <div className="whitespace-pre-wrap text-[15px] leading-relaxed text-foreground">
                    {result.answer}
                  </div>
                </div>

                {/* Sources */}
                {result.sources.length > 0 && (
                  <div>
                    <h3 className="mb-3 text-sm font-medium text-muted-foreground">
                      Sources ({result.sources.length})
                    </h3>
                    <div className="space-y-2">
                      {result.sources.map((source, i) => (
                        <SourceCard key={i} source={source} index={i} />
                      ))}
                    </div>
                  </div>
                )}
              </div>
            )}

            {/* Empty state */}
            {!result && !error && !loading && (
              <div className="py-16 text-center">
                <FileSearch className="mx-auto mb-3 size-12 text-muted-foreground/30" />
                <p className="text-lg font-medium text-muted-foreground">
                  Ready to analyze
                </p>
                <p className="mt-1 text-sm text-muted-foreground/60">
                  Select documents on the left, type your question, and hit Analyze
                </p>
              </div>
            )}
          </div>
        </ScrollArea>
      </div>
    </div>
  );
}
