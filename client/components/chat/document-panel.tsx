"use client";

import { useState, useEffect, useCallback } from "react";
import {
  FileText,
  Trash2,
  RefreshCw,
  CheckSquare,
  Square,
  ChevronRight,
  Loader2,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Badge } from "@/components/ui/badge";
import { getDocuments, deleteDocument } from "@/lib/api";
import type { DocumentMeta } from "@/lib/types";
import { cn } from "@/lib/utils";

interface DocumentPanelProps {
  selectedDocIds: string[];
  onSelectionChange: (ids: string[]) => void;
  refreshTrigger?: number;
}

export function DocumentPanel({
  selectedDocIds,
  onSelectionChange,
  refreshTrigger,
}: DocumentPanelProps) {
  const [documents, setDocuments] = useState<DocumentMeta[]>([]);
  const [loading, setLoading] = useState(true);
  const [collapsed, setCollapsed] = useState(false);

  const fetchDocs = useCallback(async () => {
    setLoading(true);
    try {
      const res = await getDocuments();
      setDocuments(res.documents);
    } catch {
      // silently fail
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchDocs();
  }, [fetchDocs, refreshTrigger]);

  const toggleDoc = (docId: string) => {
    if (selectedDocIds.includes(docId)) {
      onSelectionChange(selectedDocIds.filter((id) => id !== docId));
    } else {
      onSelectionChange([...selectedDocIds, docId]);
    }
  };

  const toggleAll = () => {
    if (selectedDocIds.length === documents.length) {
      onSelectionChange([]);
    } else {
      onSelectionChange(documents.map((d) => d.doc_id));
    }
  };

  const handleDelete = async (docId: string) => {
    try {
      await deleteDocument(docId);
      onSelectionChange(selectedDocIds.filter((id) => id !== docId));
      fetchDocs();
    } catch {
      // silently fail
    }
  };

  const fileIcon = (fileType: string) => {
    const colors: Record<string, string> = {
      "application/pdf": "text-red-400",
      "text/markdown": "text-blue-400",
      "text/plain": "text-zinc-400",
      "text/html": "text-orange-400",
    };
    return colors[fileType] || "text-zinc-400";
  };

  if (collapsed) {
    return (
      <div className="flex h-screen w-10 flex-col items-center border-l border-border bg-sidebar pt-4">
        <Button
          variant="ghost"
          size="icon-xs"
          onClick={() => setCollapsed(false)}
          className="text-muted-foreground"
        >
          <ChevronRight className="size-4 rotate-180" />
        </Button>
      </div>
    );
  }

  return (
    <div className="flex h-screen w-[280px] flex-col border-l border-border bg-sidebar">
      <div className="flex items-center justify-between border-b border-border px-3 py-3">
        <h3 className="text-sm font-medium text-foreground">Documents</h3>
        <div className="flex items-center gap-1">
          <Button
            variant="ghost"
            size="icon-xs"
            onClick={fetchDocs}
            disabled={loading}
            className="text-muted-foreground"
          >
            <RefreshCw className={cn("size-3", loading && "animate-spin")} />
          </Button>
          <Button
            variant="ghost"
            size="icon-xs"
            onClick={() => setCollapsed(true)}
            className="text-muted-foreground"
          >
            <ChevronRight className="size-3" />
          </Button>
        </div>
      </div>

      {documents.length > 0 && (
        <div className="flex items-center gap-2 border-b border-border px-3 py-2">
          <Button
            variant="ghost"
            size="xs"
            onClick={toggleAll}
            className="gap-1.5 text-xs text-muted-foreground"
          >
            {selectedDocIds.length === documents.length ? (
              <CheckSquare className="size-3" />
            ) : (
              <Square className="size-3" />
            )}
            {selectedDocIds.length === documents.length
              ? "Deselect All"
              : "Select All"}
          </Button>
          {selectedDocIds.length > 0 && (
            <Badge variant="secondary" className="text-xs">
              {selectedDocIds.length} selected
            </Badge>
          )}
        </div>
      )}

      <ScrollArea className="flex-1">
        {loading && documents.length === 0 ? (
          <div className="flex items-center justify-center py-12">
            <Loader2 className="size-5 animate-spin text-muted-foreground" />
          </div>
        ) : documents.length === 0 ? (
          <div className="px-4 py-12 text-center">
            <FileText className="mx-auto mb-2 size-8 text-muted-foreground/40" />
            <p className="text-sm text-muted-foreground">No documents yet</p>
            <p className="mt-1 text-xs text-muted-foreground/60">
              Upload files using the sidebar
            </p>
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
                    "group flex cursor-pointer items-start gap-2.5 rounded-lg px-2.5 py-2 transition-colors",
                    isSelected
                      ? "bg-accent/80"
                      : "hover:bg-accent/40"
                  )}
                >
                  <div className="mt-0.5">
                    {isSelected ? (
                      <CheckSquare className="size-3.5 text-foreground" />
                    ) : (
                      <Square className="size-3.5 text-muted-foreground" />
                    )}
                  </div>

                  <div className="min-w-0 flex-1">
                    <div className="flex items-center gap-1.5">
                      <FileText
                        className={cn("size-3.5 shrink-0", fileIcon(doc.file_type))}
                      />
                      <p className="truncate text-sm font-medium text-foreground">
                        {doc.filename}
                      </p>
                    </div>
                    {doc.chunk_count && (
                      <p className="mt-0.5 text-xs text-muted-foreground">
                        {doc.chunk_count} chunks
                        {doc.page_count ? ` · ${doc.page_count} pages` : ""}
                      </p>
                    )}
                  </div>

                  <Button
                    variant="ghost"
                    size="icon-xs"
                    onClick={(e) => {
                      e.stopPropagation();
                      handleDelete(doc.doc_id);
                    }}
                    className="shrink-0 opacity-0 transition-opacity group-hover:opacity-100"
                  >
                    <Trash2 className="size-3 text-muted-foreground hover:text-destructive" />
                  </Button>
                </div>
              );
            })}
          </div>
        )}
      </ScrollArea>

      <div className="border-t border-border px-3 py-2">
        <p className="text-xs text-muted-foreground/60">
          {selectedDocIds.length === 0
            ? "No filter — querying all docs"
            : `Querying ${selectedDocIds.length} doc${selectedDocIds.length > 1 ? "s" : ""}`}
        </p>
      </div>
    </div>
  );
}
