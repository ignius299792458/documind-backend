"use client";

import { useState } from "react";
import { FileText, ChevronDown, ChevronUp } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import type { SourceChunk } from "@/lib/types";
import { cn } from "@/lib/utils";

interface SourceCardProps {
  source: SourceChunk;
  index: number;
}

export function SourceCard({ source, index }: SourceCardProps) {
  const [expanded, setExpanded] = useState(false);

  const relevanceColor =
    source.relevance_score >= 0.7
      ? "text-emerald-400"
      : source.relevance_score >= 0.4
        ? "text-amber-400"
        : "text-red-400";

  return (
    <div className="rounded-lg border border-border bg-card">
      <button
        onClick={() => setExpanded(!expanded)}
        className="flex w-full items-center gap-3 px-3 py-2.5 text-left transition-colors hover:bg-accent/30"
      >
        <span className="flex size-5 shrink-0 items-center justify-center rounded bg-accent text-xs font-medium text-muted-foreground">
          {index + 1}
        </span>
        <FileText className="size-3.5 shrink-0 text-muted-foreground" />
        <span className="min-w-0 flex-1 truncate text-sm font-medium text-foreground">
          {source.filename}
        </span>
        {source.page !== null && (
          <Badge variant="secondary" className="shrink-0 text-xs">
            p.{source.page}
          </Badge>
        )}
        <span className={cn("shrink-0 text-xs font-medium", relevanceColor)}>
          {(source.relevance_score * 100).toFixed(0)}%
        </span>
        {expanded ? (
          <ChevronUp className="size-3.5 shrink-0 text-muted-foreground" />
        ) : (
          <ChevronDown className="size-3.5 shrink-0 text-muted-foreground" />
        )}
      </button>
      {expanded && (
        <div className="border-t border-border px-3 py-2.5">
          <p className="text-sm leading-relaxed text-muted-foreground whitespace-pre-wrap">
            {source.content}
          </p>
        </div>
      )}
    </div>
  );
}
