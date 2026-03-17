"use client";

import { Brain, User } from "lucide-react";
import type { ChatMessage as ChatMessageType } from "@/lib/types";
import { cn } from "@/lib/utils";

interface ChatMessageProps {
  message: ChatMessageType;
}

export function ChatMessage({ message }: ChatMessageProps) {
  const isUser = message.role === "user";

  return (
    <div
      className={cn(
        "group flex gap-3 px-4 py-4",
        isUser ? "justify-end" : "justify-start"
      )}
    >
      {!isUser && (
        <div className="flex size-7 shrink-0 items-center justify-center rounded-full bg-accent">
          <Brain className="size-4 text-foreground" />
        </div>
      )}

      <div
        className={cn(
          "max-w-[720px] space-y-2",
          isUser ? "text-right" : "text-left"
        )}
      >
        <div
          className={cn(
            "inline-block rounded-2xl px-4 py-2.5 text-[15px] leading-relaxed",
            isUser
              ? "bg-accent text-foreground"
              : "text-foreground"
          )}
        >
          <div className="whitespace-pre-wrap">{message.content}</div>
          {message.isStreaming && (
            <span className="ml-0.5 inline-block h-4 w-0.5 animate-pulse bg-foreground/60" />
          )}
        </div>
      </div>

      {isUser && (
        <div className="flex size-7 shrink-0 items-center justify-center rounded-full bg-foreground/10">
          <User className="size-4 text-foreground" />
        </div>
      )}
    </div>
  );
}
