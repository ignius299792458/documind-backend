"use client";

import { useState } from "react";
import { Sidebar } from "@/components/sidebar";
import { ChatView } from "@/components/chat/chat-view";

export default function ChatPage() {
  const [refreshTrigger, setRefreshTrigger] = useState(0);

  return (
    <div className="flex h-screen bg-background">
      <Sidebar onDocumentsChange={() => setRefreshTrigger((n) => n + 1)} />
      <ChatView refreshTrigger={refreshTrigger} />
    </div>
  );
}
