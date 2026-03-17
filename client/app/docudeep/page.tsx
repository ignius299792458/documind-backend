"use client";

import { useState } from "react";
import { Sidebar } from "@/components/sidebar";
import { DocuDeepView } from "@/components/docudeep/docudeep-view";

export default function DocuDeepPage() {
  const [refreshTrigger, setRefreshTrigger] = useState(0);

  return (
    <div className="flex h-screen bg-background">
      <Sidebar onDocumentsChange={() => setRefreshTrigger((n) => n + 1)} />
      <DocuDeepView refreshTrigger={refreshTrigger} />
    </div>
  );
}
