"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import {
  MessageSquare,
  FileSearch,
  Plus,
  Link as LinkIcon,
  Brain,
} from "lucide-react";
import { cn } from "@/lib/utils";
import { Button } from "@/components/ui/button";
import { UploadDialog } from "@/components/upload-dialog";
import { UrlIngestDialog } from "@/components/url-ingest-dialog";
import { useState } from "react";

const navItems = [
  { href: "/", label: "Chat", icon: MessageSquare },
  { href: "/docudeep", label: "DocuDeep", icon: FileSearch },
];

interface SidebarProps {
  onDocumentsChange?: () => void;
}

export function Sidebar({ onDocumentsChange }: SidebarProps) {
  const pathname = usePathname();
  const [uploadOpen, setUploadOpen] = useState(false);
  const [urlOpen, setUrlOpen] = useState(false);

  const handleIngestSuccess = () => {
    onDocumentsChange?.();
  };

  return (
    <>
      <aside className="flex h-screen w-[220px] flex-col border-r border-border bg-sidebar">
        <div className="flex items-center gap-2 px-4 py-5">
          <Brain className="size-6 text-foreground" />
          <span className="text-lg font-semibold tracking-tight text-foreground">
            DocuMind
          </span>
        </div>

        <nav className="flex-1 space-y-1 px-2">
          {navItems.map((item) => {
            const isActive = pathname === item.href;
            return (
              <Link key={item.href} href={item.href}>
                <div
                  className={cn(
                    "flex items-center gap-3 rounded-lg px-3 py-2.5 text-sm font-medium transition-colors",
                    isActive
                      ? "bg-accent text-accent-foreground"
                      : "text-muted-foreground hover:bg-accent/50 hover:text-foreground"
                  )}
                >
                  <item.icon className="size-4" />
                  {item.label}
                </div>
              </Link>
            );
          })}
        </nav>

        <div className="space-y-1.5 border-t border-border p-3">
          <Button
            variant="outline"
            size="sm"
            className="w-full justify-start gap-2"
            onClick={() => setUploadOpen(true)}
          >
            <Plus className="size-3.5" />
            Upload File
          </Button>
          <Button
            variant="ghost"
            size="sm"
            className="w-full justify-start gap-2 text-muted-foreground"
            onClick={() => setUrlOpen(true)}
          >
            <LinkIcon className="size-3.5" />
            Ingest URL
          </Button>
        </div>
      </aside>

      <UploadDialog
        open={uploadOpen}
        onOpenChange={setUploadOpen}
        onSuccess={handleIngestSuccess}
      />
      <UrlIngestDialog
        open={urlOpen}
        onOpenChange={setUrlOpen}
        onSuccess={handleIngestSuccess}
      />
    </>
  );
}
