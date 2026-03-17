"use client";

import { useState } from "react";
import { Globe, Loader2 } from "lucide-react";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { ingestURL } from "@/lib/api";

interface UrlIngestDialogProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  onSuccess?: () => void;
}

export function UrlIngestDialog({
  open,
  onOpenChange,
  onSuccess,
}: UrlIngestDialogProps) {
  const [url, setUrl] = useState("");
  const [displayName, setDisplayName] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleSubmit = async () => {
    if (!url.trim()) return;
    setLoading(true);
    setError(null);

    try {
      await ingestURL({
        url: url.trim(),
        display_name: displayName.trim() || undefined,
      });
      setUrl("");
      setDisplayName("");
      onOpenChange(false);
      onSuccess?.();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Ingestion failed");
    } finally {
      setLoading(false);
    }
  };

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="sm:max-w-md">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2">
            <Globe className="size-4" />
            Ingest from URL
          </DialogTitle>
        </DialogHeader>

        <div className="space-y-3">
          <div>
            <label className="mb-1.5 block text-sm text-muted-foreground">
              URL
            </label>
            <Input
              placeholder="https://example.com/document"
              value={url}
              onChange={(e) => setUrl(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && handleSubmit()}
            />
          </div>
          <div>
            <label className="mb-1.5 block text-sm text-muted-foreground">
              Display Name (optional)
            </label>
            <Input
              placeholder="My Document"
              value={displayName}
              onChange={(e) => setDisplayName(e.target.value)}
            />
          </div>
        </div>

        {error && (
          <p className="text-sm text-destructive">{error}</p>
        )}

        <div className="flex justify-end gap-2">
          <Button
            variant="ghost"
            onClick={() => onOpenChange(false)}
            disabled={loading}
          >
            Cancel
          </Button>
          <Button
            onClick={handleSubmit}
            disabled={!url.trim() || loading}
          >
            {loading ? (
              <>
                <Loader2 className="size-4 animate-spin" />
                Ingesting...
              </>
            ) : (
              "Ingest"
            )}
          </Button>
        </div>
      </DialogContent>
    </Dialog>
  );
}
