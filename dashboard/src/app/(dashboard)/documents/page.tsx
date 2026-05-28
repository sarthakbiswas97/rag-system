"use client";

import { useCallback, useRef, useState } from "react";
import { ingestFiles, type IngestResponse } from "@/lib/api";

type UploadState = "idle" | "uploading" | "done" | "error";

const ACCEPTED_TYPES = [
  "application/pdf",
  "text/plain",
  "text/markdown",
  "text/csv",
];
const MAX_FILE_SIZE_MB = 50;
const MAX_FILE_SIZE = MAX_FILE_SIZE_MB * 1024 * 1024;

function validateFiles(files: File[]): string | null {
  for (const file of files) {
    if (file.size > MAX_FILE_SIZE) {
      return `${file.name} exceeds ${MAX_FILE_SIZE_MB}MB limit`;
    }
  }
  return null;
}

export default function DocumentsPage() {
  const [files, setFiles] = useState<File[]>([]);
  const [dragOver, setDragOver] = useState(false);
  const [state, setState] = useState<UploadState>("idle");
  const [result, setResult] = useState<IngestResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  const addFiles = useCallback((incoming: FileList | File[]) => {
    const arr = Array.from(incoming).filter(
      (f) =>
        ACCEPTED_TYPES.includes(f.type) ||
        f.name.endsWith(".md") ||
        f.name.endsWith(".txt") ||
        f.name.endsWith(".pdf") ||
        f.name.endsWith(".csv"),
    );
    setFiles((prev) => [...prev, ...arr]);
    setState("idle");
    setResult(null);
    setError(null);
  }, []);

  const removeFile = useCallback((index: number) => {
    setFiles((prev) => prev.filter((_, i) => i !== index));
  }, []);

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setDragOver(false);
      if (e.dataTransfer.files.length > 0) {
        addFiles(e.dataTransfer.files);
      }
    },
    [addFiles],
  );

  const handleUpload = async () => {
    if (files.length === 0) return;

    const validationError = validateFiles(files);
    if (validationError) {
      setError(validationError);
      return;
    }

    setState("uploading");
    setError(null);
    setResult(null);

    try {
      const res = await ingestFiles(files);
      setResult(res);
      setState("done");
      setFiles([]);
    } catch (err: unknown) {
      const message =
        err instanceof Error ? err.message : "Upload failed";
      setError(message);
      setState("error");
    }
  };

  const formatSize = (bytes: number): string => {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  };

  return (
    <div className="max-w-3xl">
      <h1 className="text-2xl font-semibold text-gray-900">Documents</h1>
      <p className="mt-1 text-sm text-gray-500">
        Upload documents to index them for RAG queries.
      </p>

      {/* Drop zone */}
      <div
        onDragOver={(e) => {
          e.preventDefault();
          setDragOver(true);
        }}
        onDragLeave={() => setDragOver(false)}
        onDrop={handleDrop}
        onClick={() => inputRef.current?.click()}
        className={`mt-6 flex cursor-pointer flex-col items-center justify-center rounded-lg border-2 border-dashed p-10 transition-colors ${
          dragOver
            ? "border-blue-400 bg-blue-50"
            : "border-gray-300 bg-gray-50 hover:border-gray-400"
        }`}
      >
        <svg
          className="mb-3 h-10 w-10 text-gray-400"
          fill="none"
          stroke="currentColor"
          viewBox="0 0 24 24"
        >
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            strokeWidth={1.5}
            d="M12 16V4m0 0l-4 4m4-4l4 4M4 20h16"
          />
        </svg>
        <p className="text-sm text-gray-600">
          <span className="font-medium text-blue-600">Click to upload</span>
          {" "}or drag and drop
        </p>
        <p className="mt-1 text-xs text-gray-400">
          PDF, TXT, Markdown, CSV — up to {MAX_FILE_SIZE_MB}MB each
        </p>
        <input
          ref={inputRef}
          type="file"
          multiple
          accept=".pdf,.txt,.md,.csv"
          className="hidden"
          onChange={(e) => {
            if (e.target.files) addFiles(e.target.files);
            e.target.value = "";
          }}
        />
      </div>

      {/* File list */}
      {files.length > 0 && (
        <div className="mt-4 space-y-2">
          {files.map((file, i) => (
            <div
              key={`${file.name}-${i}`}
              className="flex items-center justify-between rounded-md border bg-white px-4 py-2"
            >
              <div className="min-w-0 flex-1">
                <p className="truncate text-sm text-gray-900">{file.name}</p>
                <p className="text-xs text-gray-400">{formatSize(file.size)}</p>
              </div>
              <button
                onClick={() => removeFile(i)}
                className="ml-3 text-gray-400 hover:text-red-500"
                disabled={state === "uploading"}
              >
                <svg className="h-4 w-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                </svg>
              </button>
            </div>
          ))}

          <button
            onClick={handleUpload}
            disabled={state === "uploading"}
            className="mt-2 rounded-md bg-blue-600 px-4 py-2 text-sm font-medium text-white hover:bg-blue-700 disabled:opacity-50"
          >
            {state === "uploading" ? "Uploading..." : `Upload ${files.length} file${files.length > 1 ? "s" : ""}`}
          </button>
        </div>
      )}

      {/* Result */}
      {result && (
        <div className="mt-6 rounded-lg border border-green-200 bg-green-50 p-4">
          <h3 className="text-sm font-medium text-green-800">Upload complete</h3>
          <dl className="mt-2 grid grid-cols-2 gap-2 text-sm text-green-700">
            <div>
              <dt className="text-xs text-green-600">Processed</dt>
              <dd className="font-medium">{result.documents_processed}</dd>
            </div>
            <div>
              <dt className="text-xs text-green-600">Chunks created</dt>
              <dd className="font-medium">{result.chunks_created}</dd>
            </div>
            <div>
              <dt className="text-xs text-green-600">Skipped</dt>
              <dd className="font-medium">{result.documents_skipped}</dd>
            </div>
            <div>
              <dt className="text-xs text-green-600">Time</dt>
              <dd className="font-medium">{(result.elapsed_ms / 1000).toFixed(1)}s</dd>
            </div>
          </dl>
        </div>
      )}

      {/* Error */}
      {error && (
        <div className="mt-4 rounded-lg border border-red-200 bg-red-50 p-4">
          <p className="text-sm text-red-700">{error}</p>
        </div>
      )}
    </div>
  );
}
