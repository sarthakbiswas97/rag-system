"use client";

import { useCallback, useRef, useState } from "react";
import { query, type Citation, type QueryResponse } from "@/lib/api";
import { useToast } from "@/components/toast";

interface Message {
  id: string;
  role: "user" | "assistant";
  content: string;
  citations?: Citation[];
  confidence?: number;
  is_abstention?: boolean;
  timing?: QueryResponse["timing"];
}

const EXAMPLE_QUESTIONS = [
  "What are the key topics in my documents?",
  "Summarize the main findings",
  "What does the data say about performance?",
];

function ConfidenceBadge({ value }: { value: number }) {
  const pct = Math.round(value * 100);
  let color = "bg-green-100 text-green-800";
  if (pct < 50) color = "bg-red-100 text-red-800";
  else if (pct < 75) color = "bg-yellow-100 text-yellow-800";
  return (
    <span className={`inline-block rounded-full px-2.5 py-0.5 text-xs font-semibold ${color}`}>
      {pct}% confidence
    </span>
  );
}

function CitationCard({ citation }: { citation: Citation }) {
  return (
    <div className="rounded-lg border border-gray-200 bg-gray-50 px-3 py-2.5 text-xs">
      <div className="flex items-center gap-2">
        <span className="flex h-5 w-5 shrink-0 items-center justify-center rounded-full bg-blue-100 text-[10px] font-bold text-blue-700">
          {citation.index}
        </span>
        <span className="truncate font-semibold text-gray-800">
          {citation.source}
        </span>
      </div>
      <p className="mt-1.5 text-gray-600 line-clamp-3">{citation.snippet}</p>
    </div>
  );
}

function MessageBubble({ message }: { message: Message }) {
  const isUser = message.role === "user";

  return (
    <div className={`flex ${isUser ? "justify-end" : "justify-start"}`}>
      <div
        className={`max-w-[80%] rounded-xl px-4 py-3 ${
          isUser
            ? "bg-blue-600 text-white"
            : "border border-gray-200 bg-white text-gray-900 shadow-sm"
        }`}
      >
        {message.is_abstention && (
          <div className="mb-2.5 rounded-lg bg-yellow-50 border border-yellow-200 px-3 py-2 text-xs font-medium text-yellow-800">
            Could not find enough relevant information to answer confidently.
          </div>
        )}

        <p className="whitespace-pre-wrap text-sm leading-relaxed">{message.content}</p>

        {!isUser && message.confidence !== undefined && (
          <div className="mt-2.5 flex items-center gap-2">
            <ConfidenceBadge value={message.confidence} />
            {message.timing && (
              <span className="text-xs text-gray-400">
                {(message.timing.total_ms / 1000).toFixed(1)}s
              </span>
            )}
          </div>
        )}

        {message.citations && message.citations.length > 0 && (
          <div className="mt-3 space-y-2">
            <p className="text-xs font-semibold text-gray-500">Sources</p>
            {message.citations.map((c) => (
              <CitationCard key={c.chunk_id} citation={c} />
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

export default function ChatPage() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [lastFailedQuestion, setLastFailedQuestion] = useState<string | null>(null);
  const scrollRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);
  const { toast } = useToast();

  const scrollToBottom = useCallback(() => {
    requestAnimationFrame(() => {
      scrollRef.current?.scrollTo({
        top: scrollRef.current.scrollHeight,
        behavior: "smooth",
      });
    });
  }, []);

  const submitQuestion = async (question: string) => {
    if (!question || loading) return;

    const userMsg: Message = {
      id: crypto.randomUUID(),
      role: "user",
      content: question,
    };

    setMessages((prev) => [...prev, userMsg]);
    setInput("");
    setLoading(true);
    setLastFailedQuestion(null);
    scrollToBottom();

    try {
      const res = await query(question);
      const assistantMsg: Message = {
        id: crypto.randomUUID(),
        role: "assistant",
        content: res.answer,
        citations: res.citations,
        confidence: res.confidence,
        is_abstention: res.is_abstention,
        timing: res.timing,
      };
      setMessages((prev) => [...prev, assistantMsg]);
    } catch (err: unknown) {
      const detail = err instanceof Error ? err.message : "Something went wrong";
      toast.error(detail);
      setLastFailedQuestion(question);
      // Remove the user message since it failed
      setMessages((prev) => prev.filter((m) => m.id !== userMsg.id));
    } finally {
      setLoading(false);
      scrollToBottom();
      inputRef.current?.focus();
    }
  };

  const handleSubmit = async (e?: React.FormEvent) => {
    e?.preventDefault();
    await submitQuestion(input.trim());
  };

  const handleRetry = () => {
    if (lastFailedQuestion) {
      submitQuestion(lastFailedQuestion);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSubmit();
    }
  };

  return (
    <div className="flex h-full flex-col">
      {/* Messages area */}
      <div ref={scrollRef} className="flex-1 overflow-y-auto px-4 py-6">
        {messages.length === 0 && !loading ? (
          <div className="flex h-full flex-col items-center justify-center text-center">
            <svg
              className="mb-4 h-14 w-14 text-gray-300"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={1}
                d="M8 10h.01M12 10h.01M16 10h.01M9 16H5a2 2 0 01-2-2V6a2 2 0 012-2h14a2 2 0 012 2v8a2 2 0 01-2 2h-5l-5 5v-5z"
              />
            </svg>
            <p className="text-base font-semibold text-gray-700">
              Ask questions about your documents
            </p>
            <p className="mt-1 text-sm text-gray-400">
              Upload documents in the Documents tab first, then come back here to chat
            </p>

            {/* Example question chips */}
            <div className="mt-6 flex flex-wrap justify-center gap-2 max-w-lg">
              {EXAMPLE_QUESTIONS.map((q) => (
                <button
                  key={q}
                  onClick={() => {
                    setInput(q);
                    inputRef.current?.focus();
                  }}
                  className="rounded-full border border-gray-200 bg-white px-4 py-2 text-xs text-gray-600 hover:border-blue-300 hover:bg-blue-50 hover:text-blue-700 transition-colors shadow-sm"
                >
                  {q}
                </button>
              ))}
            </div>
          </div>
        ) : (
          <div className="mx-auto max-w-3xl space-y-4">
            {messages.map((msg) => (
              <MessageBubble key={msg.id} message={msg} />
            ))}
            {loading && (
              <div className="flex justify-start">
                <div className="rounded-xl border border-gray-200 bg-white px-4 py-3 shadow-sm">
                  <div className="flex items-center gap-2">
                    <div className="flex items-center gap-1">
                      <span className="h-1.5 w-1.5 animate-bounce rounded-full bg-blue-400 [animation-delay:0ms]" />
                      <span className="h-1.5 w-1.5 animate-bounce rounded-full bg-blue-400 [animation-delay:150ms]" />
                      <span className="h-1.5 w-1.5 animate-bounce rounded-full bg-blue-400 [animation-delay:300ms]" />
                    </div>
                    <span className="text-xs text-gray-500">
                      Searching your documents...
                    </span>
                  </div>
                </div>
              </div>
            )}
          </div>
        )}
      </div>

      {/* Retry bar */}
      {lastFailedQuestion && !loading && (
        <div className="border-t border-amber-200 bg-amber-50 px-4 py-2 flex items-center justify-between">
          <p className="text-xs text-amber-700">Last question failed to send</p>
          <button
            onClick={handleRetry}
            className="rounded-md bg-amber-100 px-3 py-1 text-xs font-semibold text-amber-800 hover:bg-amber-200 transition-colors"
          >
            Retry
          </button>
        </div>
      )}

      {/* Input area */}
      <div className="border-t border-gray-200 bg-white px-4 py-4">
        <form
          onSubmit={handleSubmit}
          className="mx-auto flex max-w-3xl items-end gap-3"
        >
          <textarea
            ref={inputRef}
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Ask a question..."
            rows={1}
            disabled={loading}
            className="flex-1 resize-none rounded-xl border border-gray-300 bg-white px-4 py-3 text-sm text-gray-900 placeholder-gray-400 focus:border-blue-500 focus:outline-none focus:ring-2 focus:ring-blue-500 disabled:opacity-50"
          />
          <button
            type="submit"
            disabled={loading || !input.trim()}
            className="rounded-xl bg-blue-600 px-5 py-3 text-sm font-semibold text-white hover:bg-blue-700 disabled:opacity-50 transition-colors"
          >
            Send
          </button>
        </form>
      </div>
    </div>
  );
}
