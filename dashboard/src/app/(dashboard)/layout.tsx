"use client";

import { useEffect } from "react";
import { useRouter } from "next/navigation";
import { useAuth } from "@/lib/auth-context";
import { Sidebar } from "@/components/sidebar";
import { SkeletonShell } from "@/components/skeleton";

export default function DashboardLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  const { apiKey, isLoading } = useAuth();
  const router = useRouter();

  useEffect(() => {
    if (!isLoading && !apiKey) {
      router.replace("/login");
    }
  }, [apiKey, isLoading, router]);

  if (isLoading) {
    return <SkeletonShell />;
  }

  if (!apiKey) return null;

  return (
    <div className="flex h-screen bg-gray-50">
      <Sidebar />
      <main className="flex-1 overflow-auto p-8">{children}</main>
    </div>
  );
}
