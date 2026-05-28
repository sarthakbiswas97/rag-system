"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useAuth } from "@/lib/auth-context";

const navItems = [
  { href: "/home", label: "Home" },
  { href: "/chat", label: "Chat" },
  { href: "/documents", label: "Documents" },
  { href: "/settings", label: "Settings" },
];

export function Sidebar() {
  const pathname = usePathname();
  const { tenant, logout } = useAuth();

  return (
    <aside className="flex h-screen w-60 flex-col border-r border-gray-200 bg-white">
      <div className="px-5 py-5 border-b border-gray-200">
        <h1 className="text-lg font-bold text-gray-900">RAG Platform</h1>
        {tenant && (
          <p className="text-xs text-gray-500 truncate mt-1">
            {tenant.name}
          </p>
        )}
      </div>

      <nav className="flex-1 px-3 py-4 space-y-1">
        {navItems.map((item) => {
          const isActive = pathname === item.href;
          return (
            <Link
              key={item.href}
              href={item.href}
              className={`block rounded-lg px-3 py-2.5 text-sm font-medium transition-colors ${
                isActive
                  ? "bg-blue-50 text-blue-700"
                  : "text-gray-600 hover:bg-gray-100 hover:text-gray-900"
              }`}
            >
              {item.label}
            </Link>
          );
        })}
      </nav>

      <div className="border-t border-gray-200 px-5 py-4">
        <button
          onClick={logout}
          className="text-sm font-medium text-gray-500 hover:text-gray-900 transition-colors"
        >
          Sign Out
        </button>
      </div>
    </aside>
  );
}
