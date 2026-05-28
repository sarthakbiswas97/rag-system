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
    <aside className="flex h-screen w-56 flex-col border-r bg-white">
      <div className="px-4 py-5 border-b">
        <h1 className="text-lg font-semibold">RAG Platform</h1>
        {tenant && (
          <p className="text-xs text-gray-500 truncate mt-0.5">
            {tenant.name}
          </p>
        )}
      </div>

      <nav className="flex-1 px-2 py-4 space-y-1">
        {navItems.map((item) => {
          const isActive = pathname === item.href;
          return (
            <Link
              key={item.href}
              href={item.href}
              className={`block rounded-md px-3 py-2 text-sm ${
                isActive
                  ? "bg-gray-100 font-medium text-gray-900"
                  : "text-gray-600 hover:bg-gray-50 hover:text-gray-900"
              }`}
            >
              {item.label}
            </Link>
          );
        })}
      </nav>

      <div className="border-t px-4 py-3">
        <button
          onClick={logout}
          className="text-sm text-gray-500 hover:text-gray-900"
        >
          Sign Out
        </button>
      </div>
    </aside>
  );
}
