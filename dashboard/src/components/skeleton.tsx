export function SkeletonCard() {
  return (
    <div className="rounded-xl border border-gray-200 bg-white p-6 shadow-sm">
      <div className="h-4 w-20 animate-pulse rounded bg-gray-200" />
      <div className="mt-3 h-8 w-16 animate-pulse rounded bg-gray-200" />
      <div className="mt-2 h-3 w-28 animate-pulse rounded bg-gray-100" />
    </div>
  );
}

export function SkeletonRow() {
  return (
    <div className="flex items-center justify-between px-6 py-3">
      <div className="flex items-center gap-3">
        <div className="h-5 w-14 animate-pulse rounded-full bg-gray-200" />
        <div className="h-4 w-24 animate-pulse rounded bg-gray-200" />
      </div>
      <div className="h-3 w-32 animate-pulse rounded bg-gray-100" />
    </div>
  );
}

export function SkeletonShell() {
  return (
    <div className="flex h-screen bg-gray-50">
      <div className="w-60 border-r border-gray-200 bg-white p-5">
        <div className="h-6 w-28 animate-pulse rounded bg-gray-200" />
        <div className="mt-6 space-y-3">
          {[1, 2, 3, 4].map((i) => (
            <div key={i} className="h-8 w-full animate-pulse rounded-lg bg-gray-100" />
          ))}
        </div>
      </div>
      <div className="flex-1 p-8">
        <div className="h-7 w-40 animate-pulse rounded bg-gray-200" />
        <div className="mt-2 h-4 w-56 animate-pulse rounded bg-gray-100" />
        <div className="mt-8 grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
          {[1, 2, 3, 4].map((i) => (
            <SkeletonCard key={i} />
          ))}
        </div>
      </div>
    </div>
  );
}
