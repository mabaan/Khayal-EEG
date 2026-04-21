import { clsx } from "clsx";

interface StatusBadgeProps {
  label: string;
  tone?: "neutral" | "good" | "warn" | "error";
}

export function StatusBadge({ label, tone = "neutral" }: StatusBadgeProps) {
  return (
    <span
      className={clsx(
        "inline-flex items-center gap-1.5 rounded-full border px-2.5 py-1 text-[11px] font-bold tracking-wide",
        tone === "neutral" && "border-slate-200 bg-slate-50 text-slate-700",
        tone === "good" && "border-emerald-200 bg-emerald-50 text-emerald-700",
        tone === "warn" && "border-amber-200 bg-amber-50 text-amber-700",
        tone === "error" && "border-rose-200 bg-rose-50 text-rose-700"
      )}
    >
      <span className="h-1.5 w-1.5 rounded-full bg-current opacity-80" />
      {label}
    </span>
  );
}
