import type { InferenceResult } from "@/lib/types";
import { StatusBadge } from "@/components/status-badge";

interface FinalPredictionCardProps {
  result: InferenceResult | null;
}

export function FinalPredictionCard({ result }: FinalPredictionCardProps) {
  const ready = Boolean(result?.prediction?.arabic);
  return (
    <section className="kh-panel-strong p-5">
      <div className="flex items-start justify-between gap-3">
        <div className="min-w-0 flex-1">
          <p className="kh-kicker">Predicted Sentence</p>
          <h3 className="mt-1 text-xl font-bold text-slate-900">Final Decoded Output</h3>
        </div>
        <StatusBadge label={ready ? "Ready" : "Waiting"} tone={ready ? "good" : "neutral"} />
      </div>

      <div className="mt-5 rounded-[28px] border border-teal-100 bg-[linear-gradient(135deg,#f8fffe,#ecfeff)] px-6 py-8 text-center shadow-[0_14px_36px_rgba(14,116,144,0.08)]">
        <p dir="rtl" className="kh-ar text-4xl font-bold leading-[1.7] text-slate-900 md:text-5xl">
          {result?.prediction.arabic ?? "-"}
        </p>
        <p className="mt-4 break-words text-base font-semibold uppercase tracking-[0.18em] text-slate-500">
          {result?.prediction.romanized ?? "Awaiting inference"}
        </p>
      </div>

      <div className="mt-4 grid gap-3 sm:grid-cols-3">
        <div className="rounded-2xl border border-slate-200 bg-white px-4 py-3">
          <p className="text-xs font-semibold uppercase tracking-[0.12em] text-slate-500">Sentence ID</p>
          <p className="mt-1 text-lg font-bold text-slate-900">{result?.prediction.sentence_id ?? "-"}</p>
        </div>
        <div className="rounded-2xl border border-slate-200 bg-white px-4 py-3">
          <p className="text-xs font-semibold uppercase tracking-[0.12em] text-slate-500">Selected Score</p>
          <p className="mt-1 text-lg font-bold text-slate-900">
            {typeof result?.prediction.score === "number" ? result.prediction.score.toFixed(3) : "-"}
          </p>
        </div>
        <div className="rounded-2xl border border-slate-200 bg-white px-4 py-3">
          <p className="text-xs font-semibold uppercase tracking-[0.12em] text-slate-500">Fallback Used</p>
          <p className="mt-1 text-lg font-bold text-slate-900">{result?.stage2.used_fallback ? "Yes" : ready ? "No" : "-"}</p>
        </div>
      </div>
    </section>
  );
}
