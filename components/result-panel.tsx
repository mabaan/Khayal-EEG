import type { InferenceResult } from "@/lib/types";
import { StatusBadge } from "@/components/status-badge";

interface ResultPanelProps {
  result: InferenceResult | null;
}

export function ResultPanel({ result }: ResultPanelProps) {
  return (
    <section className="kh-panel p-4 md:p-5">
      <div className="mb-3 flex items-start justify-between gap-3">
        <div>
          <p className="kh-kicker">Predicted Sentence</p>
          <h3 className="mt-1 text-base font-bold text-slate-900">Final Arabic Output</h3>
        </div>
        <StatusBadge label={result ? "Complete" : "Waiting"} tone={result ? "good" : "neutral"} />
      </div>

      <div className="rounded-xl border border-teal-100 bg-[linear-gradient(120deg,#ecfeff,#f0fdfa)] p-4">
        <p dir="rtl" className="kh-ar text-2xl font-bold leading-relaxed text-slate-900">
          {result?.final_sentence ?? "-"}
        </p>
        <p className="mt-2 text-xs text-slate-600">{result?.message ?? "Run a Session to decode the final sentence."}</p>
      </div>

      {result ? (
        <div className="mt-3 grid gap-2 text-xs text-slate-600">
          <p className="rounded-lg border border-slate-200 bg-slate-50 px-2 py-1.5">Selected sentence id: {result.selected_sentence_id}</p>
          <p className="rounded-lg border border-slate-200 bg-slate-50 px-2 py-1.5">Fallback used: {result.used_fallback ? "Yes" : "No"}</p>
        </div>
      ) : null}
    </section>
  );
}
