import type { Stage1Info } from "@/lib/types";

interface Stage1EvidencePanelProps {
  stage1: Stage1Info | null;
}

export function Stage1EvidencePanel({ stage1 }: Stage1EvidencePanelProps) {
  return (
    <section className="kh-panel p-5">
      <div className="mb-4">
        <p className="kh-kicker">Stage 1 Evidence</p>
        <h3 className="mt-1 text-lg font-bold text-slate-900">Top-k Word Evidence Per Slot</h3>
        <p className="mt-2 text-sm text-slate-600">Stage 2 uses these full posterior distributions instead of only the top-1 words.</p>
      </div>

      <div className="grid gap-4 xl:grid-cols-3">
        {(stage1?.slots ?? [{ slot: 1, probabilities: {}, top_k: [] }, { slot: 2, probabilities: {}, top_k: [] }, { slot: 3, probabilities: {}, top_k: [] }]).map((slot) => (
          <div key={slot.slot} className="rounded-2xl border border-slate-200 bg-white p-4">
            <div className="flex items-center justify-between gap-3">
              <div>
                <p className="text-xs font-semibold uppercase tracking-[0.12em] text-slate-500">Slot {slot.slot}</p>
                <p className="mt-1 text-sm font-semibold text-slate-800">{slot.top_k?.[0]?.word ?? "No evidence yet"}</p>
              </div>
              <span className="rounded-full border border-teal-100 bg-teal-50 px-2.5 py-1 text-[11px] font-bold text-teal-700">
                Top {slot.top_k?.length ?? 0}
              </span>
            </div>

            <div className="mt-4 space-y-3">
              {(slot.top_k ?? []).map((candidate) => {
                const width = `${Math.max(4, Math.round(candidate.probability * 100))}%`;
                return (
                  <div key={`${slot.slot}-${candidate.label_id}`}>
                    <div className="flex items-start justify-between gap-3">
                      <div className="min-w-0">
                        <p className="kh-ar text-base font-bold text-slate-900" dir="rtl">
                          {candidate.arabic}
                        </p>
                        <p className="text-xs font-semibold uppercase tracking-[0.12em] text-slate-500">{candidate.word}</p>
                      </div>
                      <p className="text-sm font-semibold text-slate-700">{(candidate.probability * 100).toFixed(1)}%</p>
                    </div>
                    <div className="mt-2 h-2.5 rounded-full bg-slate-100">
                      <div className="h-2.5 rounded-full bg-[linear-gradient(90deg,#0f766e,#0ea5c6)]" style={{ width }} />
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        ))}
      </div>
    </section>
  );
}
