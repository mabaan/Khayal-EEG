import type { PredictionInfo, Stage2Info } from "@/lib/types";

interface Stage2CandidatesPanelProps {
  stage2: Stage2Info | null;
  prediction: PredictionInfo | null;
}

export function Stage2CandidatesPanel({ stage2, prediction }: Stage2CandidatesPanelProps) {
  const candidates = stage2?.candidate_sentences ?? [];
  return (
    <section className="kh-panel p-5">
      <div className="mb-4">
        <p className="kh-kicker">Stage 2 Candidates</p>
        <h3 className="mt-1 text-lg font-bold text-slate-900">Sentence Shortlist and Selection</h3>
      </div>

      <div className="space-y-3">
        {candidates.length === 0 ? (
          <div className="rounded-2xl border border-slate-200 bg-slate-50 px-4 py-5 text-sm text-slate-500">
            No Stage 2 candidates yet.
          </div>
        ) : (
          candidates.map((candidate) => {
            const selected = candidate.sentence_id === prediction?.sentence_id;
            return (
              <div
                key={candidate.sentence_id}
                className={`rounded-2xl border px-4 py-4 ${selected ? "border-teal-200 bg-teal-50/70" : "border-slate-200 bg-white"}`}
              >
                <div className="flex flex-wrap items-center justify-between gap-3">
                  <div className="flex items-center gap-2">
                    <span className="rounded-full border border-slate-200 bg-slate-50 px-2.5 py-1 text-[11px] font-bold text-slate-600">
                      Rank {candidate.rank}
                    </span>
                    <span className="rounded-full border border-slate-200 bg-white px-2.5 py-1 text-[11px] font-bold text-slate-600">
                      {candidate.sentence_id}
                    </span>
                    {selected ? (
                      <span className="rounded-full border border-emerald-200 bg-emerald-50 px-2.5 py-1 text-[11px] font-bold text-emerald-700">
                        Selected
                      </span>
                    ) : null}
                  </div>
                  <p className="text-sm font-semibold text-slate-700">{candidate.retrieval_score.toFixed(3)}</p>
                </div>

                <p dir="rtl" className="kh-ar mt-3 text-xl font-bold text-slate-900">
                  {candidate.arabic}
                </p>
                <p className="mt-1 text-sm font-semibold uppercase tracking-[0.12em] text-slate-500">{candidate.romanized}</p>

                <div className="mt-3 grid gap-2 text-xs text-slate-500 sm:grid-cols-3">
                  <p className="rounded-xl border border-slate-200 bg-slate-50 px-3 py-2">Posterior: {candidate.posterior_score.toFixed(3)}</p>
                  <p className="rounded-xl border border-slate-200 bg-slate-50 px-3 py-2">
                    Transformer: {typeof candidate.transformer_score === "number" ? candidate.transformer_score.toFixed(3) : "Unavailable"}
                  </p>
                  <p className="rounded-xl border border-slate-200 bg-slate-50 px-3 py-2">
                    Slot probs: {candidate.word_probabilities.map((value) => value.toFixed(3)).join(" / ")}
                  </p>
                </div>
              </div>
            );
          })
        )}
      </div>
    </section>
  );
}
