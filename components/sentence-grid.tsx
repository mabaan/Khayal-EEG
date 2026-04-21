import { SENTENCE_CATALOG } from "@/lib/sentence-set";

export function SentenceGrid() {
  return (
    <div className="grid gap-3 md:grid-cols-2">
      {SENTENCE_CATALOG.map((sentence) => (
        <article key={sentence.sentence_id} className="rounded-xl border border-slate-200 bg-white px-4 py-3 transition hover:border-teal-200 hover:shadow-md">
          <div className="flex items-center justify-between">
            <p className="text-[11px] font-bold uppercase tracking-[0.13em] text-slate-500">Sentence {sentence.sentence_id}</p>
            <span className="rounded-full bg-slate-100 px-2 py-0.5 text-[10px] font-bold text-slate-600">3 words</span>
          </div>
          <p dir="rtl" className="kh-ar mt-2 text-lg font-semibold text-slate-900">
            {sentence.arabic}
          </p>
          <p className="mt-2 text-xs text-slate-500">{sentence.word_tokens.join(" | ")}</p>
        </article>
      ))}
    </div>
  );
}
