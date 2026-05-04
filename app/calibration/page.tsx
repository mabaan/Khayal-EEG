import { AppShell } from "@/components/app-shell";
import { SentenceGrid } from "@/components/sentence-grid";
import { LABELS } from "@/lib/sentence-set";

export default function CatalogPage() {
  return (
    <AppShell title="Catalog" subtitle="Review the Khayal sentence catalog and word vocabulary">
      <div className="space-y-5">
        <section className="kh-panel p-5">
          <p className="kh-kicker">Sentences</p>
          <div className="mt-3">
            <SentenceGrid />
          </div>
        </section>

        <section className="kh-panel p-5">
          <p className="kh-kicker">Words</p>
          <div className="mt-3 grid gap-3 sm:grid-cols-2 xl:grid-cols-3">
            {LABELS.map((label) => (
              <article key={label.id} className="rounded-xl border border-slate-200 bg-white px-4 py-3">
                <div className="flex items-center justify-between gap-3">
                  <p className="text-[11px] font-bold uppercase tracking-[0.13em] text-slate-500">
                    Word {label.id + 1}
                  </p>
                  <span className="rounded-full bg-slate-100 px-2 py-0.5 text-[10px] font-bold text-slate-600">
                    {label.word}
                  </span>
                </div>
                <p dir="rtl" className="kh-ar mt-2 text-lg font-semibold text-slate-900">
                  {label.arabic}
                </p>
                <p className="mt-1 text-xs text-slate-500">{label.english}</p>
              </article>
            ))}
          </div>
        </section>
      </div>
    </AppShell>
  );
}
