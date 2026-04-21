export const dynamic = "force-dynamic";

import Link from "next/link";
import { AppShell } from "@/components/app-shell";
import { ModelCard } from "@/components/model-card";
import { SessionTable } from "@/components/session-table";
import { getActiveProfile, listProfiles } from "@/lib/profile-store";
import { listSessionRecords } from "@/lib/history-store";

const actions = [
  {
    href: "/setup",
    title: "Setup Profile",
    description: "Create/select profile and confirm base model path."
  },
  {
    href: "/calibration",
    title: "Calibration Upload",
    description: "Upload raw EDF files using the fixed prompt protocol."
  },
  {
    href: "/training",
    title: "Train Stage 1",
    description: "Fine-tune personalized Diff-E checkpoint."
  },
  {
    href: "/inference",
    title: "Start Session",
    description: "Decode sentence with fixed Diff-E + RAG flow."
  }
];

export default async function HomePage() {
  const [activeProfile, profiles, sessions] = await Promise.all([
    getActiveProfile(),
    listProfiles(),
    listSessionRecords()
  ]);

  const recent = sessions.slice(0, 8);

  return (
    <AppShell title="Home" subtitle="Local-first imagined speech workflow with one fixed production path">
      <section className="kh-panel-strong mb-5 p-5 md:p-6">
        <p className="kh-kicker">Overview</p>
        <div className="mt-3 grid gap-3 md:grid-cols-3">
          <article className="rounded-xl border border-slate-200 bg-white p-4">
            <p className="text-xs font-semibold uppercase tracking-[0.12em] text-slate-500">Active Profile</p>
            <p className="mt-2 truncate text-lg font-bold text-slate-900">{activeProfile?.name ?? "None Selected"}</p>
          </article>
          <article className="rounded-xl border border-slate-200 bg-white p-4">
            <p className="text-xs font-semibold uppercase tracking-[0.12em] text-slate-500">Profiles</p>
            <p className="mt-2 text-lg font-bold text-slate-900">{profiles.length}</p>
          </article>
          <article className="rounded-xl border border-slate-200 bg-white p-4">
            <p className="text-xs font-semibold uppercase tracking-[0.12em] text-slate-500">Sessions Logged</p>
            <p className="mt-2 text-lg font-bold text-slate-900">{sessions.length}</p>
          </article>
        </div>
      </section>

      <div className="grid gap-5 xl:grid-cols-3">
        <section className="kh-panel p-5 xl:col-span-2">
          <p className="kh-kicker">Quick Actions</p>
          <h2 className="mt-1 text-lg font-bold text-slate-900">Run the default end-to-end flow</h2>

          <div className="mt-4 grid gap-3 sm:grid-cols-2">
            {actions.map((action) => (
              <Link key={action.href} href={action.href} className="rounded-xl border border-slate-200 bg-white p-4 transition hover:border-teal-200 hover:shadow-md">
                <p className="text-sm font-bold text-slate-900">{action.title}</p>
                <p className="mt-1 text-xs text-slate-600">{action.description}</p>
              </Link>
            ))}
          </div>
        </section>

        <ModelCard profile={activeProfile} />
      </div>

      <section className="mt-6">
        <div className="mb-2">
          <p className="kh-kicker">Recent Sessions</p>
        </div>
        <SessionTable sessions={recent} />
      </section>
    </AppShell>
  );
}
