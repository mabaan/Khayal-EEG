export const dynamic = "force-dynamic";

import { AppShell } from "@/components/app-shell";
import { SessionTable } from "@/components/session-table";
import { listSessionRecords } from "@/lib/history-store";

export default async function HistoryPage() {
  const sessions = await listSessionRecords();

  return (
    <AppShell title="Recent Sessions" subtitle="Review calibration, training, and Session history from local manifests">
      <section className="kh-panel p-5">
        <SessionTable sessions={sessions} />
      </section>
    </AppShell>
  );
}
