export const dynamic = "force-dynamic";

import { promises as fs } from "node:fs";
import { AppShell } from "@/components/app-shell";
import { BASE_MODEL_DEFAULT_PATH, OLLAMA_BASE_URL, PYTHON_SERVICE_URL } from "@/lib/constants";
import { paths } from "@/lib/local-paths";

export default async function SettingsPage() {
  const [baseModelExists, profilesRootExists] = await Promise.all([
    fs.access(BASE_MODEL_DEFAULT_PATH).then(() => true).catch(() => false),
    fs.access(paths.profilesRoot).then(() => true).catch(() => false)
  ]);

  return (
    <AppShell title="Settings" subtitle="Local diagnostics for fixed-path runtime and offline service endpoints">
      <div className="grid gap-5 md:grid-cols-2">
        <section className="kh-panel p-5">
          <p className="kh-kicker">Local Endpoints</p>
          <div className="mt-3 space-y-2 text-sm text-slate-700">
            <p className="rounded-xl border border-slate-200 bg-slate-50 px-3 py-2">Python Service: {PYTHON_SERVICE_URL}</p>
            <p className="rounded-xl border border-slate-200 bg-slate-50 px-3 py-2">Ollama Stage 2: {OLLAMA_BASE_URL}</p>
          </div>
        </section>

        <section className="kh-panel p-5">
          <p className="kh-kicker">Storage Diagnostics</p>
          <div className="mt-3 space-y-2 text-sm text-slate-700">
            <p className="rounded-xl border border-slate-200 bg-slate-50 px-3 py-2">Base model default path: {BASE_MODEL_DEFAULT_PATH}</p>
            <p className="rounded-xl border border-slate-200 bg-slate-50 px-3 py-2">Base model found: {baseModelExists ? "Yes" : "No"}</p>
            <p className="rounded-xl border border-slate-200 bg-slate-50 px-3 py-2">Profiles folder found: {profilesRootExists ? "Yes" : "No"}</p>
          </div>
        </section>
      </div>
    </AppShell>
  );
}
