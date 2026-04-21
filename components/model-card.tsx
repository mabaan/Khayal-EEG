import type { ProfileManifest } from "@/lib/types";
import { StatusBadge } from "@/components/status-badge";

interface ModelCardProps {
  profile: ProfileManifest | null;
}

export function ModelCard({ profile }: ModelCardProps) {
  if (!profile) {
    return (
      <div className="kh-panel p-5">
        <p className="kh-kicker">Model Ready</p>
        <h3 className="mt-2 text-lg font-bold text-slate-900">No active profile</h3>
        <p className="mt-2 text-sm text-slate-600">Create or select a profile in Setup before running calibration and training.</p>
      </div>
    );
  }

  return (
    <div className="kh-panel p-5">
      <div className="flex items-start justify-between gap-3">
        <div>
          <p className="kh-kicker">Model Ready</p>
          <h3 className="mt-2 text-lg font-bold text-slate-900">Diff-E Personalization</h3>
        </div>
        <StatusBadge label={profile.model_ready ? "Ready" : "Needs Training"} tone={profile.model_ready ? "good" : "warn"} />
      </div>

      <dl className="mt-4 space-y-3 text-xs text-slate-600">
        <div>
          <dt className="font-semibold text-slate-500">Base Model Path</dt>
          <dd className="mt-1 break-all rounded-lg border border-slate-200 bg-slate-50 px-2 py-1.5">{profile.base_model_path}</dd>
        </div>
        <div>
          <dt className="font-semibold text-slate-500">User Checkpoint</dt>
          <dd className="mt-1 break-all rounded-lg border border-slate-200 bg-slate-50 px-2 py-1.5">{profile.user_model_path ?? "Not trained yet"}</dd>
        </div>
      </dl>
    </div>
  );
}
