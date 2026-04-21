"use client";

import { FormEvent, useEffect, useState } from "react";
import { AppShell } from "@/components/app-shell";
import { createProfile, fetchProfiles, selectProfile, updateBaseModelPath } from "@/lib/api-client";
import type { ProfileManifest } from "@/lib/types";
import { BASE_MODEL_DEFAULT_PATH } from "@/lib/constants";

export default function SetupPage() {
  const [profiles, setProfiles] = useState<ProfileManifest[]>([]);
  const [activeProfile, setActiveProfile] = useState<string | null>(null);
  const [name, setName] = useState("");
  const [basePath, setBasePath] = useState(BASE_MODEL_DEFAULT_PATH);
  const [message, setMessage] = useState("Create a profile to begin setup.");
  const [error, setError] = useState<string | null>(null);

  async function refresh() {
    const payload = await fetchProfiles();
    setProfiles(payload.profiles);
    setActiveProfile(payload.active_profile);
  }

  useEffect(() => {
    refresh().catch((err) => setError(err instanceof Error ? err.message : "Failed to load profiles."));
  }, []);

  async function onCreate(event: FormEvent) {
    event.preventDefault();
    setError(null);
    try {
      const result = await createProfile(name);
      setMessage(`Profile created: ${result.profile.name}`);
      setName("");
      await refresh();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Profile creation failed.");
    }
  }

  async function onSelect(profileId: string) {
    setError(null);
    try {
      await selectProfile(profileId);
      setMessage("Active profile updated.");
      await refresh();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to select profile.");
    }
  }

  async function onSaveBaseModel() {
    setError(null);
    try {
      await updateBaseModelPath(basePath);
      setMessage("Base model path saved to active profile.");
      await refresh();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to save model path.");
    }
  }

  return (
    <AppShell title="Setup" subtitle="Create or select a local profile, then lock the base Diff-E checkpoint path">
      <section className="kh-panel-strong mb-5 p-5">
        <p className="kh-kicker">First-Time Flow</p>
        <h2 className="mt-1 text-lg font-bold text-slate-900">Profile + Base Model</h2>
        <p className="mt-2 text-sm text-slate-600">
          Each user must have one personalized Stage 1 checkpoint. Setup defines the active local profile before calibration and training.
        </p>
      </section>

      <div className="grid gap-5 xl:grid-cols-2">
        <form onSubmit={onCreate} className="kh-panel p-5">
          <p className="kh-kicker">Create Profile</p>
          <label className="mt-3 block text-xs font-semibold uppercase tracking-[0.12em] text-slate-500" htmlFor="profile-name">
            Profile Name
          </label>
          <input
            id="profile-name"
            value={name}
            onChange={(event) => setName(event.target.value)}
            placeholder="e.g., Patient A"
            className="kh-input mt-2"
          />
          <button className="kh-btn mt-3" type="submit">
            Create Profile
          </button>
        </form>

        <section className="kh-panel p-5">
          <p className="kh-kicker">Select Active Profile</p>
          <p className="mt-2 rounded-xl border border-slate-200 bg-slate-50 px-3 py-2 text-sm text-slate-700">{activeProfile ?? "No profile selected"}</p>

          <div className="mt-3 grid gap-2">
            {profiles.length === 0 ? (
              <p className="text-sm text-slate-500">No profiles yet.</p>
            ) : (
              profiles.map((profile) => (
                <button
                  key={profile.id}
                  type="button"
                  onClick={() => onSelect(profile.id)}
                  className="rounded-xl border border-slate-200 bg-white px-3 py-2 text-left text-sm font-semibold text-slate-700 transition hover:border-teal-200"
                >
                  {profile.name}
                </button>
              ))
            )}
          </div>
        </section>
      </div>

      <section className="kh-panel mt-5 p-5">
        <p className="kh-kicker">Base Diff-E Path</p>
        <input value={basePath} onChange={(event) => setBasePath(event.target.value)} className="kh-input mt-2" />
        <button type="button" onClick={onSaveBaseModel} className="kh-btn kh-btn-secondary mt-3">
          Save Base Path
        </button>
        <p className="mt-3 text-sm text-slate-700">{message}</p>
        {error ? <p className="mt-1 text-sm font-semibold text-rose-600">{error}</p> : null}
      </section>
    </AppShell>
  );
}
