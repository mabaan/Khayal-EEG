"use client";

import Image from "next/image";
import { Eye, EyeOff } from "lucide-react";
import { useRouter } from "next/navigation";
import { FormEvent, useEffect, useState } from "react";

export default function AuthPage() {
  const router = useRouter();

  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [showPassword, setShowPassword] = useState(false);
  const [pending, setPending] = useState(false);
  const [pendingAction, setPendingAction] = useState<"login" | "register" | null>(null);
  const [message, setMessage] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetch("/api/auth/session", { cache: "no-store" })
      .then((response) => response.json())
      .then((payload: { authenticated?: boolean }) => {
        if (payload.authenticated) {
          router.replace("/");
        }
      })
      .catch(() => {
        // keep page interactive even if session check fails
      });
  }, [router]);

  async function submit(action: "login" | "register") {
    setPending(true);
    setPendingAction(action);
    setError(null);
    setMessage(null);

    try {
      const response = await fetch(`/api/auth/${action}`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json"
        },
        body: JSON.stringify({ username, password })
      });

      const payload = (await response.json()) as { error?: string; username?: string };
      if (!response.ok) {
        throw new Error(payload.error ?? "Authentication request failed.");
      }

      const verb = action === "login" ? "Welcome back" : "Account created";
      setMessage(`${verb}, ${payload.username ?? username}. Redirecting...`);
      router.replace("/");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Authentication failed.");
    } finally {
      setPending(false);
      setPendingAction(null);
    }
  }

  async function onSubmit(event: FormEvent) {
    event.preventDefault();
    await submit("login");
  }

  return (
    <main className="flex min-h-screen items-center justify-center px-4 py-10">
      <section className="w-full max-w-[390px] rounded-2xl border border-slate-200 bg-white/95 px-6 py-7 shadow-[0_20px_45px_rgba(15,28,42,0.12)]">
        <div className="flex flex-col items-center">
          <div className="mb-3 rounded-xl border border-slate-200 bg-slate-50 p-3">
            <Image src="/logo.png" alt="Khayal logo" width={48} height={48} className="h-12 w-12 object-contain" priority />
          </div>
          <h1 className="text-4xl font-extrabold tracking-tight text-slate-900">Khayal</h1>
          <p className="mt-2 text-sm text-slate-600">Classify your thoughts into clear Arabic sentences.</p>
        </div>

        <form onSubmit={onSubmit} className="mt-7 space-y-4">
          <div>
            <label className="mb-1 block text-sm font-semibold text-slate-700">Username</label>
            <input
              value={username}
              onChange={(event) => setUsername(event.target.value)}
              placeholder="Enter your username"
              className="kh-input"
              autoComplete="username"
              required
            />
          </div>

          <div>
            <label className="mb-1 block text-sm font-semibold text-slate-700">Password</label>
            <div className="relative">
              <input
                type={showPassword ? "text" : "password"}
                value={password}
                onChange={(event) => setPassword(event.target.value)}
                placeholder="Enter your password"
                className="kh-input pr-11"
                autoComplete="current-password"
                required
              />
              <button
                type="button"
                onClick={() => setShowPassword((prev) => !prev)}
                className="absolute inset-y-0 right-2 my-auto inline-flex h-8 w-8 items-center justify-center rounded-md text-slate-500 hover:bg-slate-100"
                aria-label={showPassword ? "Hide password" : "Show password"}
              >
                {showPassword ? <EyeOff size={16} /> : <Eye size={16} />}
              </button>
            </div>
          </div>

          <div className="grid gap-2">
            <button
              type="submit"
              className="kh-btn w-full"
              disabled={pending}
            >
              {pending && pendingAction === "login" ? "Logging in..." : "Login"}
            </button>
            <button
              type="button"
              className="kh-btn kh-btn-ghost w-full"
              disabled={pending}
              onClick={async () => submit("register")}
            >
              {pending && pendingAction === "register" ? "Registering..." : "Register"}
            </button>
          </div>

          <div className="pt-1 text-center text-sm">
            <button
              type="button"
              className="text-sky-700 underline underline-offset-4 hover:text-sky-800"
              onClick={() => setError("Password reset is not enabled in local-only mode.")}
            >
              Forgot Password?
            </button>
          </div>

          {message ? <p className="text-sm font-medium text-emerald-700">{message}</p> : null}
          {error ? <p className="text-sm font-medium text-rose-700">{error}</p> : null}
        </form>

      </section>
    </main>
  );
}
