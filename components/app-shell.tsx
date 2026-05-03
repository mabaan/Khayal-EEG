import { ReactNode } from "react";
import Image from "next/image";
import { Sidebar } from "@/components/sidebar";
import { LogoutButton } from "@/components/logout-button";

interface AppShellProps {
  title: string;
  subtitle?: string;
  children: ReactNode;
}

export function AppShell({ title, subtitle, children }: AppShellProps) {
  return (
    <div className="min-h-screen text-slate-900 kh-page-enter">
      <div className="mx-auto grid min-h-screen max-w-[1500px] md:grid-cols-[280px_minmax(0,1fr)]">
        <Sidebar />

        <main className="min-w-0 px-4 pb-10 pt-4 md:px-8 md:pb-12 md:pt-7">
          <header className="kh-panel-strong mb-6 flex flex-wrap items-start justify-between gap-4 px-5 py-4 md:px-6 md:py-5">
            <div className="min-w-0">
              <div className="flex items-center gap-2">
                <Image src="/logo.png" alt="Khayal logo" width={24} height={24} className="h-6 w-6 object-contain" priority />
                <p className="kh-kicker">Khayal Local</p>
              </div>
              <h1 className="kh-heading mt-1 truncate">{title}</h1>
              {subtitle ? <p className="kh-subtext mt-2 max-w-3xl">{subtitle}</p> : null}
            </div>
            <div className="flex items-center gap-2">
              <span className="kh-chip">v1 Fixed Pipeline</span>
              <LogoutButton />
            </div>
          </header>

          <div className="kh-stagger">{children}</div>
        </main>
      </div>
    </div>
  );
}
