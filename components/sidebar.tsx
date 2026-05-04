"use client";

import Image from "next/image";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { BookOpen, Home, History, Settings, UserPlus, Waves } from "lucide-react";
import { clsx } from "clsx";

const items = [
  { href: "/", label: "Home", icon: Home },
  { href: "/setup", label: "Setup", icon: UserPlus },
  { href: "/calibration", label: "Catalog", icon: BookOpen },
  { href: "/inference", label: "Session", icon: Waves },
  { href: "/history", label: "Recent Sessions", icon: History },
  { href: "/settings", label: "Settings", icon: Settings }
];

export function Sidebar() {
  const pathname = usePathname();

  return (
    <aside className="border-b border-slate-200/80 bg-white/75 p-4 backdrop-blur md:sticky md:top-0 md:h-screen md:border-b-0 md:border-r md:p-5">
      <div className="rounded-2xl bg-[linear-gradient(120deg,#0f766e,#0ea5c6)] px-4 py-5 text-white shadow-[0_20px_35px_rgba(15,118,110,0.32)]">
        <div className="mb-3 inline-flex items-center rounded-xl bg-white/15 px-2 py-2">
          <div className="rounded-lg bg-white p-1.5">
            <Image src="/logo.png" alt="Khayal logo" width={34} height={34} className="h-8 w-8 object-contain" priority />
          </div>
        </div>
        <h2 className="mt-2 text-xl font-extrabold tracking-tight">Imagined Speech</h2>
        <p className="mt-2 text-xs leading-relaxed text-white/90">Classify your thoughts into clear Arabic sentences.</p>
      </div>

      <div className="mt-6">
        <p className="kh-kicker mb-2 pl-1">Workflow</p>
        <nav className="grid grid-cols-2 gap-2 md:grid-cols-1">
          {items.map((item) => {
            const Icon = item.icon;
            const active = pathname === item.href;
            return (
              <Link
                key={item.href}
                href={item.href}
                className={clsx(
                  "group relative flex items-center gap-3 rounded-xl border px-3 py-2.5 text-sm font-semibold transition",
                  active
                    ? "border-teal-200 bg-teal-50 text-teal-800"
                    : "border-transparent bg-white/70 text-slate-600 hover:border-slate-200 hover:bg-white"
                )}
              >
                <Icon size={16} className={clsx(active ? "text-teal-700" : "text-slate-500 group-hover:text-slate-700")} />
                <span className="truncate">{item.label}</span>
                {active ? <span className="absolute inset-y-2 right-2 w-1 rounded-full bg-teal-500" /> : null}
              </Link>
            );
          })}
        </nav>
      </div>
    </aside>
  );
}
