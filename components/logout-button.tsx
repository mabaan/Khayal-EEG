"use client";

import { LogOut } from "lucide-react";
import { useRouter } from "next/navigation";
import { useState } from "react";

export function LogoutButton() {
  const router = useRouter();
  const [pending, setPending] = useState(false);

  return (
    <button
      type="button"
      className="kh-chip hover:border-slate-300 hover:bg-white disabled:opacity-70"
      disabled={pending}
      onClick={async () => {
        setPending(true);
        try {
          await fetch("/api/auth/logout", { method: "POST" });
        } finally {
          router.replace("/auth");
          router.refresh();
          setPending(false);
        }
      }}
    >
      <LogOut size={13} />
      {pending ? "Logging Out..." : "Log Out"}
    </button>
  );
}
