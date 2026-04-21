import type { SessionRecord } from "@/lib/types";
import { formatDate } from "@/lib/session-utils";

interface SessionTableProps {
  sessions: SessionRecord[];
}

function toneForStatus(status: SessionRecord["status"]): string {
  if (status === "success") {
    return "border-emerald-200 bg-emerald-50 text-emerald-700";
  }
  return "border-rose-200 bg-rose-50 text-rose-700";
}

export function SessionTable({ sessions }: SessionTableProps) {
  return (
    <div className="kh-panel overflow-hidden">
      <div className="overflow-x-auto">
        <table className="w-full min-w-[760px] text-left text-sm">
          <thead className="bg-slate-50 text-[11px] uppercase tracking-[0.13em] text-slate-500">
            <tr>
              <th className="px-4 py-3">Type</th>
              <th className="px-4 py-3">Created</th>
              <th className="px-4 py-3">Status</th>
              <th className="px-4 py-3">Signal Status</th>
              <th className="px-4 py-3">Predicted Sentence</th>
            </tr>
          </thead>
          <tbody>
            {sessions.length === 0 ? (
              <tr>
                <td colSpan={5} className="px-4 py-9 text-center text-slate-500">
                  No sessions yet.
                </td>
              </tr>
            ) : (
              sessions.map((session) => (
                <tr key={session.session_id} className="border-t border-slate-100">
                  <td className="px-4 py-3 font-semibold capitalize text-slate-700">{session.type}</td>
                  <td className="px-4 py-3 text-slate-600">{formatDate(session.created_at)}</td>
                  <td className="px-4 py-3">
                    <span className={`rounded-full border px-2 py-0.5 text-xs font-semibold ${toneForStatus(session.status)}`}>{session.status}</span>
                  </td>
                  <td className="px-4 py-3 text-slate-600">{session.signal_status}</td>
                  <td className="kh-ar px-4 py-3 text-slate-700" dir="rtl">
                    {session.predicted_sentence ?? "-"}
                  </td>
                </tr>
              ))
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
}
