import { AlertTriangle, CheckCircle2, CircleDashed, Loader2, XCircle } from "lucide-react";
import type { TimelineStep } from "@/lib/types";

interface InferenceTimelineProps {
  timeline: TimelineStep[];
}

function iconForStatus(status: TimelineStep["status"]) {
  if (status === "complete") {
    return <CheckCircle2 className="h-5 w-5 text-emerald-600" />;
  }
  if (status === "warning") {
    return <AlertTriangle className="h-5 w-5 text-amber-600" />;
  }
  if (status === "failed") {
    return <XCircle className="h-5 w-5 text-rose-600" />;
  }
  if (status === "running") {
    return <Loader2 className="h-5 w-5 animate-spin text-sky-600" />;
  }
  return <CircleDashed className="h-5 w-5 text-slate-400" />;
}

export function InferenceTimeline({ timeline }: InferenceTimelineProps) {
  return (
    <section className="kh-panel p-5">
      <div className="mb-4">
        <p className="kh-kicker">Inference Timeline</p>
        <h3 className="mt-1 text-lg font-bold text-slate-900">Step-by-Step Execution</h3>
      </div>

      <div className="space-y-3">
        {timeline.map((step) => (
          <div key={step.id} className="grid gap-3 rounded-2xl border border-slate-200 bg-white px-4 py-3 md:grid-cols-[auto_minmax(0,1fr)]">
            <div className="pt-0.5">{iconForStatus(step.status)}</div>
            <div className="min-w-0">
              <div className="flex flex-wrap items-center gap-2">
                <p className="text-sm font-semibold text-slate-800">{step.label}</p>
                <span className="rounded-full border border-slate-200 bg-slate-50 px-2 py-0.5 text-[11px] font-bold uppercase tracking-[0.12em] text-slate-500">
                  {step.status}
                </span>
              </div>
              {step.detail ? <p className="mt-1 text-sm text-slate-600">{step.detail}</p> : null}
              {step.warnings.length > 0 ? (
                <div className="mt-2 space-y-1">
                  {step.warnings.map((warning) => (
                    <p key={warning} className="text-xs font-medium text-amber-700">
                      {warning}
                    </p>
                  ))}
                </div>
              ) : null}
            </div>
          </div>
        ))}
      </div>
    </section>
  );
}
