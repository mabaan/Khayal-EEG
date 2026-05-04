import type { InferenceResult, ProfileManifest } from "@/lib/types";

interface DebugDetailsPanelProps {
  profile: ProfileManifest | null;
  result: InferenceResult | null;
}

export function DebugDetailsPanel({ profile, result }: DebugDetailsPanelProps) {
  return (
    <details className="kh-panel overflow-hidden">
      <summary className="cursor-pointer list-none px-5 py-4 text-sm font-bold text-slate-800">
        Debug Details
      </summary>
      <div className="border-t border-slate-100 bg-slate-50 px-5 py-5">
        <div className="grid gap-4 lg:grid-cols-2">
          <div className="space-y-3 text-sm text-slate-700">
            <p className="break-all"><span className="font-semibold text-slate-900">Active profile:</span> {profile?.name ?? "-"}</p>
            <p className="break-all"><span className="font-semibold text-slate-900">Model path:</span> {result?.model.path ?? profile?.user_model_path ?? "-"}</p>
            <p className="break-all"><span className="font-semibold text-slate-900">EDF path:</span> {result?.edf.path ?? "-"}</p>
            <p className="break-all"><span className="font-semibold text-slate-900">Marker path:</span> {result?.edf.marker_csv ?? "-"}</p>
            <p><span className="font-semibold text-slate-900">Stage 1 device:</span> {result?.model.device ?? "-"}</p>
            <p><span className="font-semibold text-slate-900">Window size:</span> {result?.model.window_size ?? "-"}</p>
          </div>

          <div className="space-y-3 text-sm text-slate-700">
            <p><span className="font-semibold text-slate-900">Tensor shapes:</span> {(result?.preprocessing.slot_tensor_shapes ?? []).map((shape) => shape.join("x")).join(", ") || "-"}</p>
            <p><span className="font-semibold text-slate-900">Stage 2 mode:</span> {result?.stage2.mode ?? "-"}</p>
            <p><span className="font-semibold text-slate-900">Stage 2 device:</span> {result?.stage2.device ?? "-"}</p>
            <p><span className="font-semibold text-slate-900">Retrieval top-k:</span> {result?.stage2.retrieval_topk ?? "-"}</p>
            <p><span className="font-semibold text-slate-900">Local retrieval:</span> {result?.stage2.transformer_retrieval_used ? "Used" : "Fallback used"}</p>
            <p><span className="font-semibold text-slate-900">Total timing:</span> {result?.timing.total_ms ?? 0} ms</p>
          </div>
        </div>

        {result?.stage2.raw_llm_output ? (
          <div className="mt-4 rounded-2xl border border-slate-200 bg-white p-4">
            <p className="text-xs font-semibold uppercase tracking-[0.12em] text-slate-500">LLM Raw Output</p>
            <pre className="mt-2 whitespace-pre-wrap break-words text-xs text-slate-700">{result.stage2.raw_llm_output}</pre>
          </div>
        ) : null}
      </div>
    </details>
  );
}
