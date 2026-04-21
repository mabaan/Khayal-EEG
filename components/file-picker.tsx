"use client";

interface FilePickerProps {
  label: string;
  accept: string;
  onSelect: (file: File | null) => void;
}

export function FilePicker({ label, accept, onSelect }: FilePickerProps) {
  return (
    <label className="group flex cursor-pointer flex-col gap-2 rounded-2xl border border-dashed border-teal-200 bg-teal-50/40 p-4 text-sm transition hover:border-teal-300 hover:bg-teal-50">
      <span className="font-semibold text-slate-700">{label}</span>
      <input
        type="file"
        accept={accept}
        onChange={(event) => onSelect(event.target.files?.[0] ?? null)}
        className="rounded-lg border border-slate-200 bg-white px-2 py-1.5 text-sm text-slate-600"
      />
      <span className="text-xs text-slate-500">Accepted file type: {accept}</span>
    </label>
  );
}
