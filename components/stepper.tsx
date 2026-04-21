interface StepperProps {
  steps: string[];
  current: number;
}

export function Stepper({ steps, current }: StepperProps) {
  return (
    <ol className="grid gap-2 md:grid-cols-4">
      {steps.map((step, index) => {
        const done = index <= current;
        return (
          <li
            key={step}
            className={`relative rounded-xl border px-3 py-3 text-sm transition ${
              done ? "border-teal-200 bg-teal-50 text-teal-800" : "border-slate-200 bg-white text-slate-500"
            }`}
          >
            <span className="mb-2 inline-flex h-6 w-6 items-center justify-center rounded-full bg-white text-xs font-extrabold text-slate-700 ring-1 ring-slate-200">
              {index + 1}
            </span>
            <p className="font-semibold">{step}</p>
          </li>
        );
      })}
    </ol>
  );
}
