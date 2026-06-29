import { Info } from "lucide-react";

interface Help {
  title: string;
  what: string;
  tradeoff: string;
  use: string;
}

// beginner-friendly explanation for each HPA target
const HELP: Record<string, Help> = {
  "0.5": {
    title: "Conservative — 50%",
    what: "Adds pods so each one runs only about half-busy.",
    tradeoff: "Lots of spare capacity, so it rarely gets overloaded — but it runs more pods, which costs more energy.",
    use: "The common default teams pick when unsure: safety first.",
  },
  "0.6": {
    title: "Moderate — 60%",
    what: "Keeps each pod around 60% busy.",
    tradeoff: "A bit leaner than 50% while still keeping a healthy buffer.",
    use: "A middle ground between saving and staying safe.",
  },
  "0.7": {
    title: "Tuned — 70%",
    what: "Keeps each pod around 70% busy.",
    tradeoff: "Lean but still safe — the balanced 'sweet spot', though it takes tuning to find.",
    use: "What a careful operator tunes toward. The RL agent lands near here on its own, with no tuning.",
  },
  "0.9": {
    title: "Aggressive — 90%",
    what: "Packs each pod up to about 90% busy.",
    tradeoff: "Uses very few pods (cheapest), but has almost no buffer — so it breaks the SLA when traffic spikes suddenly.",
    use: "Only when cutting cost matters more than reliability.",
  },
};

export default function HpaHelp({ target }: { target: number }) {
  const info = HELP[String(target)] ?? HELP["0.5"];
  // 0.5 -> 0% (safer / more pods), 0.9 -> 100% (cheaper / riskier)
  const pos = Math.max(0, Math.min(100, ((target - 0.5) / 0.4) * 100));

  return (
    <div className="card p-5">
      <div className="mb-3 flex items-center gap-2">
        <Info size={15} className="text-eco-light" />
        <h2 className="text-sm font-semibold text-slate-100">
          What is "HPA @ {Math.round(target * 100)}%"?
        </h2>
      </div>

      <p className="mb-3 text-xs leading-relaxed text-slate-400">
        HPA (the standard Kubernetes autoscaler) keeps each pod at a fixed{" "}
        <span className="font-medium text-slate-200">CPU target</span>. The number is how
        busy it tries to keep every pod before adding more.
      </p>

      <div className="mb-1 text-sm font-semibold text-eco-light">{info.title}</div>
      <dl className="space-y-2 text-xs">
        <Line k="What it does" v={info.what} />
        <Line k="Trade-off" v={info.tradeoff} />
        <Line k="When to use" v={info.use} />
      </dl>

      {/* safer ↔ cheaper spectrum with a marker at the current target */}
      <div className="mt-4">
        <div className="relative h-2 rounded-full bg-gradient-to-r from-eco-green via-eco-amber to-eco-red">
          <div
            className="absolute -top-1 h-4 w-4 -translate-x-1/2 rounded-full border-2 border-[#0a0e14] bg-white shadow"
            style={{ left: `${pos}%` }}
          />
        </div>
        <div className="mt-1.5 flex justify-between text-[10px] text-slate-500">
          <span>safer · more pods</span>
          <span>cheaper · riskier</span>
        </div>
      </div>
    </div>
  );
}

function Line({ k, v }: { k: string; v: string }) {
  return (
    <div>
      <dt className="label">{k}</dt>
      <dd className="mt-0.5 leading-relaxed text-slate-300">{v}</dd>
    </div>
  );
}
