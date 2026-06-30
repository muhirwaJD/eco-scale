import { Clock, ChevronDown } from "lucide-react";
import { cn } from "@/lib/utils";

const RANGES = ["5m", "15m", "1h", "6h", "24h", "7d"];

export function TimeRangePicker({
  value,
  onChange,
  refreshSec,
  onRefreshChange,
}: {
  value: string;
  onChange: (v: string) => void;
  refreshSec: number;
  onRefreshChange: (v: number) => void;
}) {
  return (
    <div className="flex flex-wrap items-center gap-2">
      <div className="inline-flex items-center gap-1 rounded-lg border border-border bg-card p-1">
        <Clock className="ml-1.5 h-3.5 w-3.5 text-muted-foreground" />
        {RANGES.map((r) => (
          <button
            key={r}
            onClick={() => onChange(r)}
            className={cn(
              "rounded-md px-2 py-1 text-xs font-medium",
              value === r ? "bg-secondary text-foreground" : "text-muted-foreground hover:text-foreground",
            )}
          >
            {r}
          </button>
        ))}
      </div>
      <label className="inline-flex items-center gap-1.5 rounded-lg border border-border bg-card px-2.5 py-1.5 text-xs text-muted-foreground">
        Refresh
        <select
          value={refreshSec}
          onChange={(e) => onRefreshChange(parseInt(e.target.value))}
          className="bg-transparent text-foreground focus:outline-none"
        >
          {[0, 5, 15, 30, 60].map((s) => (
            <option key={s} value={s} className="bg-card">
              {s === 0 ? "off" : `${s}s`}
            </option>
          ))}
        </select>
        <ChevronDown className="h-3 w-3" />
      </label>
    </div>
  );
}

export function VariableBar({
  vars,
  onChange,
}: {
  vars: { key: string; value: string; options: string[] }[];
  onChange: (key: string, value: string) => void;
}) {
  return (
    <div className="flex flex-wrap items-center gap-2">
      {vars.map((v) => (
        <label
          key={v.key}
          className="inline-flex items-center gap-1.5 rounded-lg border border-border bg-card px-2.5 py-1.5 text-xs"
        >
          <span className="text-muted-foreground">${v.key}</span>
          <select
            value={v.value}
            onChange={(e) => onChange(v.key, e.target.value)}
            className="bg-transparent font-medium text-foreground focus:outline-none"
          >
            {v.options.map((o) => (
              <option key={o} value={o} className="bg-card">
                {o}
              </option>
            ))}
          </select>
        </label>
      ))}
    </div>
  );
}
