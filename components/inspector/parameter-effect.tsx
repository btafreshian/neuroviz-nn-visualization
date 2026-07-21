import type { ReactNode } from "react"
import { ChevronDown } from "lucide-react"

interface ParameterEffectProps {
  summary: string
  metric?: string
  children: ReactNode
}

export function ParameterEffect({ summary, metric, children }: ParameterEffectProps) {
  return (
    <details className="group rounded-md bg-muted/35 px-2.5 py-2 text-xs text-muted-foreground">
      <summary className="flex cursor-pointer list-none items-start justify-between gap-2 outline-none focus-visible:ring-2 focus-visible:ring-ring/50">
        <span className="leading-relaxed">{summary}</span>
        <span className="flex shrink-0 items-center gap-1 font-mono text-[10px] text-foreground">
          {metric}
          <ChevronDown className="h-3 w-3 transition-transform group-open:rotate-180" aria-hidden="true" />
        </span>
      </summary>
      <div className="mt-2 border-t pt-2 leading-relaxed">{children}</div>
    </details>
  )
}
