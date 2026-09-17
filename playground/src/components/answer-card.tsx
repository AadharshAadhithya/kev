import type { Answer, Question } from "@/lib/kev";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";

function Bar({ label, p, highlight, delta }: { label: string; p: number; highlight?: boolean; delta?: number }) {
  return (
    <div className="grid grid-cols-[minmax(0,1fr)_3rem] items-center gap-2 text-xs">
      <div className="flex items-center gap-2 min-w-0">
        <span className={`truncate ${highlight ? "font-semibold" : "text-muted-foreground"}`} title={label}>{label}</span>
        <div className="h-2 flex-1 rounded bg-muted overflow-hidden">
          <div className={`h-full ${highlight ? "bg-foreground" : "bg-foreground/40"}`} style={{ width: `${Math.round(p * 100)}%` }} />
        </div>
      </div>
      <span className="text-right font-mono tabular-nums">
        {p.toFixed(2)}
        {delta !== undefined && Math.abs(delta) >= 0.01 && <span className="ml-1 text-[10px] text-muted-foreground">{delta > 0 ? "+" : ""}{delta.toFixed(2)}</span>}
      </span>
    </div>
  );
}

function confidenceTone(c: number) {
  return c >= 0.7 ? "default" : c >= 0.4 ? "secondary" : "destructive";
}

export function AnswerCard({ id, question, answer, compare }: { id: string; question?: Question; answer: Answer; compare?: Answer }) {
  const instr = question ? (typeof question.instructions === "string" ? question.instructions : JSON.stringify(question.instructions)) : "";
  return (
    <Card size="sm">
      <CardHeader>
        <CardTitle className="flex items-center gap-2 font-mono text-sm">
          {id}
          <Badge variant="outline" className="font-sans uppercase text-[10px]">{answer.type}</Badge>
          {"confidence" in answer && (
            <Badge variant={confidenceTone(answer.confidence)} className="ml-auto font-mono">conf {answer.confidence.toFixed(2)}</Badge>
          )}
        </CardTitle>
        {instr && <CardDescription className="truncate" title={instr}>{instr}</CardDescription>}
      </CardHeader>
      <CardContent className="flex flex-col gap-1.5">
        {answer.type === "noul" && (
          <>
            <Bar label="yes" p={answer.noul} highlight={answer.noul >= 0.5} delta={compare?.type === "noul" ? answer.noul - compare.noul : undefined} />
            <Bar label="no" p={1 - answer.noul} highlight={answer.noul < 0.5} />
          </>
        )}
        {answer.type === "choice" &&
          Object.entries(answer.probabilities)
            .sort((a, b) => b[1] - a[1])
            .map(([k, p]) => (
              <Bar key={k} label={k} p={p} highlight={k === answer.choice} delta={compare?.type === "choice" ? p - (compare.probabilities[k] ?? 0) : undefined} />
            ))}
        {answer.type === "score" && (
          <>
            <div className="text-xs text-muted-foreground">
              expected level <span className="font-mono font-semibold text-foreground">{answer.score.toFixed(2)}</span> of {Object.keys(answer.legend).length - 1}
            </div>
            {Object.entries(answer.probabilities).map(([k, p]) => (
              <Bar key={k} label={`${k} · ${answer.legend[k]}`} p={p} highlight={Number(k) === Math.round(answer.score)} delta={compare?.type === "score" ? p - (compare.probabilities[k] ?? 0) : undefined} />
            ))}
          </>
        )}
      </CardContent>
    </Card>
  );
}
