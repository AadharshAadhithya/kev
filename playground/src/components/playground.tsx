"use client";

import { useEffect, useMemo, useState } from "react";
import { Loader2, Play, Shuffle, SplitSquareHorizontal } from "lucide-react";
import { api, PRESETS, type PermuteResponse, type Question, type SystemOneRequest, type SystemOneResponse } from "@/lib/kev";
import { AnswerCard } from "@/components/answer-card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Label } from "@/components/ui/label";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Textarea } from "@/components/ui/textarea";

const pretty = (v: unknown) => JSON.stringify(v, null, 2);

function parseState(s: string) {
  const t = s.trim();
  if (t.startsWith("{") || t.startsWith("[")) {
    try { return JSON.parse(t); } catch { /* fall through: treat as plain text */ }
  }
  return s;
}

export function Playground() {
  const [presetIdx, setPresetIdx] = useState(0);
  const [stateText, setStateText] = useState(() => typeof PRESETS[0].state === "string" ? PRESETS[0].state : pretty(PRESETS[0].state));
  const [questionsText, setQuestionsText] = useState(() => pretty(PRESETS[0].questions));
  const [result, setResult] = useState<SystemOneResponse | null>(null);
  const [separate, setSeparate] = useState<(SystemOneResponse & { latency_ms: number }) | null>(null);
  const [permute, setPermute] = useState<{ question: string; data: PermuteResponse } | null>(null);
  const [busy, setBusy] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [model, setModel] = useState<{ run: string; base: string } | { error: string } | null>(null);
  const [showRaw, setShowRaw] = useState(false);

  useEffect(() => {
    api.models().then((m) => setModel(m.models[0])).catch((e: Error) => setModel({ error: e.message }));
  }, []);

  const parsed = useMemo<{ req?: SystemOneRequest; err?: string }>(() => {
    try {
      const questions = JSON.parse(questionsText) as Record<string, Question>;
      return { req: { state: parseState(stateText), model: "kev-latest", questions } };
    } catch (e) { return { err: (e as Error).message }; }
  }, [stateText, questionsText]);

  const choiceIds = useMemo(() => (parsed.req ? Object.entries(parsed.req.questions).filter(([, q]) => q.type === "choice" && Object.keys(q.criteria).length >= 2).map(([id]) => id) : []), [parsed.req]);

  function loadPreset(i: number) {
    const p = PRESETS[i];
    setPresetIdx(i);
    setStateText(typeof p.state === "string" ? p.state : pretty(p.state));
    setQuestionsText(pretty(p.questions));
    setResult(null); setSeparate(null); setPermute(null); setError(null);
  }

  async function run<T>(label: string, fn: () => Promise<T>, done: (t: T) => void) {
    if (!parsed.req) return;
    setBusy(label); setError(null);
    try { done(await fn()); } catch (e) { setError((e as Error).message); } finally { setBusy(null); }
  }

  const onRun = () => run("run", () => api.systemOne(parsed.req!), (r) => { setResult(r); setSeparate(null); setPermute(null); });

  // Cmd/Ctrl+Enter runs from anywhere on the page, including inside the textareas.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) { e.preventDefault(); onRun(); }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  });
  const onSeparate = () => run("separate", async () => ({ packed: await api.systemOne(parsed.req!), sep: await api.separate(parsed.req!) }), ({ packed, sep }) => { setResult(packed); setSeparate(sep); });
  const onPermute = (qid: string) => run("permute", () => api.permute(parsed.req!, qid, 6), (d) => setPermute({ question: qid, data: d }));

  const maxDiff = useMemo(() => {
    if (!result || !separate) return null;
    let m = 0;
    for (const [id, a] of Object.entries(result.answers)) {
      const b = separate.answers[id];
      if (!b) continue;
      if (a.type === "noul" && b.type === "noul") m = Math.max(m, Math.abs(a.noul - b.noul));
      else if ("probabilities" in a && "probabilities" in b) for (const k of Object.keys(a.probabilities)) m = Math.max(m, Math.abs(a.probabilities[k] - (b.probabilities[k] ?? 0)));
    }
    return m;
  }, [result, separate]);

  return (
    <div className="mx-auto flex w-full max-w-7xl flex-col gap-4 p-4 md:p-6">
      <header className="flex flex-wrap items-baseline gap-x-3 gap-y-1">
        <h1 className="text-xl font-semibold tracking-tight">kev playground</h1>
        <span className="text-sm text-muted-foreground">a Jev-style decision model: shared state, isolated questions, probabilities read out in one forward pass</span>
        <span className="ml-auto font-mono text-xs text-muted-foreground">
          {model === null ? "connecting…" : "error" in model ? `backend error: ${model.error}` : `${model.base} · ${model.run}`}
        </span>
      </header>

      <div className="flex flex-wrap gap-1.5">
        {PRESETS.map((p, i) => (
          <Button key={p.name} size="sm" variant={i === presetIdx ? "default" : "outline"} onClick={() => loadPreset(i)}>{p.name}</Button>
        ))}
      </div>
      <p className="-mt-2 text-xs text-muted-foreground">{PRESETS[presetIdx].blurb}</p>

      <div className="grid gap-4 lg:grid-cols-[minmax(0,5fr)_minmax(0,6fr)]">
        <div className="flex flex-col gap-3">
          <div className="flex flex-col gap-1.5">
            <Label htmlFor="state">state <span className="font-normal text-muted-foreground">(text, or a JSON object / array)</span></Label>
            <Textarea id="state" value={stateText} onChange={(e) => setStateText(e.target.value)} className="min-h-28 font-mono text-xs" />
          </div>
          <div className="flex flex-col gap-1.5">
            <Label htmlFor="questions">questions <span className="font-normal text-muted-foreground">(TypeSafe schema: noul · choice · score)</span></Label>
            <Textarea id="questions" value={questionsText} onChange={(e) => setQuestionsText(e.target.value)} className="min-h-[26rem] font-mono text-xs" aria-invalid={!!parsed.err} />
            {parsed.err && <span className="text-xs text-destructive">{parsed.err}</span>}
          </div>
        </div>

        <div className="flex flex-col gap-3">
          <Tabs defaultValue="answers">
            <div className="flex items-center gap-2">
              <TabsList>
                <TabsTrigger value="answers">Answers</TabsTrigger>
                <TabsTrigger value="permute" disabled={!permute}>Permutation</TabsTrigger>
                <TabsTrigger value="raw" disabled={!result}>Raw JSON</TabsTrigger>
              </TabsList>
              {result && (
                <div className="ml-auto flex flex-wrap items-center gap-1.5 font-mono text-xs text-muted-foreground">
                  <Badge variant="outline">{result.latency_ms?.toFixed(0)} ms</Badge>
                  <Badge variant="outline">{result.usage.input_tokens} in</Badge>
                  <Badge variant="outline">{Object.keys(result.answers).length} q</Badge>
                </div>
              )}
            </div>

            <TabsContent value="answers">
              {!result && <div className="rounded-lg border border-dashed p-8 text-center text-sm text-muted-foreground">Run a request to see the probability distributions.</div>}
              {result && separate && (
                <Card size="sm" className="mb-3">
                  <CardHeader>
                    <CardTitle className="text-sm">Packed vs separate</CardTitle>
                    <CardDescription>
                      One request with {Object.keys(result.answers).length} questions took <b>{result.latency_ms?.toFixed(0)} ms</b> ({result.usage.input_tokens} tokens);
                      {" "}{Object.keys(result.answers).length} separate requests took <b>{separate.latency_ms.toFixed(0)} ms</b> ({separate.usage.input_tokens} tokens).
                      Max probability difference between the two: <b className="font-mono">{maxDiff?.toFixed(4)}</b>
                      {maxDiff !== null && maxDiff < 0.011 ? " — questions are isolated; siblings do not change an answer." : " — larger than rounding; check the request."}
                    </CardDescription>
                  </CardHeader>
                </Card>
              )}
              <div className="grid gap-3 md:grid-cols-2">
                {result && Object.entries(result.answers).map(([id, a]) => (
                  <AnswerCard key={id} id={id} question={parsed.req?.questions[id]} answer={a} compare={separate?.answers[id]} />
                ))}
              </div>
            </TabsContent>

            <TabsContent value="permute">
              {permute && (
                <Card size="sm">
                  <CardHeader>
                    <CardTitle className="flex items-center gap-2 text-sm">
                      <span className="font-mono">{permute.question}</span> under {permute.data.runs.length} option orders
                      <Badge variant={permute.data.argmax_stable ? "default" : "destructive"} className="ml-auto">{permute.data.argmax_stable ? "argmax stable" : "argmax flips"}</Badge>
                    </CardTitle>
                    <CardDescription>Each row is the same question with its options in a different order. Spread = max − min probability per option across orders.</CardDescription>
                  </CardHeader>
                  <CardContent>
                    <div className="overflow-x-auto">
                      <table className="w-full text-xs">
                        <thead>
                          <tr className="text-left text-muted-foreground">
                            <th className="py-1 pr-3 font-medium">order</th>
                            {Object.keys(permute.data.spread).map((k) => <th key={k} className="py-1 pr-3 font-mono font-medium">{k}</th>)}
                          </tr>
                        </thead>
                        <tbody className="font-mono tabular-nums">
                          {permute.data.runs.map((r, i) => (
                            <tr key={i} className="border-t">
                              <td className="py-1 pr-3 text-muted-foreground" title={r.order.join(" → ")}>{r.order.map((o) => o.slice(0, 3)).join("·")}</td>
                              {Object.keys(permute.data.spread).map((k) => (
                                <td key={k} className={`py-1 pr-3 ${r.choice === k ? "font-semibold" : ""}`}>{r.probabilities[k].toFixed(2)}</td>
                              ))}
                            </tr>
                          ))}
                          <tr className="border-t text-muted-foreground">
                            <td className="py-1 pr-3">spread</td>
                            {Object.entries(permute.data.spread).map(([k, s]) => <td key={k} className={`py-1 pr-3 ${s > 0.1 ? "text-destructive" : ""}`}>{s.toFixed(2)}</td>)}
                          </tr>
                        </tbody>
                      </table>
                    </div>
                  </CardContent>
                </Card>
              )}
            </TabsContent>

            <TabsContent value="raw">
              {result && (
                <div className="flex flex-col gap-2">
                  <div className="flex items-center gap-2 text-xs">
                    <Button size="xs" variant={showRaw ? "outline" : "default"} onClick={() => setShowRaw(false)}>response</Button>
                    <Button size="xs" variant={showRaw ? "default" : "outline"} onClick={() => setShowRaw(true)}>request</Button>
                  </div>
                  <pre className="max-h-[36rem] overflow-auto rounded-md bg-muted p-3 font-mono text-xs">{pretty(showRaw ? parsed.req : result)}</pre>
                </div>
              )}
            </TabsContent>
          </Tabs>
        </div>
      </div>

      <div className="sticky bottom-0 z-10 -mx-4 mt-2 border-t bg-background/90 px-4 py-3 backdrop-blur md:-mx-6 md:px-6">
        <div className="mx-auto flex w-full max-w-7xl flex-col gap-2">
          <div className="flex flex-wrap items-center gap-2">
            <Button onClick={onRun} disabled={!parsed.req || !!busy}>
              {busy === "run" ? <Loader2 className="animate-spin" /> : <Play />} Run
              <kbd className="ml-1 rounded border border-primary-foreground/30 px-1 font-mono text-[10px] opacity-80">⌘↵</kbd>
            </Button>
            <Button variant="outline" onClick={onSeparate} disabled={!parsed.req || !!busy} title="Answer every question in its own request, then compare with the packed answer">
              {busy === "separate" ? <Loader2 className="animate-spin" /> : <SplitSquareHorizontal />} Packed vs separate
            </Button>
            {choiceIds.map((id) => (
              <Button key={id} variant="outline" size="sm" onClick={() => onPermute(id)} disabled={!!busy} title={`Re-ask "${id}" under 6 option orders`}>
                {busy === "permute" && permute?.question === id ? <Loader2 className="animate-spin" /> : <Shuffle />} Permute {id}
              </Button>
            ))}
          </div>
          {error && <pre className="whitespace-pre-wrap rounded-md bg-destructive/10 p-2 text-xs text-destructive">{error}</pre>}
        </div>
      </div>
    </div>
  );
}
