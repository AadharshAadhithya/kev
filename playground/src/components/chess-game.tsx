"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Chess, type Square } from "chess.js";
import Link from "next/link";
import { askModel, EVAL_LEVELS, loadGames, saveGames, resultText, type Mode, type ModelMove, type SavedGame } from "@/lib/chess";
import { api } from "@/lib/kev";
import { AnswerCard } from "@/components/answer-card";
import { ChessBoard } from "@/components/chess-board";
import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import { Switch } from "@/components/ui/switch";

const MODES: { value: Mode; label: string }[] = [
  { value: "self", label: "Model vs model" },
  { value: "white", label: "You play White" },
  { value: "black", label: "You play Black" },
];

function newGame(mode: Mode): SavedGame {
  return { id: `${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 6)}`, startedAt: Date.now(), mode, pgn: "", moves: [] };
}

function rebuild(g: SavedGame) {
  const c = new Chess();
  for (const m of g.moves) c.move(m.san);
  return c;
}

export function ChessGame() {
  const [games, setGames] = useState<SavedGame[]>([]);
  const [game, setGame] = useState<SavedGame | null>(null);
  const [mode, setMode] = useState<Mode>("self");
  const [sample, setSample] = useState(false);
  const [auto, setAuto] = useState(false);
  const [thinking, setThinking] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [selected, setSelected] = useState<Square | null>(null);
  const [model, setModel] = useState<string | null>(null);
  const autoRef = useRef(false);

  // load from localStorage (an external store) after mount; deferred so SSR and first client render match
  useEffect(() => {
    const t = setTimeout(() => {
      const saved = loadGames();
      setGames(saved);
      const last = saved.at(-1);
      if (last && !last.result) { setGame(last); setMode(last.mode); } else setGame(newGame("self"));
    }, 0);
    api.models().then((m) => setModel(m.models[0].run)).catch(() => setModel(null));
    return () => clearTimeout(t);
  }, []);

  const chess = useMemo(() => (game ? rebuild(game) : new Chess()), [game]);
  const lastMove = useMemo(() => { const h = chess.history({ verbose: true }); const m = h.at(-1); return m ? { from: m.from, to: m.to } : null; }, [chess]);
  const humanSide = mode === "white" ? "w" : mode === "black" ? "b" : null;
  const humanToMove = !!game && !game.result && humanSide === chess.turn();
  const modelToMove = !!game && !game.result && humanSide !== chess.turn();
  const lastModel = useMemo(() => [...(game?.moves ?? [])].reverse().find((m) => m.model)?.model, [game]);

  const persist = useCallback((g: SavedGame) => {
    setGame(g);
    setGames((prev) => { const next = [...prev.filter((x) => x.id !== g.id), g]; saveGames(next); return next; });
  }, []);

  const applyMove = useCallback((san: string, by: "human" | "model", info?: ModelMove) => {
    if (!game) return;
    const c = rebuild(game);
    c.move(san);
    persist({ ...game, moves: [...game.moves, { san, by, model: info }], pgn: c.pgn(), result: resultText(c) });
  }, [game, persist]);

  const modelMove = useCallback(async () => {
    if (!game || game.result || thinking) return;
    setThinking(true); setError(null);
    try {
      const c = rebuild(game);
      const info = await askModel(c, sample);
      c.move(info.san);
      applyMove(info.san, "model", info);
      if (c.isGameOver()) { setAuto(false); autoRef.current = false; }
    } catch (e) { setError((e as Error).message); setAuto(false); autoRef.current = false; }
    finally { setThinking(false); }
  }, [game, sample, thinking, applyMove]);

  // autoplay loop: after each state change, if it's the model's turn and auto is on, move again
  useEffect(() => { autoRef.current = auto; }, [auto]);
  useEffect(() => {
    if (!auto || !game || game.result || thinking) return;
    if (humanSide !== null && humanSide === chess.turn()) return;
    const t = setTimeout(() => { if (autoRef.current) modelMove(); }, 250);
    return () => clearTimeout(t);
  }, [auto, game, thinking, chess, humanSide, modelMove]);

  // in human-vs-model modes, the model replies automatically (and opens when you play Black)
  useEffect(() => {
    if (humanSide === null || !game || game.result || thinking || chess.turn() === humanSide) return;
    const last = game.moves.at(-1);
    if (game.moves.length === 0 || last?.by === "human") {
      const t = setTimeout(modelMove, 150);
      return () => clearTimeout(t);
    }
  }, [game, chess, humanSide, thinking, modelMove]);

  const targets = useMemo(() => new Set(selected ? chess.moves({ square: selected, verbose: true }).map((m) => m.to) : []), [chess, selected]);

  function onSquare(sq: Square) {
    if (!humanToMove || thinking) return;
    const piece = chess.get(sq);
    if (selected && targets.has(sq)) {
      const mv = chess.moves({ square: selected, verbose: true }).find((m) => m.to === sq);
      if (mv) applyMove(mv.san, "human");
      setSelected(null); return;
    }
    if (piece && piece.color === chess.turn()) setSelected(sq); else setSelected(null);
  }

  function start(m: Mode) {
    setAuto(false); autoRef.current = false; setMode(m); setSelected(null); setError(null);
    persist(newGame(m));
  }

  function undo() {
    if (!game || game.moves.length === 0) return;
    setAuto(false); autoRef.current = false;
    // in human modes undo the model reply too, so it is the human's turn again
    const n = humanSide && game.moves.at(-1)?.by === "model" ? 2 : 1;
    const moves = game.moves.slice(0, -n);
    const c = new Chess(); for (const m of moves) c.move(m.san);
    persist({ ...game, moves, pgn: c.pgn(), result: undefined });
  }

  const moveAnswer = lastModel && { type: "choice" as const, choice: lastModel.san, confidence: lastModel.confidence, probabilities: lastModel.probabilities };
  const evalAnswer = lastModel && { type: "score" as const, score: lastModel.evaluation, confidence: lastModel.evalConfidence ?? 0, legend: Object.fromEntries(EVAL_LEVELS.map((l, i) => [String(i), l])), probabilities: lastModel.evalProbabilities };
  const finished = games.filter((g) => g.result);

  return (
    <div className="mx-auto flex w-full max-w-6xl flex-col px-6 pt-8 pb-16 md:px-10">
      <header className="flex flex-wrap items-baseline justify-between gap-x-6 gap-y-1">
        <nav className="flex items-baseline gap-4 text-[15px]">
          <Link href="/" className="text-muted-foreground hover:text-foreground">kev</Link>
          <span className="font-medium tracking-tight">chess</span>
        </nav>
        <p className="text-[13px] text-muted-foreground">{model ? <span className="font-mono">{model}</span> : "connecting"}</p>
      </header>

      <div className="mt-10 max-w-2xl">
        <h1 className="text-2xl font-medium tracking-tight">Every move is a Choice question.</h1>
        <p className="mt-2 text-[15px] leading-6 text-muted-foreground">
          The legal moves are the options, the board is the state. The model returns a probability for each move and a Score for who is better, in one request. It is a 0.5B model that has never seen a chess game, so expect the distributions to be more interesting than the play.
        </p>
      </div>

      <div className="mt-8 grid gap-10 lg:grid-cols-[minmax(0,34rem)_minmax(0,1fr)]">
        <div className="flex flex-col gap-4">
          <div className="flex flex-wrap items-center gap-x-5 gap-y-2 text-sm">
            {MODES.map((m) => (
              <button key={m.value} type="button" onClick={() => start(m.value)} aria-current={mode === m.value ? "true" : undefined}
                className={`border-b pb-0.5 ${mode === m.value ? "border-foreground text-foreground" : "border-transparent text-muted-foreground hover:text-foreground"}`}>
                {m.label}
              </button>
            ))}
          </div>

          <ChessBoard chess={chess} selected={selected} targets={targets} lastMove={lastMove} flipped={mode === "black"} onSquare={onSquare} disabled={!humanToMove || thinking} />

          <div className="flex flex-wrap items-center gap-2">
            {mode === "self" ? (
              <>
                <Button onClick={() => setAuto((a) => !a)} disabled={!!game?.result} className="rounded-md">{auto ? "Pause" : "Play"}</Button>
                <Button variant="outline" onClick={modelMove} disabled={auto || thinking || !!game?.result} className="rounded-md shadow-none">Step</Button>
              </>
            ) : (
              <Button variant="outline" onClick={modelMove} disabled={!modelToMove || thinking} className="rounded-md shadow-none">Model moves</Button>
            )}
            <Button variant="ghost" onClick={undo} disabled={!game || game.moves.length === 0 || thinking} className="rounded-md text-muted-foreground">Undo</Button>
            <Button variant="ghost" onClick={() => start(mode)} className="rounded-md text-muted-foreground">New game</Button>
            <div className="ml-auto flex items-center gap-2">
              <Switch id="sample" checked={sample} onCheckedChange={(v) => setSample(!!v)} size="sm" />
              <Label htmlFor="sample" className="text-[13px] text-muted-foreground">Sample from distribution</Label>
            </div>
          </div>

          <p className="min-h-5 text-[13px] text-muted-foreground">
            {game?.result ? <span className="text-foreground">{game.result}</span>
              : thinking ? "Model is choosing"
              : humanToMove ? `Your move (${humanSide === "w" ? "White" : "Black"}). Click a piece, then a square.`
              : `${chess.turn() === "w" ? "White" : "Black"} to move`}
            {chess.inCheck() && !game?.result ? " · check" : ""}
          </p>
          {error && <pre className="whitespace-pre-wrap text-[13px] text-destructive">{error}</pre>}
        </div>

        <div className="flex min-w-0 flex-col gap-4">
          {moveAnswer && evalAnswer ? (
            <>
              <AnswerCard id="move" answer={moveAnswer} question={{ type: "choice", instructions: `Best move among ${lastModel.n_legal} legal moves`, criteria: {} }} />
              <AnswerCard id="evaluation" answer={evalAnswer} question={{ type: "score", instructions: "Who is better in this position?", criteria: EVAL_LEVELS }} />
              <p className="text-[13px] tabular-nums text-muted-foreground">{lastModel.latency_ms.toFixed(0)} ms · {lastModel.input_tokens} input tokens · {lastModel.n_legal} options</p>
            </>
          ) : (
            <p className="rounded-md border border-dashed border-border p-8 text-center text-sm text-muted-foreground">The model&apos;s move distribution appears here after its first move.</p>
          )}

          <div className="rounded-md border border-border bg-card px-4 py-3">
            <p className="text-[12px] text-muted-foreground">Moves</p>
            <ol className="mt-1 grid grid-cols-[2.5rem_1fr_1fr] gap-y-0.5 font-mono text-[13px] tabular-nums">
              {Array.from({ length: Math.ceil((game?.moves.length ?? 0) / 2) }, (_, i) => (
                <li key={i} className="contents">
                  <span className="text-muted-foreground">{i + 1}.</span>
                  <span>{game!.moves[2 * i]?.san}</span>
                  <span>{game!.moves[2 * i + 1]?.san ?? ""}</span>
                </li>
              ))}
            </ol>
            {game && game.moves.length === 0 && <p className="mt-1 text-[13px] text-muted-foreground">No moves yet.</p>}
          </div>

          {finished.length > 0 && (
            <div className="rounded-md border border-border bg-card px-4 py-3">
              <p className="text-[12px] text-muted-foreground">Previous games (stored in this browser)</p>
              <ul className="mt-1 flex flex-col text-[13px]">
                {[...finished].reverse().slice(0, 8).map((g) => (
                  <li key={g.id} className="flex items-baseline justify-between gap-3 py-0.5">
                    <span className="text-muted-foreground">{new Date(g.startedAt).toLocaleString()} · {MODES.find((m) => m.value === g.mode)?.label}</span>
                    <span className="tabular-nums">{g.moves.length} plies · {g.result}</span>
                  </li>
                ))}
              </ul>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
