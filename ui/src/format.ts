import type { Answer, UsageRow } from './api';

export function pct(p: number): string {
  return `${(p * 100).toFixed(p >= 0.995 || p < 0.005 ? 0 : 1)}%`;
}

/** The answer as one line of text: "card_arrival", "yes (72%)", "3.4 / 4". */
export function headline(a: Answer): string {
  if (a.type === 'choice') return a.choice ?? '';
  if (a.type === 'noul') return `${(a.noul ?? 0) >= 0.5 ? 'yes' : 'no'} (${pct(a.noul ?? 0)} yes)`;
  const levels = Object.keys(a.probabilities ?? {}).length;
  return `${(a.score ?? 0).toFixed(2)} / ${levels - 1}`;
}

/** Top n options by probability, with score levels labelled by their legend. */
export function topOptions(a: Answer, n = 8): { label: string; p: number }[] {
  const probs = a.type === 'noul' ? { no: 1 - (a.noul ?? 0), yes: a.noul ?? 0 } : (a.probabilities ?? {});
  return Object.entries(probs)
    .map(([k, p]) => ({ label: a.legend?.[k] ? `${k} · ${a.legend[k]}` : k, p }))
    .sort((x, y) => y.p - x.p)
    .slice(0, n);
}

/** Usage rows (per key per day) -> totals per day, in the given day order. */
export function totalsByDay(days: string[], rows: UsageRow[]) {
  const by = new Map(days.map((d) => [d, { day: d, requests: 0, questions: 0, tokens: 0 }]));
  for (const r of rows) {
    const t = by.get(r.day);
    if (t) {
      t.requests += r.requests;
      t.questions += r.questions;
      t.tokens += r.tokens;
    }
  }
  return days.map((d) => by.get(d)!);
}
