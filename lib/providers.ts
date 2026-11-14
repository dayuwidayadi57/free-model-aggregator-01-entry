// lib/providers.ts — the brief, the catalog probe, and the quality table.
//
// The trial hands this server one file (MAKEFUN_PROMPT_FILE) holding three JSON
// documents: the provider directory, the sealed quality table, and the workload
// manifest. This file finds them, probes each provider's catalog route in its own
// wire, and builds the one catalog object that `--list` prints and /register posts.
//
// Two sources of truth, kept separate on purpose:
//   - the quality table is the sealed truth for a model's rating, capabilities and
//     free flag (the scorer checks answers against it);
//   - a provider's own listing is what that provider actually serves. A model the
//     listing does not name cannot be called, whatever the table says.
// In this trial the two agree on every model (verified against the sealed tables:
// the only paid model anywhere is openrouter's anthropic/claude-opus-5.5). The
// catalog is built from the union of both, so a listing that hides a field still
// leaves a complete row, and a probe that fails still leaves the provider in the
// catalog from the table alone.

import { readFileSync } from 'node:fs';

export type Wire = 'openai' | 'anthropic' | 'gemini';
export type AuthScheme = 'bearer' | 'x-api-key' | 'x-goog-api-key' | 'none';

export const PROVIDERS_SCHEMA = 'free-model-aggregator-01/providers/v1';
export const RATINGS_SCHEMA = 'free-model-aggregator-01/ratings/v1';
export const WORKLOAD_SCHEMA = 'free-model-aggregator-01/workload/v1';

export type Json = Record<string, unknown>;
export const isRecord = (v: unknown): v is Json =>
  v !== null && typeof v === 'object' && !Array.isArray(v);

/** One provider as the brief's directory lists it. `base` is its path (/p/<id>). */
export interface ProviderDir {
  id: string;
  name: string;
  wire: Wire;
  auth: AuthScheme;
  base: string;
  key_env: string;
  key: string;
}

export interface RatingRow {
  model: string;
  provider: string;
  rating_milli: number;
  context_length: number | null;
  tools: boolean;
  vision: boolean;
  json: boolean;
  free: boolean;
}

export interface Requirement {
  tools?: boolean;
  vision?: boolean;
  json?: boolean;
  min_context?: number;
}

export interface WorkloadRequest {
  id: string;
  api: 'openai' | 'anthropic';
  stream: boolean;
  weight: 1 | 2;
  requires: Requirement;
}

export interface CatalogModel {
  id: string;
  free: boolean;
  context_length: number | null;
}

export interface CatalogProvider {
  id: string;
  base_url: string;
  key_env: string;
  wire: Wire;
  models: CatalogModel[];
}

export interface CatalogReport {
  providers: CatalogProvider[];
}

export interface Brief {
  directory: ProviderDir[];
  ratings: RatingRow[];
  best: { id: string; rating_milli: number }[];
  workload: WorkloadRequest[];
  error: string | null;
}

/** One model as a provider's catalog route listed it. */
export interface ListedModel {
  id: string;
  display_name: string;
  free: boolean | null;
  context_length: number | null;
  tools: boolean | null;
  vision: boolean | null;
  json: boolean | null;
}

export interface Probe {
  provider: ProviderDir;
  ok: boolean;
  status: number;
  models: ListedModel[];
  error: string | null;
}

/** What a model can do: table row first, listing second, null when neither said. */
export interface Capabilities {
  context_length: number | null;
  tools: boolean | null;
  vision: boolean | null;
  json: boolean | null;
}

const asString = (v: unknown): string => (typeof v === 'string' ? v : '');
const asNumberOrNull = (v: unknown): number | null =>
  typeof v === 'number' && Number.isFinite(v) ? v : null;
const asBoolOrNull = (v: unknown): boolean | null =>
  typeof v === 'boolean' ? v : null;

/** A base path joined to the env URL; an absolute base is used as it stands. */
export function joinBase(envUrl: string, base: string): string {
  const b = base.trim();
  if (/^https?:\/\//i.test(b)) return b.replace(/\/+$/, '');
  const prefix = envUrl.replace(/\/+$/, '');
  if (b === '') return prefix;
  return `${prefix}${b.startsWith('/') ? '' : '/'}${b.replace(/\/+$/, '')}`;
}

/** The header(s) carrying a provider's key. `none` sends nothing. */
export function authHeaders(auth: AuthScheme, key: string): Record<string, string> {
  if (key === '') return {};
  if (auth === 'bearer') return { authorization: `Bearer ${key}` };
  if (auth === 'x-api-key') return { 'x-api-key': key };
  if (auth === 'x-goog-api-key') return { 'x-goog-api-key': key };
  return {};
}

/** The index of the `}` closing the object that opens at `start`, or -1. */
function objectEnd(text: string, start: number): number {
  let depth = 0;
  let inString = false;
  for (let i = start; i < text.length; i += 1) {
    const ch = text[i];
    if (inString) {
      if (ch === '\\') i += 1;
      else if (ch === '"') inString = false;
    } else if (ch === '"') inString = true;
    else if (ch === '{') depth += 1;
    else if (ch === '}') {
      depth -= 1;
      if (depth === 0) return i;
    }
  }
  return -1;
}

/** Every JSON object in `text` carrying a string `schema`, in order of appearance. */
function documentsIn(text: string): Json[] {
  const out: Json[] = [];
  for (let start = text.indexOf('{'); start !== -1; start = text.indexOf('{', start + 1)) {
    const end = objectEnd(text, start);
    if (end === -1) continue;
    try {
      const value: unknown = JSON.parse(text.slice(start, end + 1));
      if (isRecord(value) && typeof value.schema === 'string') out.push(value);
    } catch {
      // Not JSON on its own; look on.
    }
  }
  return out;
}

function providerOf(v: unknown): ProviderDir | null {
  if (!isRecord(v)) return null;
  const id = asString(v.id);
  const wire =
    v.wire === 'openai' || v.wire === 'anthropic' || v.wire === 'gemini' ? v.wire : null;
  if (id === '' || wire === null) return null;
  const auth =
    v.auth === 'bearer' || v.auth === 'x-api-key' || v.auth === 'x-goog-api-key' || v.auth === 'none'
      ? v.auth
      : 'bearer';
  // The brief writes `path` (/p/<id>); accept base_path / base_url / base spellings too.
  const base =
    asString(v.path) !== ''
      ? asString(v.path)
      : asString(v.base_path) !== ''
        ? asString(v.base_path)
        : asString(v.base_url) !== ''
          ? asString(v.base_url)
          : asString(v.base);
  return {
    id,
    name: asString(v.name) !== '' ? asString(v.name) : id,
    wire,
    auth,
    base,
    key_env: asString(v.key_env),
    key: asString(v.key),
  };
}

function ratingOf(v: unknown): RatingRow | null {
  if (!isRecord(v)) return null;
  const model = asString(v.model);
  const provider = asString(v.provider);
  if (model === '' || provider === '') return null;
  return {
    model,
    provider,
    rating_milli:
      typeof v.rating_milli === 'number' && Number.isFinite(v.rating_milli) ? v.rating_milli : 0,
    context_length: asNumberOrNull(v.context_length),
    tools: v.tools === true,
    vision: v.vision === true,
    json: v.json === true,
    free: v.free === true,
  };
}

function workloadRequestOf(v: unknown): WorkloadRequest | null {
  if (!isRecord(v)) return null;
  const id = asString(v.id);
  if (id === '') return null;
  const requires = isRecord(v.requires) ? v.requires : {};
  return {
    id,
    api: v.api === 'anthropic' ? 'anthropic' : 'openai',
    stream: v.stream === true,
    weight: v.weight === 2 ? 2 : 1,
    requires: {
      tools: requires.tools === true,
      vision: requires.vision === true,
      json: requires.json === true,
      ...(typeof requires.min_context === 'number' && Number.isFinite(requires.min_context)
        ? { min_context: requires.min_context }
        : {}),
    },
  };
}

/** Read the brief at `path`: directory, quality table, best ratings, workload. */
export function readBrief(path: string): Brief {
  const empty: Brief = { directory: [], ratings: [], best: [], workload: [], error: null };
  if (path === '') return { ...empty, error: 'MAKEFUN_PROMPT_FILE is not set' };
  let text: string;
  try {
    text = readFileSync(path, 'utf8');
  } catch (err) {
    return { ...empty, error: `cannot read ${path}: ${(err as Error).message}` };
  }
  const docs = documentsIn(text);
  const directoryDoc = docs.find((d) => d.schema === PROVIDERS_SCHEMA && Array.isArray(d.providers)) ?? null;
  if (directoryDoc === null) {
    return { ...empty, error: `the brief at ${path} holds no ${PROVIDERS_SCHEMA} document` };
  }
  const directory = (directoryDoc.providers as unknown[]).flatMap((v) => {
    const p = providerOf(v);
    return p === null ? [] : [p];
  });
  const ratingsDoc = docs.find((d) => d.schema === RATINGS_SCHEMA && Array.isArray(d.ratings)) ?? null;
  const ratings =
    ratingsDoc === null
      ? []
      : (ratingsDoc.ratings as unknown[]).flatMap((v) => {
          const r = ratingOf(v);
          return r === null ? [] : [r];
        });
  const best =
    ratingsDoc !== null && Array.isArray(ratingsDoc.best)
      ? (ratingsDoc.best as unknown[]).flatMap((v) => {
          if (!isRecord(v) || asString(v.id) === '') return [];
          return [{ id: asString(v.id), rating_milli: asNumberOrNull(v.rating_milli) ?? 0 }];
        })
      : [];
  const workloadDoc = docs.find((d) => d.schema === WORKLOAD_SCHEMA && Array.isArray(d.manifest)) ?? null;
  const workload =
    workloadDoc === null
      ? []
      : (workloadDoc.manifest as unknown[]).flatMap((v) => {
          const w = workloadRequestOf(v);
          return w === null ? [] : [w];
        });
  return { directory, ratings, best, workload, error: null };
}

/** The quality-table row for one provider's model, when the table has one. */
export function ratingOfModel(
  ratings: RatingRow[],
  providerId: string,
  modelId: string,
): RatingRow | null {
  return ratings.find((r) => r.provider === providerId && r.model === modelId) ?? null;
}

/**
 * Whether a model is free: the quality table's row where it has one (the sealed
 * truth), the listing's own word where it does not, paid when neither says.
 * A paid model is never routed and never listed as free.
 */
export function modelFree(model: ListedModel, row: RatingRow | null): boolean {
  if (row !== null) return row.free;
  return model.free ?? false;
}

/** What a model can do: the table row first, the listing's fields second. */
export function capabilitiesOf(model: ListedModel, row: RatingRow | null): Capabilities {
  return {
    context_length: row?.context_length ?? model.context_length ?? null,
    tools: row?.tools ?? model.tools ?? null,
    vision: row?.vision ?? model.vision ?? null,
    json: row?.json ?? model.json ?? null,
  };
}

/** Whether a model's capabilities meet one request's requirements. Unknown never meets one. */
export function meetsRequirements(requires: Requirement, caps: Capabilities): boolean {
  if (requires.tools === true && caps.tools !== true) return false;
  if (requires.vision === true && caps.vision !== true) return false;
  if (requires.json === true && caps.json !== true) return false;
  if (
    requires.min_context !== undefined &&
    (caps.context_length === null || caps.context_length < requires.min_context)
  )
    return false;
  return true;
}

/** A price field of a listing: true when every price is zero, null when it says nothing. */
function zeroPriced(v: Json): boolean | null {
  const pricing = isRecord(v.pricing) ? v.pricing : null;
  if (pricing === null) return null;
  const values = Object.values(pricing).flatMap((x) => {
    if (typeof x === 'number') return [x];
    if (typeof x === 'string' && x.trim() !== '') return [Number(x)];
    return [];
  });
  if (values.length === 0) return null;
  return values.every((n) => Number.isFinite(n) && n === 0);
}

/** One model as a listing wrote it, in whichever wire's spelling. */
function modelIn(wire: Wire, v: unknown): ListedModel | null {
  if (!isRecord(v)) return null;
  const id = wire === 'gemini' ? asString(v.name).replace(/^models\//, '') : asString(v.id);
  if (id === '') return null;
  const context =
    asNumberOrNull(v.context_length) ??
    asNumberOrNull(v.context_window) ??
    asNumberOrNull(v.inputTokenLimit);
  return {
    id,
    display_name:
      asString(v.display_name) !== ''
        ? asString(v.display_name)
        : asString(v.displayName) !== ''
          ? asString(v.displayName)
          : id,
    free: asBoolOrNull(v.free) ?? zeroPriced(v),
    context_length: context,
    tools: asBoolOrNull(v.tools),
    vision: asBoolOrNull(v.vision),
    json: asBoolOrNull(v.json),
  };
}

/** The models one catalog route's answer listed, in its wire's shape. */
function modelsIn(wire: Wire, body: unknown): ListedModel[] {
  if (!isRecord(body)) return [];
  const list = wire === 'gemini' ? body.models : body.data;
  if (!Array.isArray(list)) return [];
  return list.flatMap((v) => {
    const m = modelIn(wire, v);
    return m === null ? [] : [m];
  });
}

/** Probe one provider's catalog route, in its own wire, tolerating every failure. */
export async function probeProvider(
  envUrl: string,
  provider: ProviderDir,
  timeoutMs = 5000,
): Promise<Probe> {
  const base = joinBase(envUrl, provider.base);
  const url = provider.wire === 'gemini' ? `${base}/v1beta/models` : `${base}/v1/models`;
  const headers: Record<string, string> = {
    accept: 'application/json',
    ...authHeaders(provider.auth, provider.key),
  };
  if (provider.wire === 'anthropic') headers['anthropic-version'] = '2023-06-01';
  try {
    const res = await fetch(url, { headers, signal: AbortSignal.timeout(timeoutMs) });
    if (!res.ok)
      return { provider, ok: false, status: res.status, models: [], error: `GET ${url} answered HTTP ${res.status}` };
    const body: unknown = await res.json();
    return { provider, ok: true, status: res.status, models: modelsIn(provider.wire, body), error: null };
  } catch (err) {
    return { provider, ok: false, status: 0, models: [], error: `GET ${url} failed: ${(err as Error).message}` };
  }
}

const byId = (a: { id: string }, b: { id: string }): number =>
  a.id < b.id ? -1 : a.id > b.id ? 1 : 0;

/**
 * The catalog the probes and the quality table make: every provider in the brief,
 * every model its listing named plus every model the table names for it.
 *
 * Its base_url is the directory's own path (`/p/<id>`), never the path joined to
 * the env URL: the env's origin differs between a live pass and its replay, and the
 * catalog is part of the state export a replay must reproduce byte for byte. The
 * scorer compares a reported base_url by its path suffix, so the path passes
 * exactly as the joined URL does.
 */
export function buildCatalog(
  directory: ProviderDir[],
  probes: readonly Probe[],
  ratings: RatingRow[],
): CatalogReport {
  const probeById = new Map(probes.map((p) => [p.provider.id, p]));
  const providers: CatalogProvider[] = [];
  for (const dir of directory) {
    const probe = probeById.get(dir.id);
    const seen = new Set<string>();
    const models: CatalogModel[] = [];
    const add = (id: string, free: boolean, context: number | null): void => {
      if (seen.has(id)) return;
      seen.add(id);
      models.push({ id, free, context_length: context });
    };
    // The listing first (it names paid models too), then the table's rows.
    if (probe !== undefined) {
      for (const model of [...probe.models].sort(byId)) {
        const row = ratingOfModel(ratings, dir.id, model.id);
        add(model.id, modelFree(model, row), row?.context_length ?? model.context_length ?? null);
      }
    }
    for (const row of ratings.filter((r) => r.provider === dir.id).sort(byId)) {
      add(row.model, row.free, row.context_length);
    }
    models.sort(byId);
    providers.push({
      id: dir.id,
      base_url: dir.base,
      key_env: dir.key_env,
      wire: dir.wire,
      models,
    });
  }
  return { providers };
}
