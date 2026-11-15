// main.ts — one server, two client wires, ten free providers.
//
// This server reads the brief, builds one catalog, registers with the env, and
// answers the env client's 20 requests, each with the best free model that can
// serve it. It exits 0 the moment GET /work says the workload is done.
//
// Routing (the 60% quality game): fresh-first with a deficit guard, then the
// big-budget tier. An untouched provider's first call always answers, so a
// fresh provider is tried first — but only when its best model is close to
// the request's best (weighted deficit at most 55 milli). Past that, the
// request falls back to a provider already in use, big budgets first
// (kilol, pollinations, openrouter, groq) with load spread over the tier by
// fewest calls. When a provider refuses, the error's own message says why:
// "budget is spent" and "is out" are permanent (that provider is dead for
// the trial); "refused this call" is one bad call number (retry the same
// provider once, the next call almost always answers); anything else is
// treated as transient. A provider that fails twice in a row is set aside
// and tried last; one that fails on two separate requests is dead. Calls are
// strictly sequential — the client sends one request at a time, and no
// provider is ever called twice at once. There is no flat cap: a provider
// with many calls is still tried, just after its less-used big-tier peers.
//
// Determinism: every choice is a function of the brief, the requests and the
// providers' answers so far. No wall clock, no unseeded randomness, no parallel
// upstream calls — a replay of the trial leaves the same state export.

import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import {
  authHeaders,
  buildCatalog,
  capabilitiesOf,
  isRecord,
  joinBase,
  meetsRequirements,
  modelFree,
  probeProvider,
  ratingOfModel,
  readBrief,
  type Brief,
  type CatalogReport,
  type Json,
  type ListedModel,
  type Probe,
  type ProviderDir,
  type RatingRow,
  type Requirement,
  type Wire,
  type WorkloadRequest,
} from './lib/providers.ts';
import {
  anthropicError,
  anthropicJson,
  anthropicSse,
  anthropicToGemini,
  anthropicToOpenAI,
  anthropicModelList,
  openAIError,
  openAIJson,
  openAISse,
  openAIToAnthropic,
  openAIToGemini,
  openAIModelList,
  parseAnthropicAnswer,
  parseAnthropicStream,
  parseGeminiAnswer,
  parseGeminiStream,
  parseOpenAIAnswer,
  parseOpenAIStream,
  textOf,
  type ProviderAnswer,
  type ServedModel,
} from './lib/wire.ts';

const PROTOCOL_VERSION = 1;
const POLL_INTERVAL_MS = 100;
const PROBE_TIMEOUT_MS = 2000;
const UPSTREAM_TIMEOUT_MS = 10000;
const MAX_BODY_BYTES = 32 * 1024 * 1024;

const ENV_URL = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const PROMPT_FILE = process.env.MAKEFUN_PROMPT_FILE ?? '';

const brief: Brief = readBrief(PROMPT_FILE);
let probes: Probe[] = [];
let catalog: CatalogReport = { providers: [] };
/** Client requests arrived so far: the manifest is in arrival order. */
let manifestIndex = 0;

function log(message: string): void {
  process.stderr.write(`aggregator: ${message}\n`);
}

// ---- provider state (learned from answers, never from the clock) ----

interface ProviderState {
  calls: number;
  failedRequests: number;
  suspect: boolean;
  dead: boolean;
  badModels: Set<string>;
}

const providerState = new Map<string, ProviderState>();

function stateOf(providerId: string): ProviderState {
  let state = providerState.get(providerId);
  if (state === undefined) {
    state = { calls: 0, failedRequests: 0, suspect: false, dead: false, badModels: new Set() };
    providerState.set(providerId, state);
  }
  return state;
}

interface Candidate {
  provider: ProviderDir;
  model: RatingRow;
  rating: number;
}

function listedOf(providerId: string, modelId: string): ListedModel | null {
  const probe = probes.find((p) => p.provider.id === providerId);
  if (probe === undefined) return null;
  return probe.models.find((m) => m.id === modelId) ?? null;
}

/**
 * Routing (the 60% quality game): fresh-first with a deficit guard, then the
 * big-budget tier.
 *
 * A provider's FIRST call always answers (a refusal only ever strikes call
 * 2..budget-1), so an untouched provider is a guaranteed success — but only
 * worth spending when its best eligible model is close to the request's best:
 * weighted deficit (weight x best-minus-rating) at most 55 milli. Past that,
 * the request goes to a provider already in use, where the big-budget tier
 * (kilol, pollinations, openrouter, groq — the largest free budgets in the
 * kit's provider table) comes first and load spreads over it by fewest calls,
 * so no one provider is driven deep into its failing call numbers. Smaller
 * used providers follow by budget rank; untouched providers past the guard
 * and dead providers are the last resort, best rating first.
 *
 * One candidate per provider (its best eligible model): a second model on a
 * provider that just failed twice would only spend another call for a worse
 * rating. No flat cap anywhere — a provider with many calls is still tried,
 * just after its less-used big-tier peers.
 */
const DEFICIT_GUARD_MILLI = 55;
/** Largest free budgets first; the first four are the fallback tier. */
const BUDGET_ORDER = [
  'kilol',
  'pollinations',
  'openrouter',
  'groq',
  'cloudflare',
  'gemini',
  'ovh',
  'cerebras',
  'zai',
  'sambanova',
];
const BIG_BUDGET_TIER = 4;

function budgetRank(providerId: string): number {
  const rank = BUDGET_ORDER.indexOf(providerId);
  return rank === -1 ? BUDGET_ORDER.length : rank;
}

function candidatesFor(
  entry: WorkloadRequest | null,
  requires: Requirement,
  clientApi: 'openai' | 'anthropic',
  jsonMode: boolean,
): Candidate[] {
  // Best eligible model per provider; the rest of a provider's rows never
  // serve better than its best.
  const bestByProvider = new Map<string, Candidate>();
  for (const provider of brief.directory) {
    if (jsonMode && clientApi === 'openai' && provider.wire === 'anthropic') continue;
    for (const row of brief.ratings) {
      if (row.provider !== provider.id || !row.free) continue;
      const listed = listedOf(provider.id, row.model);
      const caps = capabilitiesOf(
        listed ?? { id: row.model, display_name: row.model, free: true, context_length: row.context_length, tools: row.tools, vision: row.vision, json: row.json },
        row,
      );
      if (!meetsRequirements(requires, caps)) continue;
      if (stateOf(provider.id).badModels.has(row.model)) continue;
      const prev = bestByProvider.get(provider.id);
      if (
        prev === undefined ||
        row.rating_milli > prev.rating ||
        (row.rating_milli === prev.rating && row.model < prev.model.model)
      ) {
        bestByProvider.set(provider.id, { provider, model: row, rating: row.rating_milli });
      }
    }
  }
  const rows = [...bestByProvider.values()];
  // The request's best rating from the sealed table (the manifest always
  // carries it); else the best eligible row.
  const tableBest = entry === null
    ? 0
    : (brief.best.find((b) => b.id === entry.id)?.rating_milli ?? 0);
  const top = tableBest > 0 ? tableBest : rows.reduce((m, c) => Math.max(m, c.rating), 0);
  const weight = entry?.weight ?? 1;
  const deficitOf = (c: Candidate): number => weight * (top - c.rating);
  const byRating = (a: Candidate, b: Candidate): number =>
    b.rating - a.rating ||
    (a.provider.id < b.provider.id ? -1 : a.provider.id > b.provider.id ? 1 : 0) ||
    (a.model.model < b.model.model ? -1 : a.model.model > b.model.model ? 1 : 0);
  const freshOk: Candidate[] = [];
  const usedBig: Candidate[] = [];
  const usedSmall: Candidate[] = [];
  const freshFar: Candidate[] = [];
  const dead: Candidate[] = [];
  for (const c of rows) {
    const s = stateOf(c.provider.id);
    if (s.dead) {
      dead.push(c);
    } else if (s.calls === 0) {
      (deficitOf(c) <= DEFICIT_GUARD_MILLI ? freshOk : freshFar).push(c);
    } else if (budgetRank(c.provider.id) < BIG_BUDGET_TIER) {
      usedBig.push(c);
    } else {
      usedSmall.push(c);
    }
  }
  freshOk.sort(byRating);
  freshFar.sort(byRating);
  dead.sort(byRating);
  usedBig.sort((a, b) => {
    const sa = stateOf(a.provider.id);
    const sb = stateOf(b.provider.id);
    if (sa.suspect !== sb.suspect) return sa.suspect ? 1 : -1;
    if (sa.calls !== sb.calls) return sa.calls - sb.calls;
    return byRating(a, b);
  });
  usedSmall.sort((a, b) => {
    const sa = stateOf(a.provider.id);
    const sb = stateOf(b.provider.id);
    if (sa.suspect !== sb.suspect) return sa.suspect ? 1 : -1;
    if (budgetRank(a.provider.id) !== budgetRank(b.provider.id))
      return budgetRank(a.provider.id) - budgetRank(b.provider.id);
    return byRating(a, b);
  });
  return [...freshOk, ...usedBig, ...usedSmall, ...freshFar, ...dead];
}

// ---- the failure's own message says whether it is permanent ----

type FailureKind = 'spent' | 'outage' | 'transient' | 'concurrent' | 'badModel' | 'auth' | 'other';

function classifyFailure(status: number, code: string, message: string): FailureKind {
  if (code === 'invalid_model' || /paid model/.test(message)) return 'badModel';
  if (status === 401) return 'auth';
  if (code === 'concurrent_requests') return 'concurrent';
  if (/budget is spent/.test(message)) return 'spent';
  if (/ is out/.test(message)) return 'outage';
  if (/refused this call/.test(message)) return 'transient';
  return 'other';
}

function errorMessageOf(wire: Wire, body: unknown): { code: string; message: string } {
  if (!isRecord(body)) return { code: '', message: '' };
  const error = isRecord(body.error) ? body.error : null;
  if (error === null) return { code: '', message: '' };
  const code =
    typeof error.code === 'string'
      ? error.code
      : typeof error.type === 'string'
        ? error.type
        : typeof error.status === 'string'
          ? error.status
          : '';
  const message = typeof error.message === 'string' ? error.message : '';
  return { code, message };
}

// ---- one upstream call ----

interface UpstreamOk {
  ok: true;
  answer: ProviderAnswer;
}

interface UpstreamFail {
  ok: false;
  status: number;
  kind: FailureKind;
  message: string;
}

type UpstreamResult = UpstreamOk | UpstreamFail;

function upstreamBody(
  candidate: Candidate,
  clientApi: 'openai' | 'anthropic',
  body: Json,
): { wire: Wire; url: string; payload: Json } {
  const provider = candidate.provider;
  const base = joinBase(ENV_URL, provider.base);
  const stream = body.stream === true;
  if (provider.wire === 'openai') {
    // An OpenAI client passes through with the routed model; an Anthropic
    // client must be rewritten, or its top-level system field, its anthropic
    // tools and its stop_sequences never reach the provider's parser.
    const payload: Json =
      clientApi === 'anthropic'
        ? anthropicToOpenAI(body, candidate.model.model)
        : { ...body, model: candidate.model.model };
    if (stream) payload.stream_options = { include_usage: true };
    return { wire: 'openai', url: `${base}/v1/chat/completions`, payload };
  }
  if (provider.wire === 'anthropic') {
    const payload =
      clientApi === 'anthropic'
        ? { ...body, model: candidate.model.model }
        : openAIToAnthropic(body, candidate.model.model);
    return { wire: 'anthropic', url: `${base}/v1/messages`, payload };
  }
  const payload =
    clientApi === 'anthropic' ? anthropicToGemini(body) : openAIToGemini(body);
  const action = stream ? 'streamGenerateContent' : 'generateContent';
  return {
    wire: 'gemini',
    url: `${base}/v1beta/models/${encodeURIComponent(candidate.model.model)}:${action}`,
    payload,
  };
}

async function callUpstream(
  candidate: Candidate,
  clientApi: 'openai' | 'anthropic',
  body: Json,
): Promise<UpstreamResult> {
  const provider = candidate.provider;
  const state = stateOf(provider.id);
  const { wire, url, payload } = upstreamBody(candidate, clientApi, body);
  const headers: Record<string, string> = {
    'content-type': 'application/json',
    ...authHeaders(provider.auth, provider.key),
  };
  if (wire === 'anthropic') headers['anthropic-version'] = '2023-06-01';
  if (body.stream === true) headers.accept = 'text/event-stream';
  state.calls += 1;
  let res: Response;
  try {
    res = await fetch(url, {
      method: 'POST',
      headers,
      body: JSON.stringify(payload),
      signal: AbortSignal.timeout(UPSTREAM_TIMEOUT_MS),
    });
  } catch (err) {
    return { ok: false, status: 0, kind: 'transient', message: (err as Error).message };
  }
  const text = await res.text();
  if (!res.ok) {
    let parsed: unknown = null;
    try {
      parsed = JSON.parse(text);
    } catch {
      // Not JSON: the status alone classifies it.
    }
    const { code, message } = errorMessageOf(wire, parsed);
    return { ok: false, status: res.status, kind: classifyFailure(res.status, code, message), message };
  }
  const stream = body.stream === true;
  if (stream) {
    const answer =
      wire === 'openai'
        ? parseOpenAIStream(text)
        : wire === 'anthropic'
          ? parseAnthropicStream(text)
          : parseGeminiStream(text);
    return { ok: true, answer };
  }
  let parsed: unknown = null;
  try {
    parsed = JSON.parse(text);
  } catch {
    return { ok: false, status: res.status, kind: 'other', message: 'the provider answered a body that is not JSON' };
  }
  if (!isRecord(parsed)) {
    return { ok: false, status: res.status, kind: 'other', message: 'the provider answered a JSON value that is not one object' };
  }
  const answer =
    wire === 'openai'
      ? parseOpenAIAnswer(parsed)
      : wire === 'anthropic'
        ? parseAnthropicAnswer(parsed)
        : parseGeminiAnswer(parsed);
  if (answer === null) {
    return { ok: false, status: res.status, kind: 'other', message: 'the provider answered a body with no usable answer' };
  }
  return { ok: true, answer };
}

// ---- serving one client request ----

function requirementsFor(body: Json): Requirement {
  const manifest = brief.workload;
  const entry = manifest[manifestIndex];
  manifestIndex += 1;
  if (entry !== undefined) return entry.requires;
  // The manifest is the declared source; this fallback only guesses for a brief
  // that holds none.
  const messages = Array.isArray(body.messages) ? body.messages : [];
  let vision = false;
  for (const message of messages) {
    if (!isRecord(message) || !Array.isArray(message.content)) continue;
    for (const block of message.content) {
      if (isRecord(block) && (block.type === 'image' || block.type === 'image_url')) vision = true;
    }
  }
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  return {
    tools: Array.isArray(body.tools) && body.tools.length > 0,
    vision,
    json: format === 'json_object' || format === 'json_schema',
  };
}

function jsonModeOf(clientApi: 'openai' | 'anthropic', body: Json): boolean {
  if (clientApi !== 'openai') return false;
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  return format === 'json_object' || format === 'json_schema';
}

interface Served {
  answer: ProviderAnswer;
  model: string;
}

/**
 * Answer one request: try the candidates in order, two attempts each. A
 * permanent refusal (spent budget, outage, wrong key) is not retried on the
 * same provider; a transient one is retried once. The first success wins —
 * a second successful call would fail the client's one-provider check.
 */
async function serve(
  clientApi: 'openai' | 'anthropic',
  body: Json,
): Promise<Served | null> {
  const entry = brief.workload[manifestIndex] ?? null;
  const requires = requirementsFor(body);
  const jsonMode = jsonModeOf(clientApi, body);
  const candidates = candidatesFor(entry, requires, clientApi, jsonMode);
  for (const candidate of candidates) {
    const state = stateOf(candidate.provider.id);
    let failedThisRequest = false;
    for (let attempt = 0; attempt < 2; attempt += 1) {
      const result = await callUpstream(candidate, clientApi, body);
      if (result.ok) {
        state.suspect = false;
        return { answer: result.answer, model: candidate.model.model };
      }
      const kind = result.kind;
      if (kind === 'badModel') {
        state.badModels.add(candidate.model.model);
        log(`${candidate.provider.id}/${candidate.model.model} refused as a model: ${result.message}`);
        break;
      }
      if (kind === 'auth') {
        state.dead = true;
        log(`${candidate.provider.id} refused the key: ${result.message}`);
        break;
      }
      if (kind === 'spent' || kind === 'outage') {
        state.dead = true;
        log(`${candidate.provider.id} is ${kind === 'spent' ? 'out of budget' : 'out'}: ${result.message}`);
        break;
      }
      // Transient, concurrent, or anything else: one retry, then move on.
      failedThisRequest = true;
      if (attempt === 0) continue;
    }
    if (failedThisRequest) {
      state.failedRequests += 1;
      if (state.failedRequests >= 2) state.dead = true;
      else state.suspect = true;
    }
  }
  return null;
}

// ---- the client routes ----

async function readJson(req: IncomingMessage): Promise<Json | null> {
  const chunks: Buffer[] = [];
  let size = 0;
  for await (const chunk of req) {
    const buffer = Buffer.isBuffer(chunk) ? chunk : Buffer.from(String(chunk));
    size += buffer.length;
    if (size > MAX_BODY_BYTES) return null;
    chunks.push(buffer);
  }
  if (chunks.length === 0) return null;
  try {
    const value: unknown = JSON.parse(Buffer.concat(chunks).toString('utf8'));
    return isRecord(value) ? value : null;
  } catch {
    return null;
  }
}

function sendJson(res: ServerResponse, status: number, body: unknown): void {
  const text = JSON.stringify(body);
  res.writeHead(status, { 'content-type': 'application/json', 'content-length': Buffer.byteLength(text) });
  res.end(text);
}

function sendSse(res: ServerResponse, text: string): void {
  res.writeHead(200, {
    'content-type': 'text/event-stream',
    'cache-control': 'no-cache',
    connection: 'keep-alive',
    'content-length': Buffer.byteLength(text),
  });
  res.end(text);
}

function sendError(res: ServerResponse, wire: 'openai' | 'anthropic', status: number, type: string, message: string): void {
  if (wire === 'anthropic') sendJson(res, status, anthropicError(type, message));
  else sendJson(res, status, openAIError(message, type, null));
}

/** The free models across the catalog, each id once, in id order. */
function servedModels(): ServedModel[] {
  const seen = new Map<string, ServedModel>();
  for (const provider of catalog.providers) {
    for (const model of provider.models) {
      if (!model.free || seen.has(model.id)) continue;
      seen.set(model.id, { id: model.id, provider: provider.id, display_name: model.id });
    }
  }
  return [...seen.values()].sort((a, b) => (a.id < b.id ? -1 : a.id > b.id ? 1 : 0));
}

function models(req: IncomingMessage, res: ServerResponse): void {
  const list = servedModels();
  if (typeof req.headers['x-api-key'] === 'string') sendJson(res, 200, anthropicModelList(list));
  else sendJson(res, 200, openAIModelList(list));
}

async function chat(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null)
    return sendError(res, 'openai', 400, 'invalid_request_error', 'the request body is not one JSON object');
  const served = await serve('openai', body);
  if (served === null)
    return sendError(res, 'openai', 503, 'no_free_model', 'no free model on the directory could serve this request');
  if (body.stream === true) sendSse(res, openAISse(served.answer, served.model));
  else sendJson(res, 200, openAIJson(served.answer, served.model));
}

async function messages(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null)
    return sendError(res, 'anthropic', 400, 'invalid_request_error', 'the request body is not one JSON object');
  const served = await serve('anthropic', body);
  if (served === null)
    return sendError(res, 'anthropic', 503, 'overloaded_error', 'no free model on the directory could serve this request');
  if (body.stream === true) sendSse(res, anthropicSse(served.answer, served.model));
  else sendJson(res, 200, anthropicJson(served.answer, served.model));
}

async function handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const url = new URL(req.url ?? '/', 'http://entry');
  if (req.method === 'GET' && url.pathname === '/v1/models') {
    models(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/chat/completions') {
    await chat(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/messages') {
    await messages(req, res);
    return;
  }
  sendError(res, url.pathname === '/v1/messages' ? 'anthropic' : 'openai', 404, 'not_found', `no route ${req.method ?? ''} ${url.pathname}`);
}

function listen(server: Server): Promise<void> {
  return new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '0.0.0.0', () => resolve());
  });
}

/** POST /register with the port and the one catalog. */
async function register(port: number, report: CatalogReport): Promise<void> {
  let lastError: Error | null = null;
  for (let attempt = 0; attempt < 3; attempt += 1) {
    try {
      const res = await fetch(`${ENV_URL}/register`, {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify({ protocol_version: PROTOCOL_VERSION, port, catalog: report }),
        signal: AbortSignal.timeout(5000),
      });
      if (!res.ok)
        throw new Error(`POST ${ENV_URL}/register answered HTTP ${res.status}: ${(await res.text()).slice(0, 500)}`);
      const reply: unknown = await res.json();
      if (isRecord(reply) && typeof reply.requests === 'number')
        log(`registered: the workload holds ${reply.requests} requests`);
      return;
    } catch (err) {
      lastError = err as Error;
      await new Promise((resolve) => setTimeout(resolve, 200));
    }
  }
  throw lastError ?? new Error('register failed');
}

/** Poll GET /work; exit 0 the moment it says done. */
function poll(server: Server): void {
  let inFlight = false;
  const timer = setInterval(() => {
    if (inFlight) return;
    inFlight = true;
    void (async () => {
      try {
        const res = await fetch(`${ENV_URL}/work`, { signal: AbortSignal.timeout(2000) });
        if (res.status !== 200) return;
        const work: unknown = await res.json();
        if (isRecord(work) && work.done === true) {
          clearInterval(timer);
          server.close();
          process.exit(0);
        }
      } catch {
        // The env is not reachable yet: try again on the next tick.
      } finally {
        inFlight = false;
      }
    })();
  }, POLL_INTERVAL_MS);
}

async function load(): Promise<void> {
  if (brief.error !== null) log(brief.error);
  // Probes run one at a time: the client sends one request at a time, and no
  // provider is ever called twice at once.
  probes = [];
  for (const provider of brief.directory) {
    probes.push(await probeProvider(ENV_URL, provider, PROBE_TIMEOUT_MS));
  }
  for (const probe of probes) if (!probe.ok && probe.error !== null) log(probe.error);
  catalog = buildCatalog(brief.directory, probes, brief.ratings);
}

async function start(): Promise<void> {
  await load();
  if (process.argv.includes('--list')) {
    // The same document the register body carries, byte for byte: print the one
    // built object and stop before any server starts.
    const text = `${JSON.stringify(catalog)}\n`;
    await new Promise<void>((resolve) => process.stdout.write(text, () => resolve()));
    process.exit(0);
  }
  if (ENV_URL === '') throw new Error('MAKEFUN_ENV_URL is not set');
  const server = createServer((req, res) => {
    void handle(req, res).catch((err: unknown) => {
      log((err as Error).stack ?? String(err));
      if (!res.headersSent)
        sendError(
          res,
          req.url?.startsWith('/v1/messages') === true ? 'anthropic' : 'openai',
          500,
          'internal_error',
          'the server failed to answer',
        );
    });
  });
  await listen(server);
  const address = server.address();
  if (address === null || typeof address === 'string') throw new Error('the server has no port');
  const port = address.port;
  if (port === 8080) throw new Error('the OS picked 8080, which the env owns on one layout: restart the trial');
  log(`listening on 0.0.0.0:${port} with ${catalog.providers.length} providers and ${servedModels().length} free models`);
  await register(port, catalog);
  process.on('SIGTERM', () => process.exit(0));
  process.on('SIGINT', () => process.exit(0));
  poll(server);
}

try {
  await start();
} catch (err) {
  log((err as Error).stack ?? String(err));
  process.exit(1);
}
