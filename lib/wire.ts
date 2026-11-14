// lib/wire.ts — the client wires and the three upstream wires.
//
// The client speaks OpenAI Chat Completions or Anthropic Messages. Each provider
// speaks its own wire (openai, anthropic or gemini). This file translates both
// ways, in JSON and in server-sent events.
//
// The whole game is faithful forwarding: every provider answers with the text
// `ANSWER ECHO{...}`, a fingerprint of the request *as the provider parsed it*
// (system text, user text, tool names, max_tokens, stops, image bytes, JSON mode),
// and the env's client compares that echo with the request it originally sent.
// A field dropped or rewritten in translation fails that request. So every
// builder below passes the client's system prompt, user text (images included),
// tools, stop sequences, max_tokens and JSON mode through untouched, in the
// target wire's spelling, and every answer translator copies the provider's text,
// usage and stop reason back whole.
//
// Everything here is a pure function of its arguments: no clock, no randomness,
// so the same body and the same provider answer give the same bytes on a replay.
// The two model listings use fixed timestamps for the same reason.

import { isRecord, type Json } from './providers.ts';

/** One model the entry serves, as GET /v1/models lists it. */
export interface ServedModel {
  id: string;
  provider: string;
  display_name: string;
}

/** The provider's answer, normalised: its text, its usage, its stop reason. */
export interface ProviderAnswer {
  text: string;
  inputTokens: number;
  outputTokens: number;
  stop: string;
}

/** A fixed epoch second for every listed model's `created`. */
const FIXED_OPENAI_CREATED = 1767225600;
/** A fixed instant for every listed model's `created_at`. */
const FIXED_ANTHROPIC_CREATED_AT = '2026-01-01T00:00:00Z';
/** The provider splits streamed text into pieces this long; so do the emitters. */
const SSE_DELTA_CHARS = 13;

/** GET /v1/models on an OpenAI-shaped request. */
export function openAIModelList(models: readonly ServedModel[]): Json {
  return {
    object: 'list',
    data: models.map((m) => ({
      id: m.id,
      object: 'model',
      created: FIXED_OPENAI_CREATED,
      owned_by: m.provider,
    })),
  };
}

/** GET /v1/models on an Anthropic-shaped request (an x-api-key header). */
export function anthropicModelList(models: readonly ServedModel[]): Json {
  return {
    data: models.map((m) => ({
      type: 'model',
      id: m.id,
      display_name: m.display_name,
      created_at: FIXED_ANTHROPIC_CREATED_AT,
    })),
    has_more: false,
  };
}

/** An error in Anthropic's shape. */
export function anthropicError(type: string, message: string): Json {
  return { type: 'error', error: { type, message } };
}

/** An error in OpenAI's shape. */
export function openAIError(message: string, type: string, code: string | null): Json {
  return { error: { message, type, code } };
}

/** The text of a string or of a block array's text blocks, joined as the env joins them. */
export function textOf(content: unknown): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  return content
    .flatMap((b) =>
      isRecord(b) && b.type === 'text' && typeof b.text === 'string' ? [b.text] : [],
    )
    .join('\n');
}

/** The text of an Anthropic content array's text blocks, joined with nothing. */
function anthropicTextOf(content: unknown): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  const parts: string[] = [];
  for (const block of content) {
    if (isRecord(block) && block.type === 'text' && typeof block.text === 'string') parts.push(block.text);
  }
  return parts.join('');
}

/** A provider stop reason as an OpenAI finish_reason. */
export function stopToOpenAI(stop: string): string {
  if (stop === 'max_tokens' || stop === 'MAX_TOKENS' || stop === 'length') return 'length';
  if (stop === 'tool_use' || stop === 'tool_calls' || stop === 'function_call') return 'tool_calls';
  if (stop === 'refusal' || stop === 'content_filter' || stop === 'SAFETY') return 'content_filter';
  return 'stop';
}

/** A provider stop reason as an Anthropic stop_reason. */
export function stopToAnthropic(stop: string): string {
  if (stop === 'max_tokens' || stop === 'MAX_TOKENS' || stop === 'length') return 'max_tokens';
  if (stop === 'tool_use' || stop === 'tool_calls' || stop === 'function_call') return 'tool_use';
  if (stop === 'refusal' || stop === 'content_filter' || stop === 'SAFETY') return 'refusal';
  return 'end_turn';
}

function numberOrZero(value: unknown): number {
  return typeof value === 'number' && Number.isFinite(value) ? value : 0;
}

/** A tool call's arguments as an object: the provider's JSON string, or {} when not one. */
function argumentsOf(value: unknown): Json {
  if (isRecord(value)) return value;
  if (typeof value === 'string' && value.trim() !== '') {
    try {
      const parsed: unknown = JSON.parse(value);
      if (isRecord(parsed)) return parsed;
    } catch {
      // Not JSON: an empty input.
    }
  }
  return {};
}

/** A tool_result block's content as text: its text blocks, or its JSON when it holds more. */
function toolResultText(content: unknown): string {
  if (typeof content === 'string') return content;
  if (Array.isArray(content)) {
    const text = textOf(content);
    return text !== '' ? text : JSON.stringify(content);
  }
  return content === undefined ? '' : JSON.stringify(content);
}

/** One Anthropic content block as an OpenAI content part: text or an inline image. */
function anthropicPartOf(block: Json): Json | null {
  if (block.type === 'text' && typeof block.text === 'string')
    return { type: 'text', text: block.text };
  if (block.type === 'image' && isRecord(block.source)) {
    const source = block.source;
    if (source.type === 'base64' && typeof source.data === 'string') {
      const media = typeof source.media_type === 'string' ? source.media_type : 'image/png';
      return { type: 'image_url', image_url: { url: `data:${media};base64,${source.data}` } };
    }
    if (source.type === 'url' && typeof source.url === 'string')
      return { type: 'image_url', image_url: { url: source.url } };
  }
  return null;
}

/**
 * An Anthropic Messages request as an OpenAI Chat Completions request naming
 * `model`: the system prompt becomes a system message, image blocks become
 * image_url data URLs, tool_use blocks become tool_calls, tool_result blocks
 * become tool messages, tools and tool_choice are rewritten, and the output cap,
 * temperature, top_p, stop sequences and streaming pass through.
 */
export function anthropicToOpenAI(body: Json, model: string): Json {
  const messages: Json[] = [];
  const system = textOf(body.system);
  if (system !== '') messages.push({ role: 'system', content: system });
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    const assistant = raw.role === 'assistant';
    const content = raw.content;
    if (typeof content === 'string') {
      messages.push({ role: assistant ? 'assistant' : 'user', content });
      continue;
    }
    if (!Array.isArray(content)) continue;
    if (assistant) {
      const toolCalls: Json[] = [];
      for (const block of content) {
        if (!isRecord(block) || block.type !== 'tool_use') continue;
        toolCalls.push({
          id: typeof block.id === 'string' ? block.id : '',
          type: 'function',
          function: {
            name: typeof block.name === 'string' ? block.name : '',
            arguments: JSON.stringify(isRecord(block.input) ? block.input : {}),
          },
        });
      }
      const text = textOf(content);
      messages.push({
        role: 'assistant',
        content: text !== '' ? text : null,
        ...(toolCalls.length > 0 ? { tool_calls: toolCalls } : {}),
      });
      continue;
    }
    const parts: Json[] = [];
    for (const block of content) {
      if (!isRecord(block)) continue;
      if (block.type === 'tool_result') {
        messages.push({
          role: 'tool',
          tool_call_id: typeof block.tool_use_id === 'string' ? block.tool_use_id : '',
          content: toolResultText(block.content),
        });
        continue;
      }
      const part = anthropicPartOf(block);
      if (part !== null) parts.push(part);
    }
    if (parts.length > 0) messages.push({ role: 'user', content: parts });
  }

  const out: Json = { model, messages };
  if (typeof body.max_tokens === 'number') out.max_tokens = body.max_tokens;
  if (typeof body.temperature === 'number') out.temperature = body.temperature;
  if (typeof body.top_p === 'number') out.top_p = body.top_p;
  if (Array.isArray(body.stop_sequences) && body.stop_sequences.length > 0)
    out.stop = body.stop_sequences;
  if (Array.isArray(body.tools)) {
    const tools = body.tools.flatMap((raw) => {
      if (!isRecord(raw) || typeof raw.name !== 'string') return [];
      return [
        {
          type: 'function',
          function: {
            name: raw.name,
            description: typeof raw.description === 'string' ? raw.description : '',
            parameters: isRecord(raw.input_schema) ? raw.input_schema : { type: 'object', properties: {} },
          },
        },
      ];
    });
    if (tools.length > 0) out.tools = tools;
  }
  const choice = body.tool_choice;
  if (isRecord(choice)) {
    if (choice.type === 'auto') out.tool_choice = 'auto';
    else if (choice.type === 'any') out.tool_choice = 'required';
    else if (choice.type === 'tool' && typeof choice.name === 'string')
      out.tool_choice = { type: 'function', function: { name: choice.name } };
  }
  if (body.stream === true) {
    out.stream = true;
    out.stream_options = { include_usage: true };
  }
  return out;
}

/** One OpenAI content part as an Anthropic content block: text or an inline image. */
function openAIPartOf(part: Json): Json | null {
  if (part.type === 'text' && typeof part.text === 'string')
    return { type: 'text', text: part.text };
  if (part.type === 'image_url' && isRecord(part.image_url) && typeof part.image_url.url === 'string') {
    const url = part.image_url.url;
    const data = /^data:([^,]*);base64,(.*)$/s.exec(url);
    if (data !== null)
      return {
        type: 'image',
        source: { type: 'base64', media_type: data[1] === '' ? 'image/png' : data[1], data: data[2] },
      };
    return { type: 'image', source: { type: 'url', url } };
  }
  return null;
}

/**
 * An OpenAI Chat Completions request as an Anthropic Messages request naming
 * `model`: system messages become the system field, image_url parts become image
 * blocks, tool_calls become tool_use blocks, tool messages become tool_result
 * blocks, and tools, tool_choice, stop, max_tokens, temperature and streaming
 * pass through. Anthropic has no JSON mode: a JSON-mode request must never be
 * routed to an anthropic-wire provider (the router excludes them), so this
 * builder drops response_format rather than inventing one.
 */
export function openAIToAnthropic(body: Json, model: string): Json {
  const systemParts: string[] = [];
  const messages: Json[] = [];
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    const role = raw.role;
    const content = raw.content;
    if (role === 'system') {
      const text = textOf(content);
      if (text !== '') systemParts.push(text);
      continue;
    }
    if (role === 'assistant') {
      const blocks: Json[] = [];
      const text = textOf(content);
      if (text !== '') blocks.push({ type: 'text', text });
      const calls = Array.isArray(raw.tool_calls) ? raw.tool_calls : [];
      for (const call of calls) {
        if (!isRecord(call) || !isRecord(call.function)) continue;
        blocks.push({
          type: 'tool_use',
          id: typeof call.id === 'string' ? call.id : '',
          name: typeof call.function.name === 'string' ? call.function.name : '',
          input: argumentsOf(call.function.arguments),
        });
      }
      if (blocks.length > 0) messages.push({ role: 'assistant', content: blocks });
      continue;
    }
    if (role === 'tool') {
      const text = typeof content === 'string' ? content : textOf(content);
      messages.push({
        role: 'user',
        content: [
          {
            type: 'tool_result',
            tool_use_id: typeof raw.tool_call_id === 'string' ? raw.tool_call_id : '',
            content: text,
          },
        ],
      });
      continue;
    }
    // Any other role is a user message.
    if (typeof content === 'string') {
      messages.push({ role: 'user', content });
      continue;
    }
    if (Array.isArray(content)) {
      const blocks: Json[] = [];
      for (const part of content) {
        if (!isRecord(part)) continue;
        const block = openAIPartOf(part);
        if (block !== null) blocks.push(block);
      }
      if (blocks.length > 0) messages.push({ role: 'user', content: blocks });
    }
  }

  const out: Json = { model, messages };
  if (systemParts.length > 0) out.system = systemParts.join('\n');
  if (typeof body.max_tokens === 'number') out.max_tokens = body.max_tokens;
  else if (typeof body.max_completion_tokens === 'number')
    out.max_tokens = body.max_completion_tokens;
  if (typeof body.temperature === 'number') out.temperature = body.temperature;
  if (typeof body.top_p === 'number') out.top_p = body.top_p;
  if (Array.isArray(body.stop) && body.stop.length > 0) out.stop_sequences = body.stop;
  else if (typeof body.stop === 'string' && body.stop !== '') out.stop_sequences = [body.stop];
  if (Array.isArray(body.tools)) {
    const tools = body.tools.flatMap((raw) => {
      if (!isRecord(raw) || !isRecord(raw.function) || typeof raw.function.name !== 'string') return [];
      return [
        {
          name: raw.function.name,
          description: typeof raw.function.description === 'string' ? raw.function.description : '',
          input_schema: isRecord(raw.function.parameters)
            ? raw.function.parameters
            : { type: 'object', properties: {} },
        },
      ];
    });
    if (tools.length > 0) out.tools = tools;
  }
  const choice = body.tool_choice;
  if (choice === 'auto') out.tool_choice = { type: 'auto' };
  else if (choice === 'required' || choice === 'any') out.tool_choice = { type: 'any' };
  else if (choice === 'none') out.tool_choice = { type: 'none' };
  else if (isRecord(choice) && choice.type === 'function' && isRecord(choice.function))
    out.tool_choice = {
      type: 'tool',
      name: typeof choice.function.name === 'string' ? choice.function.name : '',
    };
  if (body.stream === true) out.stream = true;
  return out;
}

/** One OpenAI content part as a Gemini part: text, inline image data, or a file URI. */
function openAIGeminiPartOf(part: Json): Json | null {
  if (part.type === 'text' && typeof part.text === 'string') return { text: part.text };
  if (part.type === 'image_url' && isRecord(part.image_url) && typeof part.image_url.url === 'string') {
    const url = part.image_url.url;
    const data = /^data:([^,]*);base64,(.*)$/s.exec(url);
    if (data !== null)
      return {
        inlineData: { mimeType: data[1] === '' ? 'image/png' : data[1], data: data[2] },
      };
    return { fileData: { fileUri: url } };
  }
  return null;
}

/** One Anthropic content block as a Gemini part. */
function anthropicGeminiPartOf(block: Json): Json | null {
  if (block.type === 'text' && typeof block.text === 'string') return { text: block.text };
  if (block.type === 'image' && isRecord(block.source)) {
    const source = block.source;
    if (source.type === 'base64' && typeof source.data === 'string')
      return {
        inlineData: {
          mimeType: typeof source.media_type === 'string' ? source.media_type : 'image/png',
          data: source.data,
        },
      };
    if (source.type === 'url' && typeof source.url === 'string')
      return { fileData: { fileUri: source.url } };
  }
  return null;
}

function geminiToolConfig(choice: unknown): Json | null {
  if (choice === 'auto' || (isRecord(choice) && choice.type === 'auto'))
    return { functionCallingConfig: { mode: 'AUTO' } };
  if (choice === 'required' || choice === 'any' || (isRecord(choice) && choice.type === 'any'))
    return { functionCallingConfig: { mode: 'ANY' } };
  if (choice === 'none' || (isRecord(choice) && choice.type === 'none'))
    return { functionCallingConfig: { mode: 'NONE' } };
  if (isRecord(choice) && choice.type === 'function' && isRecord(choice.function)) {
    const name = typeof choice.function.name === 'string' ? choice.function.name : '';
    return {
      functionCallingConfig: { mode: 'ANY', allowedFunctionNames: name === '' ? [] : [name] },
    };
  }
  if (isRecord(choice) && choice.type === 'tool' && typeof choice.name === 'string')
    return { functionCallingConfig: { mode: 'ANY', allowedFunctionNames: [choice.name] } };
  return null;
}

/** The shared Gemini body: contents, system instruction, tools, generation config. */
function geminiBody(
  contents: Json[],
  systemText: string,
  declarations: Json[],
  choice: unknown,
  maxTokens: number | null,
  temperature: number | null,
  topP: number | null,
  stops: string[],
  jsonMode: boolean,
): Json {
  const body: Json = { contents };
  if (systemText !== '') body.systemInstruction = { parts: [{ text: systemText }] };
  if (declarations.length > 0) body.tools = [{ functionDeclarations: declarations }];
  const toolConfig = geminiToolConfig(choice);
  if (toolConfig !== null) body.toolConfig = toolConfig;
  const config: Json = {};
  if (maxTokens !== null) config.maxOutputTokens = maxTokens;
  if (temperature !== null) config.temperature = temperature;
  if (topP !== null) config.topP = topP;
  if (stops.length > 0) config.stopSequences = stops;
  if (jsonMode) config.responseMimeType = 'application/json';
  if (Object.keys(config).length > 0) body.generationConfig = config;
  return body;
}

function openAIToolsToDeclarations(body: Json): Json[] {
  if (!Array.isArray(body.tools)) return [];
  return body.tools.flatMap((raw) => {
    if (!isRecord(raw) || !isRecord(raw.function) || typeof raw.function.name !== 'string') return [];
    return [
      {
        name: raw.function.name,
        description: typeof raw.function.description === 'string' ? raw.function.description : '',
        parameters: isRecord(raw.function.parameters)
          ? raw.function.parameters
          : { type: 'object', properties: {} },
      },
    ];
  });
}

function anthropicToolsToDeclarations(body: Json): Json[] {
  if (!Array.isArray(body.tools)) return [];
  return body.tools.flatMap((raw) => {
    if (!isRecord(raw) || typeof raw.name !== 'string') return [];
    return [
      {
        name: raw.name,
        description: typeof raw.description === 'string' ? raw.description : '',
        parameters: isRecord(raw.input_schema) ? raw.input_schema : { type: 'object', properties: {} },
      },
    ];
  });
}

function stopListOf(v: unknown): string[] {
  if (typeof v === 'string') return v === '' ? [] : [v];
  if (Array.isArray(v)) return v.filter((x): x is string => typeof x === 'string');
  return [];
}

/**
 * An OpenAI Chat Completions request as a Gemini generateContent body. The model
 * itself goes in the URL path, not the body. System messages become the system
 * instruction, image_url parts become inline data, assistant tool_calls become
 * functionCall parts, tool messages become functionResponse parts, and
 * response_format json_object becomes responseMimeType application/json.
 */
export function openAIToGemini(body: Json): Json {
  const systemParts: string[] = [];
  const contents: Json[] = [];
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    const role = raw.role;
    const content = raw.content;
    if (role === 'system') {
      const text = textOf(content);
      if (text !== '') systemParts.push(text);
      continue;
    }
    if (role === 'assistant') {
      const parts: Json[] = [];
      const text = textOf(content);
      if (text !== '') parts.push({ text });
      const calls = Array.isArray(raw.tool_calls) ? raw.tool_calls : [];
      for (const call of calls) {
        if (!isRecord(call) || !isRecord(call.function)) continue;
        parts.push({
          functionCall: {
            name: typeof call.function.name === 'string' ? call.function.name : '',
            args: argumentsOf(call.function.arguments),
          },
        });
      }
      if (parts.length > 0) contents.push({ role: 'model', parts });
      continue;
    }
    if (role === 'tool') {
      const text = typeof content === 'string' ? content : textOf(content);
      contents.push({
        role: 'user',
        parts: [
          {
            functionResponse: {
              name: typeof raw.name === 'string' ? raw.name : '',
              response: { content: text },
            },
          },
        ],
      });
      continue;
    }
    const parts: Json[] = [];
    if (typeof content === 'string') parts.push({ text: content });
    else if (Array.isArray(content)) {
      for (const part of content) {
        if (!isRecord(part)) continue;
        const geminiPart = openAIGeminiPartOf(part);
        if (geminiPart !== null) parts.push(geminiPart);
      }
    }
    if (parts.length > 0) contents.push({ role: 'user', parts });
  }
  const maxTokens =
    typeof body.max_tokens === 'number'
      ? body.max_tokens
      : typeof body.max_completion_tokens === 'number'
        ? body.max_completion_tokens
        : null;
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  return geminiBody(
    contents,
    systemParts.join('\n'),
    openAIToolsToDeclarations(body),
    body.tool_choice,
    maxTokens,
    typeof body.temperature === 'number' ? body.temperature : null,
    typeof body.top_p === 'number' ? body.top_p : null,
    stopListOf(body.stop),
    format === 'json_object' || format === 'json_schema',
  );
}

/**
 * An Anthropic Messages request as a Gemini generateContent body. The system
 * field becomes the system instruction, image blocks become inline data,
 * tool_use blocks become functionCall parts, tool_result blocks become
 * functionResponse parts. Anthropic carries no JSON mode, so none is set.
 */
export function anthropicToGemini(body: Json): Json {
  const systemText = textOf(body.system);
  const contents: Json[] = [];
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    const assistant = raw.role === 'assistant';
    const content = raw.content;
    if (typeof content === 'string') {
      contents.push({ role: assistant ? 'model' : 'user', parts: [{ text: content }] });
      continue;
    }
    if (!Array.isArray(content)) continue;
    if (assistant) {
      const parts: Json[] = [];
      const text = textOf(content);
      if (text !== '') parts.push({ text });
      for (const block of content) {
        if (!isRecord(block) || block.type !== 'tool_use') continue;
        parts.push({
          functionCall: {
            name: typeof block.name === 'string' ? block.name : '',
            args: isRecord(block.input) ? block.input : {},
          },
        });
      }
      if (parts.length > 0) contents.push({ role: 'model', parts });
      continue;
    }
    const parts: Json[] = [];
    for (const block of content) {
      if (!isRecord(block)) continue;
      if (block.type === 'tool_result') {
        parts.push({
          functionResponse: {
            name: typeof block.name === 'string' ? block.name : '',
            response: { content: toolResultText(block.content) },
          },
        });
        continue;
      }
      const part = anthropicGeminiPartOf(block);
      if (part !== null) parts.push(part);
    }
    if (parts.length > 0) contents.push({ role: 'user', parts });
  }
  return geminiBody(
    contents,
    systemText,
    anthropicToolsToDeclarations(body),
    body.tool_choice,
    typeof body.max_tokens === 'number' ? body.max_tokens : null,
    typeof body.temperature === 'number' ? body.temperature : null,
    typeof body.top_p === 'number' ? body.top_p : null,
    stopListOf(body.stop_sequences),
    false,
  );
}

// ---- provider answers, normalised ----

/** An OpenAI Chat Completions JSON answer, normalised. */
export function parseOpenAIAnswer(body: Json): ProviderAnswer | null {
  const choices = Array.isArray(body.choices) ? body.choices : [];
  const choice = isRecord(choices[0]) ? choices[0] : null;
  if (choice === null) return null;
  const message = isRecord(choice.message) ? choice.message : {};
  const text = typeof message.content === 'string' ? message.content : anthropicTextOf(message.content);
  const usage = isRecord(body.usage) ? body.usage : {};
  return {
    text,
    inputTokens: numberOrZero(usage.prompt_tokens),
    outputTokens: numberOrZero(usage.completion_tokens),
    stop: typeof choice.finish_reason === 'string' ? choice.finish_reason : 'stop',
  };
}

/** An Anthropic Messages JSON answer, normalised. */
export function parseAnthropicAnswer(body: Json): ProviderAnswer | null {
  if (!Array.isArray(body.content)) return null;
  const text = anthropicTextOf(body.content);
  const usage = isRecord(body.usage) ? body.usage : {};
  return {
    text,
    inputTokens: numberOrZero(usage.input_tokens),
    outputTokens: numberOrZero(usage.output_tokens),
    stop: typeof body.stop_reason === 'string' ? body.stop_reason : 'end_turn',
  };
}

/** A Gemini generateContent JSON answer, normalised. */
export function parseGeminiAnswer(body: Json): ProviderAnswer | null {
  const candidates = Array.isArray(body.candidates) ? body.candidates : [];
  const candidate = isRecord(candidates[0]) ? candidates[0] : null;
  if (candidate === null) return null;
  const content = isRecord(candidate.content) ? candidate.content : {};
  const parts = Array.isArray(content.parts) ? content.parts : [];
  const texts: string[] = [];
  for (const part of parts) {
    if (isRecord(part) && typeof part.text === 'string') texts.push(part.text);
  }
  const usage = isRecord(body.usageMetadata) ? body.usageMetadata : {};
  return {
    text: texts.join(''),
    inputTokens: numberOrZero(usage.promptTokenCount),
    outputTokens: numberOrZero(usage.candidatesTokenCount),
    stop: typeof candidate.finishReason === 'string' ? candidate.finishReason : 'STOP',
  };
}

// ---- SSE: parse one provider stream into the normalised answer ----

export interface SseFrame {
  event: string | null;
  data: string;
}

/** One SSE text as its frames: an optional event name and the data lines joined. */
export function parseSseFrames(text: string): SseFrame[] {
  const frames: SseFrame[] = [];
  const normalized = text.replace(/\r\n/g, '\n');
  for (const frame of normalized.split('\n\n')) {
    if (frame.trim() === '') continue;
    let event: string | null = null;
    const data: string[] = [];
    for (const line of frame.split('\n')) {
      if (line.startsWith('event:')) event = line.slice('event:'.length).trim();
      else if (line.startsWith('data:')) data.push(line.slice('data:'.length).replace(/^ /, ''));
    }
    if (data.length > 0) frames.push({ event, data: data.join('\n') });
  }
  return frames;
}

function jsonOf(payload: string): Json | null {
  try {
    const value: unknown = JSON.parse(payload);
    return isRecord(value) ? value : null;
  } catch {
    return null;
  }
}

/** An OpenAI event stream, normalised: text deltas, the finish chunk, the usage chunk. */
export function parseOpenAIStream(text: string): ProviderAnswer {
  let out = '';
  let input = 0;
  let output = 0;
  let stop = 'stop';
  for (const frame of parseSseFrames(text)) {
    const payload = frame.data.trim();
    if (payload === '[DONE]') continue;
    const chunk = jsonOf(payload);
    if (chunk === null) continue;
    if (isRecord(chunk.usage)) {
      input = numberOrZero(chunk.usage.prompt_tokens);
      output = numberOrZero(chunk.usage.completion_tokens);
    }
    const choices = Array.isArray(chunk.choices) ? chunk.choices : [];
    const choice = isRecord(choices[0]) ? choices[0] : null;
    if (choice === null) continue;
    const delta = isRecord(choice.delta) ? choice.delta : {};
    if (typeof delta.content === 'string') out += delta.content;
    if (typeof choice.finish_reason === 'string') stop = choice.finish_reason;
  }
  return { text: out, inputTokens: input, outputTokens: output, stop };
}

/** An Anthropic event stream, normalised. */
export function parseAnthropicStream(text: string): ProviderAnswer {
  let out = '';
  let input = 0;
  let output = 0;
  let stop = 'end_turn';
  for (const frame of parseSseFrames(text)) {
    const data = jsonOf(frame.data);
    if (data === null) continue;
    if (frame.event === 'message_start' && isRecord(data.message) && isRecord(data.message.usage))
      input = numberOrZero(data.message.usage.input_tokens);
    if (frame.event === 'content_block_delta' && isRecord(data.delta)) {
      if (data.delta.type === 'text_delta' && typeof data.delta.text === 'string') out += data.delta.text;
    }
    if (frame.event === 'message_delta') {
      if (isRecord(data.delta) && typeof data.delta.stop_reason === 'string') stop = data.delta.stop_reason;
      if (isRecord(data.usage)) output = numberOrZero(data.usage.output_tokens);
    }
  }
  return { text: out, inputTokens: input, outputTokens: output, stop };
}

/** A Gemini event stream, normalised: text pieces, then the final frame's stop + usage. */
export function parseGeminiStream(text: string): ProviderAnswer {
  let out = '';
  let input = 0;
  let output = 0;
  let stop = 'STOP';
  for (const frame of parseSseFrames(text)) {
    const data = jsonOf(frame.data);
    if (data === null) continue;
    const candidates = Array.isArray(data.candidates) ? data.candidates : [];
    const candidate = isRecord(candidates[0]) ? candidates[0] : null;
    if (candidate !== null) {
      const content = isRecord(candidate.content) ? candidate.content : {};
      const parts = Array.isArray(content.parts) ? content.parts : [];
      for (const part of parts) {
        if (isRecord(part) && typeof part.text === 'string') out += part.text;
      }
      if (typeof candidate.finishReason === 'string') stop = candidate.finishReason;
    }
    if (isRecord(data.usageMetadata)) {
      input = numberOrZero(data.usageMetadata.promptTokenCount);
      output = numberOrZero(data.usageMetadata.candidatesTokenCount);
    }
  }
  return { text: out, inputTokens: input, outputTokens: output, stop };
}

// ---- SSE: emit one normalised answer in the client's wire ----

function splitText(text: string): string[] {
  const pieces: string[] = [];
  for (let at = 0; at < text.length; at += SSE_DELTA_CHARS) pieces.push(text.slice(at, at + SSE_DELTA_CHARS));
  return pieces.length === 0 ? [''] : pieces;
}

function sseEvent(event: string | null, data: unknown): string {
  return `${event === null ? '' : `event: ${event}\n`}data: ${JSON.stringify(data)}\n\n`;
}

/** The normalised answer as an OpenAI event stream, with usage and [DONE]. */
export function openAISse(answer: ProviderAnswer, model: string): string {
  const finish = stopToOpenAI(answer.stop);
  const frames: string[] = [
    sseEvent(null, {
      id: 'chatcmpl-agg',
      object: 'chat.completion.chunk',
      created: FIXED_OPENAI_CREATED,
      model,
      choices: [{ index: 0, delta: { role: 'assistant', content: '' }, finish_reason: null }],
    }),
  ];
  for (const piece of splitText(answer.text)) {
    frames.push(
      sseEvent(null, {
        id: 'chatcmpl-agg',
        object: 'chat.completion.chunk',
        created: FIXED_OPENAI_CREATED,
        model,
        choices: [{ index: 0, delta: { content: piece }, finish_reason: null }],
      }),
    );
  }
  frames.push(
    sseEvent(null, {
      id: 'chatcmpl-agg',
      object: 'chat.completion.chunk',
      created: FIXED_OPENAI_CREATED,
      model,
      choices: [{ index: 0, delta: {}, finish_reason: finish }],
    }),
    sseEvent(null, {
      id: 'chatcmpl-agg',
      object: 'chat.completion.chunk',
      created: FIXED_OPENAI_CREATED,
      model,
      choices: [],
      usage: {
        prompt_tokens: answer.inputTokens,
        completion_tokens: answer.outputTokens,
        total_tokens: answer.inputTokens + answer.outputTokens,
      },
    }),
    'data: [DONE]\n\n',
  );
  return frames.join('');
}

/** The normalised answer as an Anthropic event stream. */
export function anthropicSse(answer: ProviderAnswer, model: string): string {
  const frames: string[] = [
    sseEvent('message_start', {
      type: 'message_start',
      message: {
        id: 'msg_agg',
        type: 'message',
        role: 'assistant',
        model,
        content: [],
        stop_reason: null,
        stop_sequence: null,
        usage: { input_tokens: answer.inputTokens, output_tokens: 0 },
      },
    }),
    sseEvent('content_block_start', {
      type: 'content_block_start',
      index: 0,
      content_block: { type: 'text', text: '' },
    }),
  ];
  for (const piece of splitText(answer.text)) {
    frames.push(
      sseEvent('content_block_delta', {
        type: 'content_block_delta',
        index: 0,
        delta: { type: 'text_delta', text: piece },
      }),
    );
  }
  frames.push(
    sseEvent('content_block_stop', { type: 'content_block_stop', index: 0 }),
    sseEvent('message_delta', {
      type: 'message_delta',
      delta: { stop_reason: stopToAnthropic(answer.stop), stop_sequence: null },
      usage: { output_tokens: answer.outputTokens },
    }),
    sseEvent('message_stop', { type: 'message_stop' }),
  );
  return frames.join('');
}

// ---- JSON answers in the client's wire ----

/** The normalised answer as an OpenAI Chat Completions object. */
export function openAIJson(answer: ProviderAnswer, model: string): Json {
  return {
    id: 'chatcmpl-agg',
    object: 'chat.completion',
    created: FIXED_OPENAI_CREATED,
    model,
    choices: [
      {
        index: 0,
        message: { role: 'assistant', content: answer.text },
        finish_reason: stopToOpenAI(answer.stop),
        logprobs: null,
      },
    ],
    usage: {
      prompt_tokens: answer.inputTokens,
      completion_tokens: answer.outputTokens,
      total_tokens: answer.inputTokens + answer.outputTokens,
    },
  };
}

/** The normalised answer as an Anthropic Messages object. */
export function anthropicJson(answer: ProviderAnswer, model: string): Json {
  return {
    id: 'msg_agg',
    type: 'message',
    role: 'assistant',
    model,
    content: [{ type: 'text', text: answer.text }],
    stop_reason: stopToAnthropic(answer.stop),
    stop_sequence: null,
    usage: { input_tokens: answer.inputTokens, output_tokens: answer.outputTokens },
  };
}

