import type { ModelCatalogEntry } from "../core/contracts/models.ts";
import { normalizeOllamaRootUrl, normalizeOpenAIBaseUrl } from "../core/settings.ts";

export interface OllamaCachedMetadata {
	contextWindow: number;
	isVision: boolean;
	isReasoning: boolean;
	isEmbeddingOnly: boolean;
}

export const ollamaShowCache = new Map<string, OllamaCachedMetadata>();

export const VISION_MODEL_RE = /(?<![a-z])vl(?![a-z])|vision|4o|pixtral|llava|moondream|mllama|minicpm|cogvlm|glm-4v|internvl/i;
export const REASONING_MODEL_RE = /(?<![a-z])r1(?![a-z])|reasoning|reasoner|think|qwq|qwen|gpt|deepseek|o1|o3|ministral/i;

export function clearOllamaShowCache(): void {
	ollamaShowCache.clear();
}

interface OllamaTagModel {
	name: string;
	digest?: string;
	details?: {
		family?: string;
		families?: string[];
	};
}

interface OllamaShowResponse {
	parameters?: string;
	capabilities?: string[];
	model_info?: Record<string, unknown>;
	details?: {
		family?: string;
		families?: string[];
	};
}

/** Extract context length from Ollama show response:
 * 1. num_ctx parameter (user or modelfile allocated)
 * 2. architectural context_length in model_info
 * 3. fallback default */
function extractOllamaContextWindow(show: OllamaShowResponse, fallback: number): number {
	if (typeof show.parameters === "string") {
		const match = /num_ctx\s+(\d+)/i.exec(show.parameters);
		if (match?.[1]) {
			const parsed = Number.parseInt(match[1], 10);
			if (Number.isFinite(parsed) && parsed > 0) return parsed;
		}
	}
	if (show.model_info && typeof show.model_info === "object") {
		for (const [key, val] of Object.entries(show.model_info)) {
			if (key.endsWith(".context_length") && typeof val === "number" && Number.isFinite(val) && val > 0) {
				return val;
			}
		}
	}
	return fallback;
}

/**
 * Probe an Ollama endpoint for its model list.
 *
 * contextWindowOverride (LM Studio / local-model fix, 2026-11-04): when
 * set, wins over everything the endpoint reports — the setting exists
 * because Ollama's `/api/show` and LM Studio's `/v1/models` routinely
 * disagree with what the model was actually LOADED with, and the app
 * must be able to be told the truth explicitly. The user sets this in
 * `settings.json` → `localProviders.<provider>.contextWindowOverride`.
 */
export async function probeOllama(
	rawUrl: string,
	defaultContextWindow: number = 32768,
	contextWindowOverride?: number | typeof fetch,
	fetchFn?: typeof fetch,
): Promise<{ ok: boolean; models: ModelCatalogEntry[]; error?: string }> {
	const override = typeof contextWindowOverride === "number" ? contextWindowOverride : undefined;
	const realFetch = typeof contextWindowOverride === "function" ? contextWindowOverride : (fetchFn ?? globalThis.fetch);
	const rootUrl = normalizeOllamaRootUrl(rawUrl);
	if (!rootUrl) return { ok: false, models: [], error: "Ollama URL is empty" };

	try {
		const tagsRes = await realFetch(`${rootUrl}/api/tags`, {
			method: "GET",
			signal: AbortSignal.timeout(2500),
		});
		if (!tagsRes.ok) {
			return { ok: false, models: [], error: `Ollama returned HTTP ${tagsRes.status}` };
		}

		const data = (await tagsRes.json()) as { models?: OllamaTagModel[] };
		const rawModels = Array.isArray(data.models) ? data.models : [];

		const inspected = await Promise.allSettled(
			rawModels.map(async (m) => {
				const cacheKey = `${m.name}:${m.digest ?? "no-digest"}`;
				const cached = ollamaShowCache.get(cacheKey);
				if (cached) return { model: m, meta: cached };

				try {
					const showRes = await realFetch(`${rootUrl}/api/show`, {
						method: "POST",
						headers: { "Content-Type": "application/json" },
						body: JSON.stringify({ name: m.name }),
						signal: AbortSignal.timeout(2000),
					});
					if (!showRes.ok) {
						const fallbackMeta: OllamaCachedMetadata = {
							contextWindow: defaultContextWindow,
							isVision: VISION_MODEL_RE.test(m.name),
							isReasoning: REASONING_MODEL_RE.test(m.name),
							isEmbeddingOnly: /bert|bge|embed/i.test(m.details?.family ?? "") || /nomic-embed|bge-|embed/i.test(m.name),
						};
						return { model: m, meta: fallbackMeta };
					}

					const show = (await showRes.json()) as OllamaShowResponse;
					const caps = show.capabilities;
					const isEmbeddingOnly =
						(Array.isArray(caps) && caps.length === 1 && caps[0] === "embedding") ||
						/bert|bge|embed/i.test(show.details?.family ?? m.details?.family ?? "") ||
						/nomic-embed|bge-|embed/i.test(m.name);

					const isVision = Boolean(caps?.includes("vision")) || VISION_MODEL_RE.test(m.name);
					const isReasoning = Boolean(caps?.includes("thinking")) || REASONING_MODEL_RE.test(m.name);
					const contextWindow = override ?? extractOllamaContextWindow(show, defaultContextWindow);

					const meta: OllamaCachedMetadata = {
						contextWindow,
						isVision,
						isReasoning,
						isEmbeddingOnly,
					};
					ollamaShowCache.set(cacheKey, meta);
					return { model: m, meta };
				} catch {
					const fallbackMeta: OllamaCachedMetadata = {
						contextWindow: override ?? defaultContextWindow,
						isVision: VISION_MODEL_RE.test(m.name),
						isReasoning: REASONING_MODEL_RE.test(m.name),
						isEmbeddingOnly: /bert|bge|embed/i.test(m.details?.family ?? "") || /nomic-embed|bge-|embed/i.test(m.name),
					};
					return { model: m, meta: fallbackMeta };
				}
			}),
		);

		const models: ModelCatalogEntry[] = [];
		for (const res of inspected) {
			if (res.status !== "fulfilled") continue;
			const { model, meta } = res.value;
			if (meta.isEmbeddingOnly) continue;

			const contextWindow = meta.contextWindow;
			const maxTokens = Math.min(4096, Math.max(512, Math.floor(contextWindow / 2)));

			models.push({
				id: model.name,
				name: `Ollama / ${model.name}`,
				provider: "ollama",
				reasoning: meta.isReasoning,
				input: meta.isVision ? ["text", "image"] : ["text"],
				supportedLevels: meta.isReasoning ? ["off", "low", "medium", "high"] : ["off"],
				cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
				contextWindow,
				maxTokens,
			});
		}

		models.sort((a, b) => a.id.localeCompare(b.id));
		return { ok: true, models };
	} catch (err) {
		const message = err instanceof Error ? err.message : String(err);
		return { ok: false, models: [], error: message };
	}
}

interface VLLMModelEntry {
	id: string;
	max_model_len?: number;
	context_length?: number;
	meta?: {
		n_ctx_train?: number;
	};
}

/**
 * Probe an OpenAI-compatible endpoint (vLLM, LM Studio, llama.cpp server,
 * …) for its model list.
 *
 * contextWindowOverride (LM Studio fix, 2026-11-04): LM Studio's
 * `/v1/models` returns only {id, object, owned_by} — no context-length
 * field. Without an explicit override the app silently falls back to
 * `defaultContextWindow` and every local load looks like 32k regardless
 * of what was actually loaded. Set the override in settings.json to
 * match the context length you loaded the model with in LM Studio's UI
 * (or `--context-length` for `lms`); the app then agrees with reality.
 */
export async function probeVLLM(
	rawUrl: string,
	apiKey?: string,
	defaultContextWindow: number = 32768,
	contextWindowOverride?: number | typeof fetch,
	fetchFn?: typeof fetch,
): Promise<{ ok: boolean; models: ModelCatalogEntry[]; error?: string }> {
	const override = typeof contextWindowOverride === "number" ? contextWindowOverride : undefined;
	const realFetch = typeof contextWindowOverride === "function" ? contextWindowOverride : (fetchFn ?? globalThis.fetch);
	const baseUrl = normalizeOpenAIBaseUrl(rawUrl);
	if (!baseUrl) return { ok: false, models: [], error: "Base URL is empty" };

	try {
		const headers: Record<string, string> = {};
		if (apiKey?.trim()) {
			headers.Authorization = `Bearer ${apiKey.trim()}`;
		}

		const res = await realFetch(`${baseUrl}/models`, {
			method: "GET",
			headers,
			signal: AbortSignal.timeout(2500),
		});

		if (!res.ok) {
			return { ok: false, models: [], error: `Endpoint returned HTTP ${res.status}` };
		}

		const data = (await res.json()) as { data?: VLLMModelEntry[]; models?: VLLMModelEntry[] };
		const rawModels = Array.isArray(data.data) ? data.data : Array.isArray(data.models) ? data.models : [];

		const models: ModelCatalogEntry[] = [];
		for (const entry of rawModels) {
			if (!entry?.id || typeof entry.id !== "string") continue;
			const reported =
				(typeof entry.max_model_len === "number" && entry.max_model_len > 0 ? entry.max_model_len : undefined) ??
				(typeof entry.context_length === "number" && entry.context_length > 0 ? entry.context_length : undefined) ??
				(typeof entry.meta?.n_ctx_train === "number" && entry.meta.n_ctx_train > 0 ? entry.meta.n_ctx_train : undefined);
			const contextWindow = override ?? reported ?? defaultContextWindow;

			const isVision = VISION_MODEL_RE.test(entry.id);
			const isReasoning = REASONING_MODEL_RE.test(entry.id);
			const maxTokens = Math.min(4096, Math.max(512, Math.floor(contextWindow / 2)));

			models.push({
				id: entry.id,
				name: `Custom / ${entry.id}`,
				provider: "custom",
				reasoning: isReasoning,
				input: isVision ? ["text", "image"] : ["text"],
				supportedLevels: isReasoning ? ["off", "low", "medium", "high"] : ["off"],
				cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
				contextWindow,
				maxTokens,
			});
		}

		models.sort((a, b) => a.id.localeCompare(b.id));
		return { ok: true, models };
	} catch (err) {
		const message = err instanceof Error ? err.message : String(err);
		return { ok: false, models: [], error: message };
	}
}
