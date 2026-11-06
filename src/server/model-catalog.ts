/**
 * The model catalog's server surface (model-control wave, 2026-09-12;
 * federated since the model-slots wave, 2026-09-20): pi-ai's catalog for
 * every KEYED provider, projected to client-safe DTOs for the picker (GET
 * /api/models), plus the validation gate for settings writes (PUT
 * /api/settings/* — the per-chat settings PUT is retired from the UI;
 * legacy records' per-chat overrides still resolve server-side).
 *
 * pi-ai owns ALL the data: the catalog (id/name/cost/context/modality/
 * thinkingLevelMap) and the per-model thinking truth
 * (getSupportedThinkingLevels / clampThinkingLevel). This module only
 * projects and validates — no second registry to drift out of sync.
 *
 * Federation: `getModels()` is called per keyed provider (federated, the
 * model-slots wave), so a provider appears in the picker exactly when its
 * key is present — piModels() in src/agent/models.ts owns the provider
 * set; nothing here changes.
 */

import { clampThinkingLevel, getSupportedThinkingLevels, type Api, type Model } from "@earendil-works/pi-ai";
import type { ThinkingLevel } from "@earendil-works/pi-agent-core";
import { clearLocalProvider, createLocalProvider, piModels, setLocalProvider } from "../agent/models.ts";
import { probeOllama, probeVLLM } from "./local-models.ts";
import { chatGPTConnected, chatGPTModel, chatGPTModels, refreshChatGPTModels } from "../agent/chatgpt-bridge.ts";
import { NATIVE_SUBSCRIPTIONS } from "../core/contracts/subscriptions.ts";
import { nativeSubscriptionId, nativeSubscriptionConnected, nativeSubscriptionModel, nativeSubscriptionModels, refreshSubscriptionModels } from "../agent/subscription-bridge.ts";
import { settingsStore } from "../store/settings.ts";
import { effectiveSettings } from "./stores.ts";
import { keyValue } from "./keys.ts";
import type { ChatSettings } from "../core/contracts/chat.ts";
import type {
	CatalogDefaults,
	CatalogProvider,
	ModelCatalogEntry,
	ModelCatalogResponse,
	RolePins,
	SlotPins,
} from "../core/contracts/models.ts";
import type { EffectiveModelSpec, EffectiveSlot } from "../core/settings-resolution.ts";
import { AUTH_PROVIDER_KEYS } from "../core/auth-keys.ts";
import { BACKGROUND_MODEL_ROLES, MODEL_SLOTS, type BackgroundModelRole, type ModelSlotName, type ModelsSettings } from "../core/settings.ts";

/**
 * Re-probe local providers. The contextWindowOverride from settings.json
 * (LM Studio / local-model fix, 2026-11-04) is passed through to the
 * probe — when set it wins over anything the endpoint reports, which is
 * exactly what LM Studio needs because its `/v1/models` never reports
 * context length. Setting it once in settings.json and restarting fixes
 * the app's picture of a 31k model that was silently being treated as
 * "no idea, use default".
 */
export async function refreshLocalModels(provider?: string): Promise<void> {
	const settings = effectiveSettings().localProviders;
	const tasks: Promise<void>[] = [];

	if (!provider || provider === "ollama") {
		tasks.push((async () => {
			if (settings.ollama.enabled) {
				const res = await probeOllama(
					settings.ollama.baseUrl,
					settings.ollama.defaultContextWindow,
					settings.ollama.contextWindowOverride,
				);
				if (res.ok) {
					setLocalProvider(createLocalProvider("ollama", settings.ollama.baseUrl, undefined, res.models));
				}
			} else {
				clearLocalProvider("ollama");
			}
		})());
	}

	if (!provider || provider === "custom") {
		tasks.push((async () => {
			if (settings.custom.enabled) {
				const apiKey = keyValue("LOCAL_AI_API_KEY");
				const res = await probeVLLM(
					settings.custom.baseUrl,
					apiKey,
					settings.custom.defaultContextWindow,
					settings.custom.contextWindowOverride,
				);
				if (res.ok) {
					setLocalProvider(createLocalProvider("custom", settings.custom.baseUrl, apiKey, res.models));
				}
			} else {
				clearLocalProvider("custom");
			}
		})());
	}

	await Promise.all(tasks);
}

export async function refreshConnectionCatalog(provider?: string): Promise<void> {
	await Promise.all([
		!provider || provider === "chatgpt" ? refreshChatGPTModels(piModels().getModels("openai").filter((m): m is Model<"openai-responses"> => m.api === "openai-responses")) : Promise.resolve(),
		!provider || nativeSubscriptionId(provider) ? refreshSubscriptionModels() : Promise.resolve(),
		!provider || provider === "ollama" || provider === "custom" ? refreshLocalModels(provider) : Promise.resolve(),
	]);
}

function connectionModel(provider: string, id: string): Model<Api> | undefined {
	return provider === "chatgpt" ? chatGPTModel(id) : nativeSubscriptionId(provider) ? nativeSubscriptionModel(provider, id) : piModels().getModel(provider, id);
}

function connectionProviderIds(): string[] {
	return [...piModels().getProviders().map((p) => p.id), "chatgpt", ...NATIVE_SUBSCRIPTIONS.map((p) => p.logicalId)];
}

/** The picker's client-safe model entry (contract: core/contracts/models.ts). */
export type ModelDto = ModelCatalogEntry;

export function catalogDefaultsFor(
	strong: EffectiveSlot | null,
): CatalogDefaults | null {
	const model = strong ? { provider: strong.provider, id: strong.id } : undefined;
	if (!model) return null;
	const level = strong?.thinkingLevel;
	return {
		model,
		...(level !== undefined ? { thinkingLevel: level } : {}),
	};
}

export function catalogDefaults(): CatalogDefaults | null {
	return catalogDefaultsFor(effectiveSettings().models.strong);
}

export function rolePinsFor(
	models: Record<BackgroundModelRole, EffectiveModelSpec | undefined>,
): RolePins {
	const pins = {} as RolePins;
	for (const role of BACKGROUND_MODEL_ROLES) {
		const spec = models[role];
		pins[role] = spec
			? {
					model: { provider: spec.provider, id: spec.id },
					...(spec.thinkingLevel !== undefined ? { thinkingLevel: spec.thinkingLevel } : {}),
				}
			: null;
	}
	return pins;
}

export function catalogRolePins(): RolePins {
	return rolePinsFor(effectiveSettings().agent.models);
}

export function catalogSlots(): SlotPins {
	const slots = effectiveSettings().models;
	const pins = {} as SlotPins;
	for (const name of ["strong", "cheap", "vision"] as const) {
		const slot: EffectiveSlot | null = slots[name];
		pins[name] = slot
			? {
					model: { provider: slot.provider, id: slot.id },
					...(slot.thinkingLevel !== undefined ? { thinkingLevel: slot.thinkingLevel } : {}),
				}
			: null;
	}
	return pins;
}

export function catalogProviders(): CatalogProvider[] {
	const local = effectiveSettings().localProviders;
	return AUTH_PROVIDER_KEYS.map((p) => ({
		id: p.id,
		envVar: p.envVar,
		keyPresent: keyValue(p.envVar) !== undefined,
	})).concat([
		{ id: "chatgpt", envVar: "", keyPresent: chatGPTConnected() },
		...NATIVE_SUBSCRIPTIONS.map((p) => ({ id: p.logicalId, envVar: "", keyPresent: nativeSubscriptionConnected(p.providerId) })),
		{ id: "ollama", envVar: "", keyPresent: local.ollama.enabled },
		{ id: "custom", envVar: "LOCAL_AI_API_KEY", keyPresent: local.custom.enabled },
	]).sort((a, b) => a.id.localeCompare(b.id));
}

function keyedProviderIds(): Set<string> {
	return new Set(catalogProviders().filter((p) => p.keyPresent).map((p) => p.id));
}

/** Project one catalog model to the client DTO. */
export function toModelDto(model: Model<Api>): ModelDto {
	const supported = getSupportedThinkingLevels(model);
	const isLocal = model.provider === "ollama" || model.provider === "custom";
	const supportedLevels =
		model.reasoning && isLocal && supported.length <= 1 && supported[0] === "off"
			? (["off", "low", "medium", "high"] as ThinkingLevel[])
			: supported;

	return {
		id: model.id,
		name: model.name,
		provider: model.provider,
		reasoning: model.reasoning,
		input: [...model.input],
		supportedLevels,
		cost: {
			input: model.cost.input,
			output: model.cost.output,
			cacheRead: model.cost.cacheRead,
			cacheWrite: model.cost.cacheWrite,
		},
		contextWindow: model.contextWindow,
		maxTokens: model.maxTokens,
	};
}

export function catalogContextLimits(): Map<string, number> {
	const keyed = keyedProviderIds();
	const index = new Map<string, number>();
	for (const provider of piModels().getProviders()) {
		if (!keyed.has(provider.id)) continue;
		for (const model of piModels().getModels(provider.id)) {
			index.set(`${provider.id}:${model.id}`, model.contextWindow);
		}
	}
	for (const model of [...chatGPTModels(), ...nativeSubscriptionModels()]) index.set(`${model.provider}:${model.id}`, model.contextWindow);
	return index;
}

export function modelCatalog(): ModelCatalogResponse {
	const keyed = keyedProviderIds();
	const models = piModels()
		.getProviders()
		.filter((p) => keyed.has(p.id))
		.flatMap((p) => piModels().getModels(p.id))
		.map(toModelDto)
		.sort((a, b) => (a.provider === b.provider ? a.id.localeCompare(b.id) : a.provider.localeCompare(b.provider)))
		.concat([...chatGPTModels(), ...nativeSubscriptionModels()].map(toModelDto));
	return {
		models,
		defaults: catalogDefaults(),
		rolePins: catalogRolePins(),
		slots: catalogSlots(),
		providers: catalogProviders(),
	};
}

export type SettingsValidation =
	| { ok: true }
	| { ok: false; error: string; supported?: ThinkingLevel[] };

export function validateChatSettings(settings: ChatSettings): SettingsValidation {
	const providers = connectionProviderIds();
	if (settings.model) {
		const { provider, id } = settings.model;
		if (!providers.includes(provider)) {
			return { ok: false, error: `Unknown provider "${provider}" — configured: ${providers.join(", ")}` };
		}
		if (!connectionModel(provider, id)) {
			return { ok: false, error: `No model "${id}" in the ${provider} catalog` };
		}
	}
	return { ok: true };
}

export function validateModelsSettings(settings: ModelsSettings): SettingsValidation {
	for (const name of MODEL_SLOTS) {
		const slot = settings[name];
		if (!slot) continue;
		const verdict = validateSlotSpec(name, slot);
		if (!verdict.ok) return verdict;
	}
	return { ok: true };
}

function validateSlotSpec(name: ModelSlotName, slot: { model?: { provider: string; id: string }; thinkingLevel?: ThinkingLevel }): SettingsValidation {
	const providers = connectionProviderIds();
	if (slot.model) {
		const { provider, id } = slot.model;
		if (!providers.includes(provider)) {
			return { ok: false, error: `Unknown provider "${provider}" — configured: ${providers.join(", ")}` };
		}
		if (!connectionModel(provider, id)) {
			return { ok: false, error: `No model "${id}" in the ${provider} catalog` };
		}
	}
	return { ok: true };
}

/** Clamp one level against one model. For local models with reasoning enabled,
 * avoid clamping down to "off" if pi-ai lacks a built-in level map. */
function clampLevelFor(provider: string, id: string, level: ThinkingLevel): ThinkingLevel {
	const model = connectionModel(provider, id);
	if (!model) return level;
	if (model.reasoning && (provider === "ollama" || provider === "custom")) {
		return level;
	}
	return clampThinkingLevel(model, level);
}

export function clampModelsLevels(settings: ModelsSettings): ModelsSettings {
	const out: ModelsSettings = {};
	for (const name of MODEL_SLOTS) {
		const slot = settings[name];
		if (!slot?.model) continue;
		out[name] =
			slot.thinkingLevel !== undefined
				? {
						model: slot.model,
						thinkingLevel: clampLevelFor(slot.model.provider, slot.model.id, slot.thinkingLevel),
					}
				: slot;
	}
	return out;
}

export function clampChatLevel(settings: ChatSettings, strong: EffectiveSlot | null): ChatSettings {
	if (settings.thinkingLevel === undefined) return settings;
	const provider = settings.model?.provider ?? strong?.provider;
	const id = settings.model?.id ?? strong?.id;
	if (!provider || !id) return settings;
	return { ...settings, thinkingLevel: clampLevelFor(provider, id, settings.thinkingLevel) };
}
