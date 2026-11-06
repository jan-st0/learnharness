import { describe, it } from "node:test";
import assert from "node:assert/strict";
import {
	MAX_SUMMARIZATION_TOKENS,
	validateSummary,
	buildCompactionPrompt,
	keepRecentTokensFor,
	type PlanTree,
} from "../src/core/index.ts";
import { effectiveThresholdTokens } from "../src/server/compaction.ts";
import { persistTurn, TurnPersistenceError } from "../src/server/turn-service.ts";
import type { ChatStore } from "../src/core/ports.ts";
import type { LhMessage } from "../src/core/native-messages.ts";

describe("Compaction summarizer token limits & prompt", () => {
	it("enforces MAX_SUMMARIZATION_TOKENS constant at 2800", () => {
		assert.equal(MAX_SUMMARIZATION_TOKENS, 2800);
	});

	it("compaction prompt instructs target range ~700-1100 tokens", () => {
		const prompt = buildCompactionPrompt({
			spanText: "Test span",
			planTreeText: null,
			kind: "teach",
		});
		assert.ok(prompt.includes("700–1100 tokens"));
		assert.ok(prompt.includes("1. Learner state"));
		assert.ok(prompt.includes("2. Teaching state"));
		assert.ok(prompt.includes("3. Plan and dependencies"));
		assert.ok(prompt.includes("4. Immediate continuation"));
	});
});

describe("Summary validation and rejection", () => {
	const sampleValidSummary = `## The learner's understanding
The learner understands basic vectors and matrices.

## The teaching arc
Covered vector spaces, then moved to inner products.

## The plan dependency tree
- [completed] Vectors (vec_intro)
- [current] Inner products (inner_prod)

## The current moment
Waiting for answer on orthogonal projections.`;

	const samplePlanTree: PlanTree = {
		version: 1,
		nodes: [
			{ id: "vec_intro", title: "Vectors" },
			{ id: "inner_prod", title: "Inner products" },
		],
		edges: [{ from: "vec_intro", to: "inner_prod" }],
	};

	it("accepts a complete, valid summary with plan nodes preserved", () => {
		const res = validateSummary(sampleValidSummary, samplePlanTree);
		assert.equal(res.ok, true);
	});

	it("rejects summary if stopReason was length (truncation)", () => {
		const res = validateSummary(sampleValidSummary, samplePlanTree, "length");
		assert.equal(res.ok, false);
		assert.ok(res.reason?.includes("length limit"));
	});

	it("rejects empty or whitespace-only summary", () => {
		const res1 = validateSummary("", samplePlanTree);
		assert.equal(res1.ok, false);
		const res2 = validateSummary("   \n\t  ", samplePlanTree);
		assert.equal(res2.ok, false);
	});

	it("rejects summary missing required headings", () => {
		const incomplete = `## The learner's understanding\nUnderstands vectors.\n\n## The teaching arc\nCovered basics.`;
		const res = validateSummary(incomplete, undefined);
		assert.equal(res.ok, false);
		assert.ok(res.reason?.includes("missing required section heading"));
	});

	it("rejects summary if any plan node ID is dropped", () => {
		const droppedNodeSummary = `## The learner's understanding
Understands vectors.

## The teaching arc
Covered basics.

## The plan dependency tree
- [completed] Vectors (vec_intro)
(second item missing)

## The current moment
Next step.`;

		const res = validateSummary(droppedNodeSummary, samplePlanTree);
		assert.equal(res.ok, false);
		assert.ok(res.reason?.includes("inner_prod"));
	});
});

describe("Small-window dynamic context retention (31,000 token window)", () => {
	it("scales threshold and retention properly for 31,000 window", () => {
		const window = 31_000;
		// 30% reserve of 31,000 is 9,300 tokens; 31,000 - 9,300 = 21,700
		const threshold = effectiveThresholdTokens(21_000, window);
		assert.equal(threshold, 21_000);

		const keptTail = keepRecentTokensFor(threshold);
		assert.equal(keptTail, 10_500); // exactly half the threshold
		assert.ok(keptTail < threshold, "kept tail is strictly below threshold");
		assert.ok(threshold + 9_300 <= window, "threshold + reserve fits inside context window");
	});
});

describe("Empty or 1-token length-stopped assistant response rejection", () => {
	const mockChatStore: ChatStore = {
		getChat: () => ({
			id: "c1",
			kind: "teach",
			title: "Test",
			created: new Date().toISOString(),
			messages: [],
			tree: undefined,
		} as any),
		saveMessages: () => {},
		createChat: () => ({ id: "c1" } as any),
		deleteChat: () => true,
		listChats: () => [],
		appendTurnUsage: () => {},
		appendExtractionUsage: () => {},
		saveTree: () => {},
		saveSettings: () => {},
		renameChat: () => ({ id: "c1" } as any),
	};

	it("rejects saving an empty assistant message that stopped due to length", () => {
		const emptyLengthMsg = {
			role: "assistant",
			content: [],
			stopReason: "length",
			usage: { input: 20000, output: 1 },
		};

		assert.throws(
			() => {
				persistTurn(mockChatStore, "c1", 0, [emptyLengthMsg as any], false);
			},
			(err: any) => {
				assert.ok(err instanceof TurnPersistenceError);
				assert.ok(err.message.includes("Token limit reached or context overflowed"));
				return true;
			},
		);
	});

	it("allows normal assistant response to persist cleanly", () => {
		const validMsg = {
			role: "assistant",
			content: [{ type: "text", text: "Hello! Let's continue." }],
			stopReason: "stop",
			usage: { input: 1000, output: 25 },
		};

		const saved = persistTurn(mockChatStore, "c1", 0, [validMsg as any], false);
		assert.equal(saved.length, 1);
		assert.equal(saved[0]?.role, "assistant");
	});
});
