/**
 * Integration tests for llguidance-js
 *
 * Note: These tests require the WASM module to be built first.
 * Run `npm run build:wasm` before running these tests.
 */

import { describe, it, expect, vi, beforeAll } from 'vitest';
import { GuidanceLogitsProcessor } from '../src/processor';
import { extractTokenizerData } from '../src/tokenizer-bridge';

import * as wasm from '../pkg/llguidance_wasm.js';

describe('Integration Tests', () => {
  describe('WASM Module', () => {
    beforeAll(async () => {
      wasm.init();
    });

    it('should create parser with regex grammar', async () => {
      const grammar = JSON.stringify({
        grammars: [{ rx: '[a-z]+' }],
      });

      const tokenizer = JSON.stringify({
        vocab: { a: 0, b: 1, c: 2 },
      });

      const parser = new wasm.LLGuidanceParser(grammar, tokenizer);
      expect(parser.vocab_size()).toBeGreaterThan(0);
    });

    it('should create parser with Lark grammar', async () => {
      const grammar = JSON.stringify({
        grammars: [{ lark: 'start: "hello"' }],
      });

      const tokenizer = JSON.stringify({
        vocab: { h: 0, e: 1, l: 2, o: 3 },
      });

      const parser = new wasm.LLGuidanceParser(grammar, tokenizer);
      expect(parser.vocab_size()).toBeGreaterThan(0);
    });

    it('should create parser with JSON schema', async () => {
      const grammar = JSON.stringify({
        grammars: [
          {
            json_schema: {
              type: 'object',
              properties: {
                name: { type: 'string' },
              },
              required: ['name'],
            },
          },
        ],
      });

      const tokenizer = JSON.stringify({
        vocab: { '{': 0, '}': 1, '"': 2, ':': 3, n: 4, a: 5, m: 6, e: 7 },
      });

      const parser = new wasm.LLGuidanceParser(grammar, tokenizer);
      expect(parser.vocab_size()).toBeGreaterThan(0);
    });

    it('should get token mask', async () => {
      const grammar = JSON.stringify({
        grammars: [{ lark: 'start: "a"' }],
      });

      const tokenizer = JSON.stringify({
        vocab: { a: 0 },
      });

      const parser = new wasm.LLGuidanceParser(grammar, tokenizer);
      const mask = parser.get_token_mask();

      expect(mask).toBeInstanceOf(Uint8Array);
      expect(mask.length).toBe(parser.vocab_size());
    });

    it('should check if token is allowed', async () => {
      const grammar = JSON.stringify({
        grammars: [{ lark: 'start: "a"' }],
      });

      const tokenizer = JSON.stringify({
        vocab: { a: 0 },
      });

      const parser = new wasm.LLGuidanceParser(grammar, tokenizer);

      // Should return a boolean
      const allowed = parser.is_token_allowed(0);
      expect(typeof allowed).toBe('boolean');
    });

    it('should advance parser state', async () => {
      const grammar = JSON.stringify({
        grammars: [{ lark: 'start: "ab"' }],
      });

      const tokenizer = JSON.stringify({
        vocab: { a: 0, b: 1 },
      });

      const parser = new wasm.LLGuidanceParser(grammar, tokenizer);

      // Advance with token 0 ('a') should not throw
      expect(() => parser.advance(0)).not.toThrow();
    });

    it('should reset parser', async () => {
      const grammar = JSON.stringify({
        grammars: [{ lark: 'start: "a"' }],
      });

      const tokenizer = JSON.stringify({
        vocab: { a: 0 },
      });

      const parser = new wasm.LLGuidanceParser(grammar, tokenizer);

      // Reset with new grammar should work
      const newGrammar = JSON.stringify({
        grammars: [{ lark: 'start: "b"' }],
      });

      expect(() => parser.reset(newGrammar)).not.toThrow();
    });

    it('should report stop reason', async () => {
      const grammar = JSON.stringify({
        grammars: [{ lark: 'start: "a"' }],
      });

      const tokenizer = JSON.stringify({
        vocab: { a: 0 },
      });

      const parser = new wasm.LLGuidanceParser(grammar, tokenizer);
      const stopReason = parser.stop_reason();

      expect(typeof stopReason).toBe('string');
    });
  });

  describe('Simulated integration flow', () => {
    // These tests simulate the integration flow without real models
    // They verify the API contract and data flow

    it('should correctly wire parser to processor', () => {
      // Create mock parser
      const mockParser = {
        isTokenAllowed: vi.fn().mockImplementation((id: number) => id % 2 === 0),
        getTokenMask: vi.fn().mockReturnValue(new Uint8Array(100).fill(1)),
        advance: vi.fn(),
        isComplete: vi.fn().mockReturnValue(false),
        reset: vi.fn(),
        vocabSize: 100,
      };

      // Simulate generation loop
      const logits = new Float32Array(100);
      for (let i = 0; i < 100; i++) {
        logits[i] = Math.random() * 10 - 5;
      }

      // Use the imported processor with mock parser
      const processor = new GuidanceLogitsProcessor(mockParser as any, {
        speculationDepth: 5,
      });

      // Process logits
      const processedLogits = processor.process([], logits);

      // Verify masking happened
      expect(processedLogits).toBeInstanceOf(Float32Array);
      expect(processedLogits.length).toBe(100);

      // At least one token should have a finite value
      const finiteCount = Array.from(processedLogits).filter((x) =>
        isFinite(x),
      ).length;
      expect(finiteCount).toBeGreaterThan(0);
    });

    it('should handle the complete generation lifecycle', () => {
      // Mock parser that accepts a simple sequence
      let state = 0;
      const validSequence = [10, 20, 30]; // Expected token sequence

      const mockParser = {
        isTokenAllowed: vi.fn().mockImplementation((id: number) => {
          if (state < validSequence.length) {
            return id === validSequence[state];
          }
          return false;
        }),
        getTokenMask: vi.fn().mockImplementation(() => {
          const mask = new Uint8Array(100).fill(0);
          if (state < validSequence.length) {
            mask[validSequence[state]] = 1;
          }
          return mask;
        }),
        advance: vi.fn().mockImplementation(() => {
          state++;
        }),
        isComplete: vi
          .fn()
          .mockImplementation(() => state >= validSequence.length),
        reset: vi.fn().mockImplementation(() => {
          state = 0;
        }),
        vocabSize: 100,
      };

      const processor = new GuidanceLogitsProcessor(mockParser as any, {
        speculationDepth: 1,
      });

      // Simulate generation loop
      const generatedTokens: number[] = [];
      let iterations = 0;
      const maxIterations = 10;

      while (!processor.canStop() && iterations < maxIterations) {
        const logits = new Float32Array(100);
        logits.fill(-10);
        // Put high probability on the correct next token
        if (state < validSequence.length) {
          logits[validSequence[state]] = 10;
        }

        const processedLogits = processor.process([], logits);

        // Find argmax (simulating sampling)
        let maxIdx = 0;
        let maxVal = processedLogits[0];
        for (let i = 1; i < processedLogits.length; i++) {
          if (processedLogits[i] > maxVal) {
            maxVal = processedLogits[i];
            maxIdx = i;
          }
        }

        generatedTokens.push(maxIdx);
        processor.onToken(maxIdx);
        iterations++;
      }

      expect(generatedTokens).toEqual(validSequence);
      expect(processor.canStop()).toBe(true);
    });
  });

  describe('TokenizerData compatibility', () => {
    it('should produce valid JSON for WASM consumption', () => {
      const mockTokenizer = {
        getVocab: () => ({
          '<s>': 0,
          '</s>': 1,
          '<unk>': 2,
          hello: 3,
          world: 4,
        }),
        model: {
          merges: ['h e', 'l l', 'o _'],
        },
        added_tokens: [
          { id: 0, content: '<s>', special: true },
          { id: 1, content: '</s>', special: true },
        ],
      };

      const data = extractTokenizerData(mockTokenizer);

      // Should be JSON-serializable
      const json = JSON.stringify(data);
      expect(json).toBeDefined();

      // Should be parseable back
      const parsed = JSON.parse(json);
      expect(parsed.vocab).toEqual(data.vocab);
      expect(parsed.merges).toEqual(data.merges);
    });
  });
});
