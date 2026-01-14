import { describe, it, expect, beforeAll, vi } from 'vitest';
import type { TokenizerData, Grammar } from '../src/types';

// Mock the WASM module for unit tests
vi.mock('../pkg/llguidance_wasm', () => {
  return {
    default: vi.fn(),
    LLGuidanceParser: class MockLLGuidanceParser {
      is_token_allowed = vi.fn().mockReturnValue(true);
      get_token_mask = vi.fn().mockReturnValue(new Uint8Array(100).fill(1));
      advance = vi.fn();
      is_complete = vi.fn().mockReturnValue(false);
      reset = vi.fn();
      vocab_size = vi.fn().mockReturnValue(100);
    },
  };
});

// Import after mock setup
import { GuidanceParser } from '../src/parser';

describe('GuidanceParser', () => {
  const mockTokenizer: TokenizerData = {
    vocab: {
      hello: 0,
      world: 1,
      '!': 2,
      ' ': 3,
    },
    merges: [],
    added_tokens: [
      { id: 100, content: '<|endoftext|>', single_word: false, lstrip: false, rstrip: false, normalized: false, special: true },
    ],
  };

  const jsonSchemaGrammar: Grammar = {
    type: 'json_schema',
    schema: {
      type: 'object',
      properties: {
        name: { type: 'string' },
        age: { type: 'number' },
      },
      required: ['name', 'age'],
    },
  };

  const regexGrammar: Grammar = {
    type: 'regex',
    pattern: '[a-z]+',
  };

  const larkGrammar: Grammar = {
    type: 'lark',
    grammar: `
      start: greeting
      greeting: "hello" " " "world"
    `,
    startSymbol: 'start',
  };

  describe('create()', () => {
    it('should create a parser with JSON schema grammar', async () => {
      const parser = await GuidanceParser.create(jsonSchemaGrammar, mockTokenizer);
      expect(parser).toBeDefined();
      expect(parser.vocabSize).toBe(100);
    });

    it('should create a parser with regex grammar', async () => {
      const parser = await GuidanceParser.create(regexGrammar, mockTokenizer);
      expect(parser).toBeDefined();
    });

    it('should create a parser with Lark grammar', async () => {
      const parser = await GuidanceParser.create(larkGrammar, mockTokenizer);
      expect(parser).toBeDefined();
    });
  });

  describe('isTokenAllowed()', () => {
    it('should return true for allowed tokens', async () => {
      const parser = await GuidanceParser.create(regexGrammar, mockTokenizer);
      expect(parser.isTokenAllowed(0)).toBe(true);
    });
  });

  describe('getTokenMask()', () => {
    it('should return a Uint8Array mask', async () => {
      const parser = await GuidanceParser.create(regexGrammar, mockTokenizer);
      const mask = parser.getTokenMask();
      expect(mask).toBeInstanceOf(Uint8Array);
      expect(mask.length).toBe(100);
    });
  });

  describe('advance()', () => {
    it('should advance parser state without error', async () => {
      const parser = await GuidanceParser.create(regexGrammar, mockTokenizer);
      expect(() => parser.advance(0)).not.toThrow();
    });
  });

  describe('isComplete()', () => {
    it('should return completion status', async () => {
      const parser = await GuidanceParser.create(regexGrammar, mockTokenizer);
      expect(typeof parser.isComplete()).toBe('boolean');
    });
  });

  describe('reset()', () => {
    it('should reset parser state without error', async () => {
      const parser = await GuidanceParser.create(regexGrammar, mockTokenizer);
      parser.advance(0);
      expect(() => parser.reset()).not.toThrow();
    });
  });
});

