/**
 * Integration tests with @huggingface/transformers
 *
 * These tests verify the integration between GuidanceLogitsProcessor and
 * transformer.js text generation pipeline.
 *
 * The llguidance WASM module now supports configurable tokenizers, allowing
 * integration with real BPE tokenizers like Qwen3.
 */

import { describe, it, expect, beforeAll, vi } from 'vitest';
import { GuidanceParser } from '../src/parser';
import { GuidanceLogitsProcessor } from '../src/processor';
import { extractTokenizerData } from '../src/tokenizer-bridge';
import type { GuidanceParser as GuidanceParserType } from '../src/parser';
import type { Grammar, TokenizerData } from '../src/types';

// We'll use dynamic imports to handle potential loading issues
let transformers: typeof import('@huggingface/transformers');

/**
 * Extract tokenizer data from a transformer.js tokenizer instance.
 * This is a version adapted specifically for transformer.js tokenizer structure.
 */
function extractTransformersTokenizerData(tokenizer: any): TokenizerData {
  // transformer.js stores vocab as tokens_to_ids Map
  const tokensToIds = tokenizer.model?.tokens_to_ids;

  if (!tokensToIds) {
    throw new Error('Unable to extract vocabulary from tokenizer');
  }

  // Convert Map to Record
  const vocab: Record<string, number> = {};
  if (tokensToIds instanceof Map) {
    for (const [token, id] of tokensToIds) {
      vocab[token] = id;
    }
  } else {
    Object.assign(vocab, tokensToIds);
  }

  // Get merges if available
  const merges = tokenizer.model?.merges ?? [];

  // Get added tokens
  const added_tokens = (tokenizer.added_tokens ?? []).map((token: any) => ({
    id: token.id,
    content: token.content,
    single_word: token.single_word ?? false,
    lstrip: token.lstrip ?? false,
    rstrip: token.rstrip ?? false,
    normalized: token.normalized ?? true,
    special: token.special ?? false,
  }));

  // Find EOS token ID (supports GPT-2, Qwen, Llama style tokens)
  let eos_token_id: number | undefined;
  const eosToken = added_tokens.find(
    (t: any) =>
      t.content === '<|endoftext|>' ||
      t.content === '<|im_end|>' ||
      t.content === '</s>' ||
      t.content === '<eos>',
  );
  if (eosToken) {
    eos_token_id = eosToken.id;
  }

  return {
    vocab,
    merges,
    added_tokens,
    model_type: 'bpe',
    eos_token_id,
  };
}

describe('Transformers.js Integration', () => {
  beforeAll(async () => {
    // Dynamically import transformers.js
    transformers = await import('@huggingface/transformers');
  });

  describe('GuidanceLogitsProcessor with transformer.js', () => {
    it('should integrate with LogitsProcessor interface', async () => {
      const { LogitsProcessor, LogitsProcessorList, Tensor } = transformers;

      // Create a mock parser that allows only specific tokens
      const allowedTokens = new Set([10, 20, 30, 40, 50]);
      const mockParser = {
        isTokenAllowed: vi.fn().mockImplementation((id: number) => allowedTokens.has(id)),
        getTokenMask: vi.fn().mockImplementation(() => {
          const mask = new Uint8Array(100);
          for (const t of allowedTokens) mask[t] = 1;
          return mask;
        }),
        advance: vi.fn(),
        isComplete: vi.fn().mockReturnValue(false),
        reset: vi.fn(),
        vocabSize: 100,
      } as unknown as GuidanceParserType;

      const guidanceProcessor = new GuidanceLogitsProcessor(mockParser, {
        speculationDepth: 5,
        debug: false,
      });

      // Create a transformer.js compatible logits processor
      class GuidanceTransformersProcessor extends LogitsProcessor {
        _call(inputIds: bigint[][], logits: InstanceType<typeof Tensor>): InstanceType<typeof Tensor> {
          const logitsData = logits.data as Float32Array;
          const vocabSize = logits.dims[logits.dims.length - 1];
          const batchSize = inputIds.length;
          const result = new Float32Array(logitsData.length);

          for (let b = 0; b < batchSize; b++) {
            const offset = b * vocabSize;
            const batchLogits = logitsData.slice(offset, offset + vocabSize);
            const inputIdNumbers = inputIds[b].map(id => Number(id));
            const processedLogits = guidanceProcessor.process(inputIdNumbers, batchLogits);
            result.set(processedLogits, offset);
          }

          return new Tensor(logits.type, result, logits.dims);
        }
      }

      // Create a logits processor list
      const logitsProcessorList = new LogitsProcessorList();
      const processor = new GuidanceTransformersProcessor();
      logitsProcessorList.push(processor);

      // Verify the processor is callable (transformer.js requirement)
      expect(typeof processor).toBe('function');
      expect(logitsProcessorList.processors.length).toBe(1);

      // Test with sample logits tensor
      const vocabSize = 100;
      const logitsArray = new Float32Array(vocabSize);
      for (let i = 0; i < vocabSize; i++) {
        logitsArray[i] = Math.random() * 10 - 5;
      }
      // Put high probability on allowed token
      logitsArray[20] = 10;
      logitsArray[50] = 8;

      const logitsTensor = new Tensor('float32', logitsArray, [1, vocabSize]);
      const inputIds: bigint[][] = [[1n, 2n, 3n]];

      // Process logits
      const result = processor._call(inputIds, logitsTensor);

      // Verify result is a tensor
      expect(result).toBeInstanceOf(Tensor);
      expect(result.dims).toEqual([1, vocabSize]);

      // Verify allowed token has its original value, others are masked
      const resultData = result.data as Float32Array;
      expect(resultData[20]).toBe(10); // Allowed token keeps value
      expect(resultData[0]).toBe(-Infinity); // Non-allowed token is masked
      expect(resultData[5]).toBe(-Infinity); // Non-allowed token is masked
    });

    it('should handle batched inputs correctly', async () => {
      const { LogitsProcessor, Tensor } = transformers;

      // Mock parser that tracks calls
      const processedBatches: number[][] = [];
      const mockParser = {
        isTokenAllowed: vi.fn().mockReturnValue(true),
        getTokenMask: vi.fn().mockReturnValue(new Uint8Array(50).fill(1)),
        advance: vi.fn(),
        isComplete: vi.fn().mockReturnValue(false),
        reset: vi.fn(),
        vocabSize: 50,
      } as unknown as GuidanceParserType;

      const guidanceProcessor = new GuidanceLogitsProcessor(mockParser, {
        speculationDepth: 3,
      });

      class GuidanceTransformersProcessor extends LogitsProcessor {
        _call(inputIds: bigint[][], logits: InstanceType<typeof Tensor>): InstanceType<typeof Tensor> {
          const logitsData = logits.data as Float32Array;
          const vocabSize = logits.dims[logits.dims.length - 1];
          const batchSize = inputIds.length;
          const result = new Float32Array(logitsData.length);

          for (let b = 0; b < batchSize; b++) {
            const offset = b * vocabSize;
            const batchLogits = logitsData.slice(offset, offset + vocabSize);
            const inputIdNumbers = inputIds[b].map(id => Number(id));
            processedBatches.push(inputIdNumbers);
            const processedLogits = guidanceProcessor.process(inputIdNumbers, batchLogits);
            result.set(processedLogits, offset);
          }

          return new Tensor(logits.type, result, logits.dims);
        }
      }

      const processor = new GuidanceTransformersProcessor();

      // Create batched input
      const batchSize = 3;
      const vocabSize = 50;
      const logitsArray = new Float32Array(batchSize * vocabSize);
      for (let i = 0; i < logitsArray.length; i++) {
        logitsArray[i] = Math.random() * 10;
      }

      const logitsTensor = new Tensor('float32', logitsArray, [batchSize, vocabSize]);
      const inputIds: bigint[][] = [
        [1n, 2n, 3n],
        [4n, 5n, 6n, 7n],
        [8n, 9n],
      ];

      const result = processor._call(inputIds, logitsTensor);

      // Verify all batches were processed
      expect(processedBatches.length).toBe(3);
      expect(processedBatches[0]).toEqual([1, 2, 3]);
      expect(processedBatches[1]).toEqual([4, 5, 6, 7]);
      expect(processedBatches[2]).toEqual([8, 9]);

      // Verify output shape
      expect(result.dims).toEqual([batchSize, vocabSize]);
    });
  });

  describe('TokenizerData extraction', () => {
    it('should extract tokenizer data from a real transformer.js tokenizer', async () => {
      const { AutoTokenizer } = transformers;

      const tokenizer = await AutoTokenizer.from_pretrained('Qwen/Qwen3-0.6B', {
        progress_callback: undefined,
      });

      const tokenizerData = extractTransformersTokenizerData(tokenizer);

      // Verify tokenizer data structure
      expect(tokenizerData.vocab).toBeDefined();
      expect(typeof tokenizerData.vocab).toBe('object');
      expect(Object.keys(tokenizerData.vocab).length).toBeGreaterThan(0);

      // Qwen3 vocabulary should have around 151669 tokens
      expect(Object.keys(tokenizerData.vocab).length).toBeGreaterThanOrEqual(150000);

      // Check for common tokens (Qwen3 uses different token IDs)
      expect(tokenizerData.vocab['0']).toBeDefined();
      expect(tokenizerData.vocab['a']).toBeDefined();
      expect(tokenizerData.vocab['Ġthe']).toBeDefined(); // Qwen also uses 'Ġ' for space prefix
    });

    it('should handle added tokens correctly', async () => {
      const { AutoTokenizer } = transformers;

      const tokenizer = await AutoTokenizer.from_pretrained('Qwen/Qwen3-0.6B', {
        progress_callback: undefined,
      });

      const tokenizerData = extractTransformersTokenizerData(tokenizer);

      // Qwen3 should have added tokens (like <|endoftext|>, <|im_start|>, <|im_end|>)
      expect(tokenizerData.added_tokens).toBeDefined();
      expect(tokenizerData.added_tokens!.length).toBeGreaterThan(0);

      const eosToken = tokenizerData.added_tokens!.find(
        (t) => t.content === '<|endoftext|>' || t.content === '<|im_end|>',
      );
      expect(eosToken).toBeDefined();
      expect(eosToken?.special).toBe(true);
    });
  });

  describe('Full generation simulation', () => {
    it('should simulate constrained generation with mock model', async () => {
      const { LogitsProcessor, LogitsProcessorList, Tensor } = transformers;

      // Define a simple grammar that only allows tokens that spell "hello"
      // h=104, e=101, l=108, o=111 in ASCII
      const allowedSequence = [104, 101, 108, 108, 111]; // 'hello'
      let currentStep = 0;

      const mockParser = {
        isTokenAllowed: vi.fn().mockImplementation((id: number) => {
          if (currentStep >= allowedSequence.length) return false;
          return id === allowedSequence[currentStep];
        }),
        getTokenMask: vi.fn().mockImplementation(() => {
          const mask = new Uint8Array(256);
          if (currentStep < allowedSequence.length) {
            mask[allowedSequence[currentStep]] = 1;
          }
          return mask;
        }),
        advance: vi.fn().mockImplementation(() => {
          currentStep++;
        }),
        isComplete: vi.fn().mockImplementation(() => currentStep >= allowedSequence.length),
        reset: vi.fn().mockImplementation(() => { currentStep = 0; }),
        vocabSize: 256,
      } as unknown as GuidanceParserType;

      const guidanceProcessor = new GuidanceLogitsProcessor(mockParser, {
        speculationDepth: 5,
        debug: false,
      });

      class GuidanceTransformersProcessor extends LogitsProcessor {
        _call(inputIds: bigint[][], logits: InstanceType<typeof Tensor>): InstanceType<typeof Tensor> {
          const logitsData = logits.data as Float32Array;
          const vocabSize = logits.dims[logits.dims.length - 1];
          const result = new Float32Array(logitsData.length);

          for (let b = 0; b < inputIds.length; b++) {
            const offset = b * vocabSize;
            const batchLogits = logitsData.slice(offset, offset + vocabSize);
            const inputIdNumbers = inputIds[b].map(id => Number(id));
            const processedLogits = guidanceProcessor.process(inputIdNumbers, batchLogits);
            result.set(processedLogits, offset);
          }

          return new Tensor(logits.type, result, logits.dims);
        }
      }

      const logitsProcessorList = new LogitsProcessorList();
      logitsProcessorList.push(new GuidanceTransformersProcessor());

      // Simulate generation loop
      const generatedTokens: number[] = [];
      let inputIds: bigint[][] = [[1n]]; // Start with a prompt token

      for (let step = 0; step < 10 && !guidanceProcessor.canStop(); step++) {
        // Simulate model output (uniform logits)
        const logits = new Float32Array(256).fill(0);
        // Give high probability to a wrong token (should be masked)
        logits[0] = 10;

        const logitsTensor = new Tensor('float32', logits, [1, 256]);

        // Process logits
        const processor = new GuidanceTransformersProcessor();
        const processedTensor = processor._call(inputIds, logitsTensor);
        const processedLogits = processedTensor.data as Float32Array;

        // Find argmax (simulating greedy decoding)
        let maxIdx = 0;
        let maxVal = processedLogits[0];
        for (let i = 1; i < processedLogits.length; i++) {
          if (processedLogits[i] > maxVal) {
            maxVal = processedLogits[i];
            maxIdx = i;
          }
        }

        generatedTokens.push(maxIdx);
        guidanceProcessor.onToken(maxIdx);
        inputIds = [[...inputIds[0], BigInt(maxIdx)]];
      }

      // Verify we generated the expected sequence
      expect(generatedTokens).toEqual(allowedSequence);
      expect(guidanceProcessor.canStop()).toBe(true);

      // Decode as string
      const generatedString = String.fromCharCode(...generatedTokens);
      expect(generatedString).toBe('hello');
    });
  });

  describe('Real GuidanceParser with transformer.js tokenizer', () => {
    // Use Qwen3-0.6B for realistic testing with a modern LLM tokenizer
    const MODEL_ID = 'Qwen/Qwen3-0.6B';

    it('should create parser with Qwen3 tokenizer and validate token masks', async () => {
        const { AutoTokenizer } = transformers;

        // Load the Qwen3 tokenizer
        const tokenizer = await AutoTokenizer.from_pretrained(MODEL_ID, {
          progress_callback: undefined,
        });

        // Extract tokenizer data in the format expected by llguidance
        const tokenizerData = extractTransformersTokenizerData(tokenizer);

        // Create a simple regex grammar that matches digits
        const grammar: Grammar = {
          type: 'regex',
          pattern: '[0-9]+',
        };

        // Create the parser with real tokenizer data
        const parser = await GuidanceParser.create(grammar, tokenizerData);

        // Verify parser was created with correct vocabulary size
        expect(parser.vocabSize).toBe(Object.keys(tokenizerData.vocab).length);
        expect(parser.vocabSize).toBeGreaterThanOrEqual(150000); // Qwen3 has ~151k tokens

        // Get token mask - should allow digit tokens
        const mask = parser.getTokenMask();
        expect(mask).toBeInstanceOf(Uint8Array);
        expect(mask.length).toBe(parser.vocabSize);

        // Find digit tokens in the vocabulary
        const digit0Token = tokenizerData.vocab['0'];
        const digit1Token = tokenizerData.vocab['1'];
        const digit9Token = tokenizerData.vocab['9'];

        // Digit tokens should be allowed
        if (digit0Token !== undefined) {
          expect(mask[digit0Token]).toBe(1);
        }
        if (digit1Token !== undefined) {
          expect(mask[digit1Token]).toBe(1);
        }
        if (digit9Token !== undefined) {
          expect(mask[digit9Token]).toBe(1);
        }

        // Letter tokens should NOT be allowed for digit-only grammar
        const letterAToken = tokenizerData.vocab['a'];
        const letterBToken = tokenizerData.vocab['b'];

        if (letterAToken !== undefined) {
          expect(mask[letterAToken]).toBe(0);
        }
        if (letterBToken !== undefined) {
          expect(mask[letterBToken]).toBe(0);
        }
      },
      60000,
    );

    it('should create parser with JSON schema grammar', async () => {
        const { AutoTokenizer } = transformers;

        const tokenizer = await AutoTokenizer.from_pretrained(MODEL_ID, {
          progress_callback: undefined,
        });

        const tokenizerData = extractTransformersTokenizerData(tokenizer);

        // Create a JSON schema grammar
        const grammar: Grammar = {
          type: 'json_schema',
          schema: {
            type: 'object',
            properties: {
              name: { type: 'string' },
              age: { type: 'integer' },
            },
            required: ['name', 'age'],
          },
        };

        const parser = await GuidanceParser.create(grammar, tokenizerData);

        // Get initial mask - JSON should start with '{'
        const mask = parser.getTokenMask();

        // Find the '{' token
        const openBraceToken = tokenizerData.vocab['{'];

        // Opening brace should be allowed at the start
        if (openBraceToken !== undefined) {
          expect(mask[openBraceToken]).toBe(1);
        }

        // Letter tokens should NOT be allowed at the start (must start with {)
        const letterAToken = tokenizerData.vocab['a'];
        if (letterAToken !== undefined) {
          expect(mask[letterAToken]).toBe(0);
        }
      },
      60000,
    );

    it('should integrate GuidanceLogitsProcessor with real parser', async () => {
        const { AutoTokenizer, LogitsProcessor, Tensor } = transformers;

        const tokenizer = await AutoTokenizer.from_pretrained(MODEL_ID, {
          progress_callback: undefined,
        });

        const tokenizerData = extractTransformersTokenizerData(tokenizer);

        // Simple grammar: only allow digits
        const grammar: Grammar = {
          type: 'regex',
          pattern: '[0-9]+',
        };

        const parser = await GuidanceParser.create(grammar, tokenizerData);
        const processor = new GuidanceLogitsProcessor(parser, {
          speculationDepth: 5,
          debug: false,
        });

        // Create a transformer.js compatible processor
        class GuidanceTransformersProcessor extends LogitsProcessor {
          _call(
            inputIds: bigint[][],
            logits: InstanceType<typeof Tensor>,
          ): InstanceType<typeof Tensor> {
            const logitsData = logits.data as Float32Array;
            const vocabSize = logits.dims[logits.dims.length - 1];
            const result = new Float32Array(logitsData.length);

            for (let b = 0; b < inputIds.length; b++) {
              const offset = b * vocabSize;
              const batchLogits = logitsData.slice(offset, offset + vocabSize);
              const inputIdNumbers = inputIds[b].map((id) => Number(id));
              const processedLogits = processor.process(inputIdNumbers, batchLogits);
              result.set(processedLogits, offset);
            }

            return new Tensor(logits.type, result, logits.dims);
          }
        }

        const tfProcessor = new GuidanceTransformersProcessor();

        // Create sample logits (all tokens have equal probability)
        const vocabSize = parser.vocabSize;
        const logitsArray = new Float32Array(vocabSize).fill(0);

        // Give high probability to a letter token (should be masked)
        const letterAToken = tokenizerData.vocab['a'];
        if (letterAToken !== undefined) {
          logitsArray[letterAToken] = 10;
        }

        // Give medium probability to a digit token (should be kept)
        const digit0Token = tokenizerData.vocab['0'];
        if (digit0Token !== undefined) {
          logitsArray[digit0Token] = 5;
        }

        const logitsTensor = new Tensor('float32', logitsArray, [1, vocabSize]);
        const inputIds: bigint[][] = [[1n]];

        // Process logits
        const result = tfProcessor._call(inputIds, logitsTensor);
        const resultData = result.data as Float32Array;

        // Letter token should be masked (-Infinity)
        if (letterAToken !== undefined) {
          expect(resultData[letterAToken]).toBe(-Infinity);
        }

        // Digit token should keep its value
        if (digit0Token !== undefined) {
          expect(resultData[digit0Token]).toBe(5);
        }
      },
      60000,
    );
  });
});

