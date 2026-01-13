# llguidance-js

Structured output generation for [transformer.js](https://github.com/huggingface/transformers.js) using [llguidance](https://github.com/guidance-ai/llguidance).

This library enables constrained text generation in the browser and Node.js by integrating the high-performance llguidance Rust library with transformer.js via WebAssembly.

## Features

- **JSON Schema constraints** - Generate valid JSON matching any JSON Schema
- **Regex patterns** - Constrain output to match regular expressions
- **Lark grammars** - Full CFG support for complex structured output
- **Speculative decoding** - Optimized performance with fast-path token validation
- **Zero server dependencies** - Runs entirely in browser/Node.js

## Installation

```bash
npm install llguidance-js
```

## Quick Start

```typescript
import { pipeline } from '@huggingface/transformers';
import {
  GuidanceParser,
  GuidanceLogitsProcessor,
  extractTokenizerData,
} from 'llguidance-js';

// Load a model
const generator = await pipeline('text-generation', 'Xenova/gpt2');

// Extract tokenizer data
const tokenizerData = extractTokenizerData(generator.tokenizer);

// Create a parser with JSON schema constraint
const parser = await GuidanceParser.create({
  type: 'json_schema',
  schema: {
    type: 'object',
    properties: {
      name: { type: 'string' },
      age: { type: 'number' }
    },
    required: ['name', 'age']
  }
}, tokenizerData);

// Create logits processor
const processor = new GuidanceLogitsProcessor(parser);

// Generate constrained output
const output = await generator('Generate a person:', {
  max_new_tokens: 50,
  logits_processor: [processor],
});

console.log(output[0].generated_text);
// Output will always be valid JSON matching the schema
```

## Grammar Types

### JSON Schema

```typescript
const grammar = {
  type: 'json_schema',
  schema: {
    type: 'object',
    properties: {
      name: { type: 'string' },
      age: { type: 'integer', minimum: 0 }
    },
    required: ['name', 'age']
  }
};
```

### Regex Pattern

```typescript
const grammar = {
  type: 'regex',
  pattern: '[a-zA-Z]+@[a-zA-Z]+\\.[a-zA-Z]{2,}'
};
```

### Lark Grammar (CFG)

```typescript
const grammar = {
  type: 'lark',
  grammar: `
    start: expr
    expr: term (("+"|"-") term)*
    term: NUMBER
    NUMBER: /[0-9]+/
  `,
  startSymbol: 'start'
};
```

## API Reference

### `GuidanceParser`

The core parser that wraps the llguidance WASM module.

```typescript
class GuidanceParser {
  // Create a new parser instance
  static async create(grammar: Grammar, tokenizer: TokenizerData): Promise<GuidanceParser>;

  // Fast O(1) check if a token is allowed
  isTokenAllowed(tokenId: number): boolean;

  // Get full token mask (slower, use for fallback)
  getTokenMask(): Uint8Array;

  // Advance parser state after token selection
  advance(tokenId: number): void;

  // Check if generation can terminate
  isComplete(): boolean;

  // Reset parser for reuse
  reset(): void;

  // Get vocabulary size
  get vocabSize(): number;
}
```

### `GuidanceLogitsProcessor`

Logits processor compatible with transformer.js.

```typescript
class GuidanceLogitsProcessor {
  constructor(parser: GuidanceParser, options?: ProcessorOptions);

  // Process logits (called by transformer.js)
  process(inputIds: number[], logits: Float32Array): Float32Array;

  // Advance state after sampling (call after each token)
  onToken(tokenId: number): void;

  // Check if generation can stop
  canStop(): boolean;

  // Reset for new generation
  reset(): void;
}

interface ProcessorOptions {
  // Number of top tokens to try before full mask (default: 5)
  speculationDepth?: number;

  // Enable debug logging (default: false)
  debug?: boolean;
}
```

### Tokenizer Utilities

```typescript
// Extract tokenizer data from transformer.js tokenizer
function extractTokenizerData(tokenizer: TransformersTokenizer): TokenizerData;

// Load tokenizer data directly from HuggingFace Hub
async function loadTokenizerData(modelId: string, options?: {
  token?: string;
  baseUrl?: string;
}): Promise<TokenizerData>;
```

## How It Works

1. **Grammar compilation**: llguidance compiles your grammar (JSON schema, regex, or Lark) into an efficient state machine
2. **Speculative checking**: During generation, we first check if the model's top-k predicted tokens are valid (fast path)
3. **Fallback masking**: If no top-k tokens are valid, we compute the full token mask (slower path)
4. **Logit modification**: Invalid tokens have their logits set to -∞, ensuring they're never sampled

### Generation Loop

1. Model produces logits
2. GuidanceLogitsProcessor.process() called
  1. Try top-5 tokens with is_token_allowed()
  2. If hit: mask all except winner
  3. If miss: compute full mask with get_token_mask()
3. Sample from modified logits
4. Call processor.onToken() with sampled token
5. Repeat until processor.canStop() or max tokens

## Building from Source

### Prerequisites

- Node.js 18+
- Rust toolchain with `wasm32-unknown-unknown` target
- wasm-pack

### Build

```bash
# Install dependencies
npm install

# Build WASM module
npm run build:wasm

# Build TypeScript
npm run build

# Run tests
npm test
```

## Performance Tips

1. **Use speculative decoding**: The default `speculationDepth: 5` works well for most cases. Increase for models with more uncertain predictions.

2. **Reuse parsers**: Create the parser once and call `reset()` between generations instead of creating new instances.

3. **Batch processing**: When generating multiple outputs with the same grammar, reuse the same parser instance.

## Limitations

- Currently requires the WASM module to be built from source
- Some llguidance features may require adjustment for WASM compatibility
- Large grammars may increase WASM binary size

## License

MIT

## Acknowledgments

- [llguidance](https://github.com/guidance-ai/llguidance) - The Rust library powering the structured output
- [transformer.js](https://github.com/huggingface/transformers.js) - Machine learning in the browser
- [wasm-bindgen](https://github.com/rustwasm/wasm-bindgen) - Rust/WebAssembly interop

