/**
 * Tokenizer bridge utilities for converting transformer.js tokenizer format
 * to the format expected by llguidance WASM module.
 */

import type { TokenizerData } from './types';

/**
 * Represents a transformer.js tokenizer instance.
 * This is a minimal interface for the parts we need.
 */
export interface TransformersTokenizer {
  /** The model data containing vocabulary */
  model?: {
    vocab?: Map<string, number> | Record<string, number>;
    tokens_to_ids?: Map<string, number> | Record<string, number>;
    merges?: string[];
  };
  /** Direct vocab access for some tokenizer types */
  vocab?: Map<string, number> | Record<string, number>;
  /** Added tokens (special tokens) */
  added_tokens?: Array<{
    id: number;
    content: string;
    single_word?: boolean;
    lstrip?: boolean;
    rstrip?: boolean;
    normalized?: boolean;
    special?: boolean;
  }>;
  /** Get vocabulary method */
  getVocab?: () => Record<string, number>;
  /** Encode method for testing */
  encode?: (text: string) => number[];
  /** Special token IDs */
  eos_token_id?: number;
  bos_token_id?: number;
  pad_token_id?: number;
  unk_token_id?: number;
  /** Special tokens as strings */
  eos_token?: string;
  bos_token?: string;
  pad_token?: string;
  unk_token?: string;
}

/**
 * Extract tokenizer data from a transformer.js tokenizer instance.
 *
 * @param tokenizer The transformer.js tokenizer instance
 * @returns TokenizerData in the format expected by llguidance
 *
 * @example
 * ```typescript
 * import { AutoTokenizer } from '@huggingface/transformers';
 * import { extractTokenizerData } from 'llguidance-js';
 *
 * const tokenizer = await AutoTokenizer.from_pretrained('gpt2');
 * const tokenizerData = extractTokenizerData(tokenizer);
 * ```
 */
export function extractTokenizerData(
  tokenizer: TransformersTokenizer,
): TokenizerData {
  // Try to get vocabulary from various sources
  let vocab: Record<string, number>;

  if (tokenizer.getVocab) {
    // Preferred method - direct vocab access
    vocab = tokenizer.getVocab();
  } else if (tokenizer.model?.tokens_to_ids) {
    // transformer.js stores vocab as tokens_to_ids Map
    vocab = mapToRecord(tokenizer.model.tokens_to_ids);
  } else if (tokenizer.model?.vocab) {
    // Some tokenizers store vocab in model
    vocab = mapToRecord(tokenizer.model.vocab);
  } else if (tokenizer.vocab) {
    // Direct vocab property
    vocab = mapToRecord(tokenizer.vocab);
  } else {
    throw new Error(
      'Unable to extract vocabulary from tokenizer. ' +
        'Ensure you are passing a valid transformer.js tokenizer instance.',
    );
  }

  // Get merges if available (for BPE tokenizers)
  const merges = tokenizer.model?.merges ?? [];

  // Get added tokens (special tokens)
  const added_tokens = (tokenizer.added_tokens ?? []).map((token) => ({
    id: token.id,
    content: token.content,
    single_word: token.single_word ?? false,
    lstrip: token.lstrip ?? false,
    rstrip: token.rstrip ?? false,
    normalized: token.normalized ?? true,
    special: token.special ?? false,
  }));

  // Extract special token IDs
  const eos_token_id = extractSpecialTokenId(tokenizer, vocab, 'eos');
  const bos_token_id = extractSpecialTokenId(tokenizer, vocab, 'bos');
  const pad_token_id = extractSpecialTokenId(tokenizer, vocab, 'pad');
  const unk_token_id = extractSpecialTokenId(tokenizer, vocab, 'unk');

  return {
    vocab,
    merges,
    added_tokens,
    model_type: detectModelType(tokenizer),
    eos_token_id,
    bos_token_id,
    pad_token_id,
    unk_token_id,
  };
}

/**
 * Extract a special token ID from the tokenizer
 */
function extractSpecialTokenId(
  tokenizer: TransformersTokenizer,
  vocab: Record<string, number>,
  tokenType: 'eos' | 'bos' | 'pad' | 'unk',
): number | undefined {
  // First try direct token ID property
  const idKey = `${tokenType}_token_id` as keyof TransformersTokenizer;
  if (typeof tokenizer[idKey] === 'number') {
    return tokenizer[idKey] as number;
  }

  // Then try token string property and look up in vocab
  const tokenKey = `${tokenType}_token` as keyof TransformersTokenizer;
  const tokenStr = tokenizer[tokenKey] as string | undefined;
  if (tokenStr && vocab[tokenStr] !== undefined) {
    return vocab[tokenStr];
  }

  // For EOS, try common token names
  if (tokenType === 'eos') {
    const eosNames = ['</s>', '<|endoftext|>', '<eos>', '<|eos|>', '[SEP]'];
    for (const name of eosNames) {
      if (vocab[name] !== undefined) {
        return vocab[name];
      }
    }
  }

  // For BOS, try common token names
  if (tokenType === 'bos') {
    const bosNames = ['<s>', '<|startoftext|>', '<bos>', '<|bos|>', '[CLS]'];
    for (const name of bosNames) {
      if (vocab[name] !== undefined) {
        return vocab[name];
      }
    }
  }

  // For PAD, try common token names
  if (tokenType === 'pad') {
    const padNames = ['<pad>', '<|pad|>', '[PAD]'];
    for (const name of padNames) {
      if (vocab[name] !== undefined) {
        return vocab[name];
      }
    }
  }

  // For UNK, try common token names
  if (tokenType === 'unk') {
    const unkNames = ['<unk>', '<|unk|>', '[UNK]'];
    for (const name of unkNames) {
      if (vocab[name] !== undefined) {
        return vocab[name];
      }
    }
  }

  return undefined;
}

/**
 * Convert a Map or Record to a plain Record
 */
function mapToRecord(
  input: Map<string, number> | Record<string, number>,
): Record<string, number> {
  if (input instanceof Map) {
    const result: Record<string, number> = {};
    for (const [key, value] of input) {
      result[key] = value;
    }
    return result;
  }
  return input;
}

/**
 * Try to detect the tokenizer model type
 */
function detectModelType(tokenizer: TransformersTokenizer): string {
  // Check if it's a BPE tokenizer (has merges)
  if (tokenizer.model?.merges && tokenizer.model.merges.length > 0) {
    return 'bpe';
  }
  // Default to unknown
  return 'unknown';
}

/**
 * Load tokenizer data from a HuggingFace model ID.
 * This fetches the tokenizer.json file directly.
 *
 * @param modelId The HuggingFace model ID (e.g., 'gpt2', 'meta-llama/Llama-2-7b')
 * @param options Optional configuration
 * @returns TokenizerData in the format expected by llguidance
 *
 * @example
 * ```typescript
 * import { loadTokenizerData } from 'llguidance-js';
 *
 * const tokenizerData = await loadTokenizerData('gpt2');
 * ```
 */
export async function loadTokenizerData(
  modelId: string,
  options: {
    /** HuggingFace API token for private models */
    token?: string;
    /** Custom base URL for HuggingFace Hub */
    baseUrl?: string;
  } = {},
): Promise<TokenizerData> {
  const baseUrl = options.baseUrl ?? 'https://huggingface.co';
  const url = `${baseUrl}/${modelId}/resolve/main/tokenizer.json`;

  const headers: Record<string, string> = {};
  if (options.token) {
    headers['Authorization'] = `Bearer ${options.token}`;
  }

  const response = await fetch(url, { headers });
  if (!response.ok) {
    throw new Error(
      `Failed to fetch tokenizer from ${url}: ${response.status} ${response.statusText}`,
    );
  }

  const tokenizerJson = await response.json();
  return parseTokenizerJson(tokenizerJson);
}

/**
 * Parse a tokenizer.json file into TokenizerData format
 */
function parseTokenizerJson(json: unknown): TokenizerData {
  const data = json as {
    model?: {
      vocab?: Record<string, number>;
      merges?: string[];
      type?: string;
    };
    added_tokens?: Array<{
      id: number;
      content: string;
      single_word?: boolean;
      lstrip?: boolean;
      rstrip?: boolean;
      normalized?: boolean;
      special?: boolean;
    }>;
  };

  if (!data.model?.vocab) {
    throw new Error('Invalid tokenizer.json: missing model.vocab');
  }

  const vocab = data.model.vocab;
  const added_tokens = data.added_tokens?.map((token) => ({
    id: token.id,
    content: token.content,
    single_word: token.single_word ?? false,
    lstrip: token.lstrip ?? false,
    rstrip: token.rstrip ?? false,
    normalized: token.normalized ?? true,
    special: token.special ?? false,
  }));

  // Find special token IDs from added_tokens
  let eos_token_id: number | undefined;
  let bos_token_id: number | undefined;
  let pad_token_id: number | undefined;
  let unk_token_id: number | undefined;

  if (added_tokens) {
    for (const token of added_tokens) {
      const content = token.content.toLowerCase();
      if (
        content === '</s>' ||
        content === '<|endoftext|>' ||
        content === '<eos>'
      ) {
        eos_token_id = token.id;
      } else if (
        content === '<s>' ||
        content === '<|startoftext|>' ||
        content === '<bos>'
      ) {
        bos_token_id = token.id;
      } else if (content === '<pad>' || content === '<|pad|>') {
        pad_token_id = token.id;
      } else if (content === '<unk>' || content === '<|unk|>') {
        unk_token_id = token.id;
      }
    }
  }

  // Also check vocab for special tokens
  const eosNames = ['</s>', '<|endoftext|>', '<eos>', '<|eos|>'];
  const bosNames = ['<s>', '<|startoftext|>', '<bos>', '<|bos|>'];
  const padNames = ['<pad>', '<|pad|>'];
  const unkNames = ['<unk>', '<|unk|>'];

  if (eos_token_id === undefined) {
    for (const name of eosNames) {
      if (vocab[name] !== undefined) {
        eos_token_id = vocab[name];
        break;
      }
    }
  }
  if (bos_token_id === undefined) {
    for (const name of bosNames) {
      if (vocab[name] !== undefined) {
        bos_token_id = vocab[name];
        break;
      }
    }
  }
  if (pad_token_id === undefined) {
    for (const name of padNames) {
      if (vocab[name] !== undefined) {
        pad_token_id = vocab[name];
        break;
      }
    }
  }
  if (unk_token_id === undefined) {
    for (const name of unkNames) {
      if (vocab[name] !== undefined) {
        unk_token_id = vocab[name];
        break;
      }
    }
  }

  return {
    vocab,
    merges: data.model.merges ?? [],
    added_tokens,
    model_type: data.model.type ?? 'unknown',
    eos_token_id,
    bos_token_id,
    pad_token_id,
    unk_token_id,
  };
}

