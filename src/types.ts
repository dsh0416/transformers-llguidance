/**
 * JSON Schema grammar definition
 */
export interface JsonSchemaGrammar {
  type: 'json_schema';
  schema: Record<string, unknown>;
}

/**
 * Regular expression grammar definition
 */
export interface RegexGrammar {
  type: 'regex';
  pattern: string;
}

/**
 * Lark grammar definition (CFG)
 */
export interface LarkGrammar {
  type: 'lark';
  grammar: string;
  startSymbol?: string;
}

/**
 * Union of all supported grammar types
 */
export type Grammar = JsonSchemaGrammar | RegexGrammar | LarkGrammar;

/**
 * Tokenizer data in HuggingFace format
 * This is the format used by transformer.js tokenizers
 */
export interface TokenizerData {
  /** The vocabulary mapping tokens to IDs */
  vocab: Record<string, number>;
  /** Merge rules for BPE tokenizers */
  merges?: string[];
  /** Special tokens configuration */
  added_tokens?: Array<{
    id: number;
    content: string;
    single_word: boolean;
    lstrip: boolean;
    rstrip: boolean;
    normalized: boolean;
    special: boolean;
  }>;
  /** The model type (BPE, WordPiece, etc.) */
  model_type?: string;
  /** End of sequence token ID */
  eos_token_id?: number;
  /** Beginning of sequence token ID */
  bos_token_id?: number;
  /** Padding token ID */
  pad_token_id?: number;
  /** Unknown token ID */
  unk_token_id?: number;
}

/**
 * Options for the logits processor
 */
export interface ProcessorOptions {
  /**
   * Number of top tokens to try speculatively before falling back to full mask
   * @default 5
   */
  speculationDepth?: number;

  /**
   * Whether to enable debug logging
   * @default false
   */
  debug?: boolean;
}

