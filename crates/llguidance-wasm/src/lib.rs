//! WebAssembly bindings for llguidance
//!
//! This crate provides JavaScript-accessible bindings to the llguidance
//! constrained generation library, enabling grammar-based token validation
//! for use with transformer.js.

use js_sys::Uint8Array;
use serde::Deserialize;
use std::collections::HashMap;
use std::sync::Arc;
use wasm_bindgen::prelude::*;

use llguidance::api::TopLevelGrammar;
use llguidance::toktrie::{ApproximateTokEnv, TokRxInfo, TokTrie};
use llguidance::{Matcher, ParserFactory};

/// Grammar definition passed from JavaScript
#[derive(Debug, Deserialize)]
struct GrammarInput {
    grammars: Vec<GrammarSpec>,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum GrammarSpec {
    JsonSchema { json_schema: serde_json::Value },
    Regex { rx: String },
    Lark { lark: String },
}

/// Tokenizer data passed from JavaScript
/// This matches the TokenizerData interface in TypeScript
#[derive(Debug, Deserialize)]
struct TokenizerInput {
    /// Vocabulary mapping token strings to IDs
    vocab: HashMap<String, u32>,
    /// Optional BPE merges (not used for trie construction but kept for compatibility)
    /// Can be either strings like "Ġ t" or arrays like ["Ġ", "t"]
    #[serde(default, deserialize_with = "deserialize_merges")]
    #[allow(dead_code)]
    merges: Vec<String>,
    /// Added tokens (special tokens)
    #[serde(default)]
    added_tokens: Vec<AddedToken>,
    /// Model type (e.g., "bpe", "wordpiece")
    /// For Transformer.js compatibility, we keep this field but it is not used.
    #[serde(default)]
    #[allow(dead_code)]
    model_type: Option<String>,
    /// Special token IDs
    #[serde(default)]
    eos_token_id: Option<u32>,
    #[serde(default)]
    bos_token_id: Option<u32>,
    #[serde(default)]
    pad_token_id: Option<u32>,
    #[serde(default)]
    unk_token_id: Option<u32>,
}

/// Custom deserializer for merges that handles both string and array formats
fn deserialize_merges<'de, D>(deserializer: D) -> Result<Vec<String>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    use serde::de::{self, SeqAccess, Visitor};
    use std::fmt;

    struct MergesVisitor;

    impl<'de> Visitor<'de> for MergesVisitor {
        type Value = Vec<String>;

        fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
            formatter.write_str("a sequence of strings or arrays of strings")
        }

        fn visit_seq<S>(self, mut seq: S) -> Result<Vec<String>, S::Error>
        where
            S: SeqAccess<'de>,
        {
            let mut merges = Vec::new();

            while let Some(value) = seq.next_element::<serde_json::Value>()? {
                let merge_str = match value {
                    serde_json::Value::String(s) => s,
                    serde_json::Value::Array(arr) => {
                        // Convert array like ["Ġ", "t"] to string "Ġ t"
                        let parts: Vec<String> = arr
                            .into_iter()
                            .filter_map(|v| v.as_str().map(String::from))
                            .collect();
                        parts.join(" ")
                    }
                    _ => {
                        return Err(de::Error::custom(
                            "merge must be a string or array of strings",
                        ))
                    }
                };
                merges.push(merge_str);
            }

            Ok(merges)
        }
    }

    deserializer.deserialize_seq(MergesVisitor)
}

#[derive(Debug, Deserialize)]
struct AddedToken {
    id: u32,
    content: String,
    #[serde(default)]
    special: bool,
}

/// The main parser struct exposed to JavaScript
#[wasm_bindgen]
pub struct LLGuidanceParser {
    factory: Arc<ParserFactory>,
    matcher: Matcher,
    vocab_size: usize,
}

#[wasm_bindgen]
impl LLGuidanceParser {
    /// Create a new parser with the given grammar and tokenizer configuration
    #[wasm_bindgen(constructor)]
    pub fn new(grammar_json: &str, tokenizer_json: &str) -> Result<LLGuidanceParser, JsValue> {
        // Set up panic hook for better error messages
        #[cfg(feature = "console_error_panic_hook")]
        console_error_panic_hook::set_once();

        Self::new_inner(grammar_json, tokenizer_json).map_err(|e| JsValue::from_str(&e))
    }

    fn new_inner(grammar_json: &str, tokenizer_json: &str) -> Result<LLGuidanceParser, String> {
        // Parse the grammar
        let grammar = Self::parse_grammar(grammar_json)?;

        // Create tokenizer environment
        let tok_env = Self::create_tok_env(tokenizer_json)?;
        let vocab_size = tok_env.tok_trie().vocab_size();

        // Create parser factory
        let mut factory = ParserFactory::new_simple(&tok_env)
            .map_err(|e| format!("Failed to create parser factory: {}", e))?;

        // Minimal logging
        factory.set_stderr_log_level(0);

        let factory = Arc::new(factory);

        // Create the parser and matcher
        let parser = factory.create_parser(grammar);
        let matcher = Matcher::new(parser);

        Ok(LLGuidanceParser {
            factory,
            matcher,
            vocab_size,
        })
    }

    /// Create a tokenizer environment from the JSON configuration
    fn create_tok_env(
        tokenizer_json: &str,
    ) -> Result<Arc<dyn llguidance::toktrie::TokenizerEnv + Sync>, String> {
        // Try to parse as TokenizerInput
        let input: TokenizerInput = serde_json::from_str(tokenizer_json)
            .map_err(|e| format!("Failed to parse tokenizer JSON: {}", e))?;

        // Check if we have a valid vocabulary
        if input.vocab.is_empty() {
            return Err("Tokenizer vocabulary is empty".to_string());
        }

        // Find the maximum token ID to determine vocab size
        let max_id = input.vocab.values().copied().max().unwrap_or(0);
        let vocab_size = (max_id + 1) as usize;

        // Build the words vector (token bytes indexed by token ID)
        // Each entry is the byte representation of the token
        let mut words: Vec<Vec<u8>> = vec![Vec::new(); vocab_size];

        for (token_str, id) in &input.vocab {
            if (*id as usize) < vocab_size {
                // Handle special token encoding
                // llguidance uses \xFF prefix for special tokens
                let bytes = if input.added_tokens.iter().any(|t| t.id == *id && t.special) {
                    // Special tokens get the \xFF prefix
                    let mut special_bytes = vec![0xFF];
                    special_bytes.extend(token_str.as_bytes());
                    special_bytes
                } else {
                    // Regular tokens: decode the token string
                    // GPT-2 style tokenizers use 'Ġ' (U+0120) to represent space
                    // and other Unicode characters for byte encoding
                    decode_token_bytes(token_str)
                };
                words[*id as usize] = bytes;
            }
        }

        // Determine EOS token
        // Priority: explicit eos_token_id > added token named </s> or <|endoftext|> > last token
        let eos_token = input.eos_token_id.unwrap_or_else(|| {
            // Look for common EOS tokens in added_tokens
            for token in &input.added_tokens {
                if token.content == "</s>"
                    || token.content == "<|endoftext|>"
                    || token.content == "<eos>"
                    || token.content == "<|eos|>"
                {
                    return token.id;
                }
            }
            // Fallback to last token
            (vocab_size - 1) as u32
        });

        // Create TokRxInfo
        let mut info = TokRxInfo::new(vocab_size as u32, eos_token);
        info.tok_bos = input.bos_token_id;
        info.tok_pad = input.pad_token_id;
        info.tok_unk = input.unk_token_id;

        // Create the trie
        let trie = TokTrie::from(&info, &words);

        // Wrap in ApproximateTokEnv
        let tok_env = ApproximateTokEnv::new(trie);

        Ok(Arc::new(tok_env))
    }

    fn parse_grammar(grammar_json: &str) -> Result<TopLevelGrammar, String> {
        // Try to parse as our simplified GrammarInput format first (most common case)
        if let Ok(input) = serde_json::from_str::<GrammarInput>(grammar_json) {
            if !input.grammars.is_empty() {
                return Self::convert_grammar(&input);
            }
        }

        // Fall back to parsing directly as TopLevelGrammar (native .ll.json format)
        serde_json::from_str::<TopLevelGrammar>(grammar_json)
            .map_err(|e| format!("Failed to parse grammar JSON: {}", e))
    }

    fn convert_grammar(input: &GrammarInput) -> Result<TopLevelGrammar, String> {
        if input.grammars.is_empty() {
            return Err("No grammars provided".to_string());
        }

        // For now, handle the first grammar only
        let spec = &input.grammars[0];

        match spec {
            GrammarSpec::JsonSchema { json_schema } => {
                // Use TopLevelGrammar::from_json_schema
                Ok(TopLevelGrammar::from_json_schema(json_schema.clone()))
            }
            GrammarSpec::Regex { rx } => {
                // Create a lark grammar that matches the regex
                let lark_grammar = format!("start: /{}/", rx);
                Ok(TopLevelGrammar::from_lark(lark_grammar))
            }
            GrammarSpec::Lark { lark } => Ok(TopLevelGrammar::from_lark(lark.clone())),
        }
    }

    /// Check if a specific token is allowed at the current position
    #[wasm_bindgen]
    pub fn is_token_allowed(&mut self, token_id: u32) -> Result<bool, JsValue> {
        let mask = self
            .matcher
            .compute_mask()
            .map_err(|e| JsValue::from_str(&format!("Failed to compute mask: {}", e)))?;

        Ok(mask.is_allowed(token_id))
    }

    /// Get the full token mask for the current position
    #[wasm_bindgen]
    pub fn get_token_mask(&mut self) -> Result<Uint8Array, JsValue> {
        let mask = self
            .matcher
            .compute_mask()
            .map_err(|e| JsValue::from_str(&format!("Failed to compute mask: {}", e)))?;

        let mut mask_vec = vec![0u8; self.vocab_size];
        for (i, item) in mask_vec.iter_mut().enumerate().take(self.vocab_size) {
            if mask.is_allowed(i as u32) {
                *item = 1;
            }
        }

        let js_array = Uint8Array::new_with_length(mask_vec.len() as u32);
        js_array.copy_from(&mask_vec);
        Ok(js_array)
    }

    /// Advance the parser state after a token has been selected
    #[wasm_bindgen]
    pub fn advance(&mut self, token_id: u32) -> Result<(), JsValue> {
        self.matcher
            .consume_token(token_id)
            .map_err(|e| JsValue::from_str(&format!("Failed to consume token: {}", e)))?;
        Ok(())
    }

    /// Check if the current state represents a valid complete parse
    #[wasm_bindgen]
    pub fn is_complete(&self) -> bool {
        let reason = format!("{:?}", self.matcher.stop_reason());
        reason.contains("EndOfSentence")
            || reason.contains("NoExtension")
            || reason.contains("MaxTokensTotal")
            || reason.contains("NoExtensionBias")
    }

    /// Reset the parser to its initial state
    #[wasm_bindgen]
    pub fn reset(&mut self, grammar_json: &str) -> Result<(), JsValue> {
        let grammar = Self::parse_grammar(grammar_json).map_err(|e| JsValue::from_str(&e))?;
        let parser = self.factory.create_parser(grammar);
        self.matcher = Matcher::new(parser);
        Ok(())
    }

    /// Get the vocabulary size
    #[wasm_bindgen]
    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    /// Get the current stop reason
    #[wasm_bindgen]
    pub fn stop_reason(&self) -> String {
        format!("{:?}", self.matcher.stop_reason())
    }
}

/// Decode a token string to its byte representation
/// Handles GPT-2/BPE style encoding where special Unicode characters represent bytes
fn decode_token_bytes(token: &str) -> Vec<u8> {
    let mut result = Vec::new();

    for c in token.chars() {
        match c {
            // GPT-2 style: 'Ġ' (U+0120) represents space (0x20)
            'Ġ' => result.push(b' '),
            // GPT-2 style: 'Ċ' (U+010A) represents newline (0x0A)
            'Ċ' => result.push(b'\n'),
            // GPT-2 style: 'ċ' (U+010B) sometimes used for tab
            'ċ' => result.push(b'\t'),
            // GPT-2 uses Unicode range U+0100-U+01FF to encode bytes 0x00-0xFF
            // The mapping is: byte = unicode_codepoint - 0x100 (for 0x00-0x20, 0x7F-0xA0, 0xAD)
            // But for printable ASCII, the character is used directly
            c if c as u32 >= 0x100 && c as u32 <= 0x1FF => {
                // This is a GPT-2 byte encoding
                let byte = (c as u32 - 0x100) as u8;
                result.push(byte);
            }
            // Regular ASCII/UTF-8 characters
            c => {
                // Encode the character as UTF-8 bytes
                let mut buf = [0u8; 4];
                let s = c.encode_utf8(&mut buf);
                result.extend(s.as_bytes());
            }
        }
    }

    result
}

/// Initialize the WASM module
#[wasm_bindgen(start)]
pub fn init() {
    #[cfg(feature = "console_error_panic_hook")]
    console_error_panic_hook::set_once();
}
