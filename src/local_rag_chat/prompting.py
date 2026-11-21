"""Future milestone: context selection and complete-prompt token budgeting.

No implementation yet. Use the exact generation model's tokenizer AND chat
template, never the embedding tokenizer or a character/token estimate.

For this project, interpret 6K conservatively as 6,000 tokens, including reserved
output. The invariant will be: full_serialized_input_tokens + max_output_tokens
<= 6,000. Count instructions, question, history, source labels, delimiters, and
all special/chat-template tokens. Re-render and count after selecting chunks.
Reject an oversized fixed prompt even if there are no retrieved chunks. If the
runtime's serialization cannot be counted reliably, fail instead of claiming
the bound is enforced. No LLM is called in the current MVP.
"""
