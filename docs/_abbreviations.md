*[Token]: A chunk of text the model reads or emits; token count drives context usage and cost.
*[Logit]: Raw model score before probabilities are computed.
*[Logits]: Raw model scores before probabilities are computed.
*[Softmax]: Function that converts logits into a normalized probability distribution.
*[Temperature]: Scales logits before sampling. Lower values are more deterministic; higher values increase diversity.
*[top_p]: Nucleus sampling: keep the smallest set of tokens whose cumulative probability reaches p.
*[nucleus sampling]: Sampling from the smallest token set whose cumulative probability is at least p.
*[top-k]: Keep only the k highest-probability tokens before sampling.
*[min-p]: Keep tokens with probability at least min_p multiplied by the max token probability.
*[typical-p]: Typical sampling: prefer tokens whose surprise is close to the distribution entropy.
*[Entropy]: Measure of uncertainty in a distribution; higher entropy means more randomness and spread.
*[Decoding]: The process of selecting each next token from model probabilities.
*[Sampling]: Randomized next-token selection from a filtered probability distribution.
*[Top-p]: Nucleus sampling: keep the smallest set of tokens whose cumulative probability reaches p.
*[Top-k]: Keep only the k highest-probability tokens before sampling.
*[Min-p]: Keep tokens with probability at least min_p multiplied by the max token probability.
*[Typical-p]: Typical sampling: prefer tokens whose surprise is close to the distribution entropy.
*[Nucleus sampling]: Sampling from the smallest token set whose cumulative probability is at least p.
*[Rag]: Retrieval Augmented Generation: inject retrieved context into prompts for grounded answers.
*[Structured outputs]: Constrain responses to a schema (for example JSON) so downstream code parses reliably.
*[Determinism]: Behavior where repeated runs produce the same or nearly the same outputs.
*[Repetition penalty]: Penalty that reduces likelihood of already-used tokens to discourage loops.
*[Hallucination]: Fluent model output that is unsupported or incorrect.
*[Prompt]: Input instructions and context sent to the model.
*[Context window]: Maximum number of tokens a model can consider in one request.
*[RAG]: Retrieval Augmented Generation: inject retrieved context into prompts for grounded answers.
*[Structured output]: Constrain responses to a schema (for example JSON) so downstream code parses reliably.
*[JSON Schema]: A formal schema describing expected JSON fields, types, and constraints.
