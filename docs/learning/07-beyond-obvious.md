# Complete LLM Parameters & AI Engineering Roadmap

> **Who this is for**: Someone who has completed the 6-experiment series and wants to know what comes next.  
> **What this covers**: Every parameter category you haven't learned yet, an honest assessment of what matters vs. what's noise, and a suggested learning order for becoming a well-rounded AI engineer.

---

## First: What You Already Know (And Why It's the Hard Part)

!!! success "Strong foundation"
    You already covered the highest-impact sampling controls used in day-to-day tuning.

The 6 experiments covered what most engineers spend 80% of their tuning time on:

```
✅ Temperature         — the master dial
✅ Top-p               — adaptive nucleus filtering
✅ Top-k               — fixed-rank filtering
✅ Repetition penalty  — anti-loop pressure
✅ Combined configs    — how they interact
✅ Task-level tuning   — the calibration process
```

Everything below builds on this foundation. If you don't understand these six, the rest won't stick. If you do, the rest becomes logical extension.

---

## The Parameters You Haven't Covered Yet

### Category 1: Input-Side Parameters
*These control what goes INTO the model before any sampling happens.*

!!! info "Why this category matters"
    Input-side controls shape the problem before decoding even starts. In production, these often matter as much as sampling parameters.

---

#### 1.1 Max Tokens (also called max_new_tokens or max_length)

**What it is**: The maximum number of tokens the model is allowed to generate before it's forced to stop.

**Why it matters**: Without this limit, a model might generate 10,000 tokens for a task that needs 50. This wastes compute and money, and can produce rambling output.

**The two variants:**
- `max_tokens` in most APIs: total tokens including the prompt
- `max_new_tokens`: only the generated tokens, not the input

**Common values by task:**

| Task                 | Typical max_new_tokens  |
|:---------------------|:------------------------|
| Classification label | 5–10                    |
| Factual QA answer    | 100–300                 |
| Summarization        | 150–500                 |
| Code function        | 200–500                 |
| Full article         | 800–2000                |
| Extended reasoning   | 2000–8000               |

**The trap**: Setting max_tokens too low causes the model to cut off mid-sentence. Setting it too high wastes money and can cause the model to pad its output unnecessarily.

!!! warning "Common failure mode"
    Most teams initially over-allocate token budgets "just in case," then discover avoidable cost and latency regressions.

**Learning priority: High.** You'll set this on every single API call you ever make.

---

#### 1.2 System Prompt

**What it is**: A special message at the start of the conversation that sets the model's persona, rules, and task context. Unlike the user message, the system prompt is usually hidden from the end user.

**Why it matters**: The system prompt is often more important than any sampling parameter. A well-written system prompt can make a T=0.7 model behave as precisely as a T=0.2 model for a specific task, because it shifts the logit distribution *before* sampling even starts.

**Example:**

```
System: You are a JSON-only classifier. You must respond ONLY with valid JSON.
        Never include any text outside the JSON object.
        
User: Is this email spam? "Win a free iPhone now!!!"

Model: {"classification": "spam", "confidence": 0.97}
```

Without the system prompt, the model might say "Yes, that looks like spam. Here's my analysis..." — technically correct but not machine-parseable.

**The key insight**: System prompts constrain the *content* of outputs; sampling parameters constrain the *distribution* of tokens. Both are necessary tools.

!!! tip "Practical pattern"
    Lock format and policy constraints in the system prompt, then use temperature/top-p to tune style and variability.

**Learning priority: Critical.** More impactful than any single sampling parameter.

---

#### 1.3 Context Window / Context Length

**What it is**: The maximum number of tokens (input + output combined) the model can process in a single call. Modern models range from 4,096 tokens to 1,000,000+ tokens.

**Why it matters**: If your input exceeds the context window, the model either errors out or silently truncates your input — often cutting the most important parts (the beginning or middle).

**Tokens vs words**: Roughly 1 token ≈ 0.75 words in English. 4,096 tokens ≈ 3,000 words ≈ 6 pages.

**Context window sizes by current generation (as of mid-2025):**

| Model family        | Approximate context |
|:--------------------|:--------------------|
| GPT-4o              | 128,000 tokens      |
| Claude Sonnet 4.6   | 200,000 tokens      |
| Gemini 1.5 Pro      | 1,000,000 tokens    |
| Smaller/open models | 4,096–32,768 tokens |

**Why "more context" isn't always better**: Models often exhibit "lost in the middle" behavior — they pay more attention to the beginning and end of the context and miss information in the middle. Long contexts also cost more compute.

!!! note "Design implication"
    Context quality usually beats context quantity. Retrieval quality, chunking, and ordering often outperform simply increasing tokens.

**Learning priority: High.** Determines what tasks are even possible with a given model.

---

#### 1.4 Stop Sequences (also called stop tokens)

**What it is**: A list of strings that, when generated, immediately halt generation. Like a hard stop sign in the output.

**Example:**

```python
response = client.messages.create(
    model="claude-sonnet-4-6",
    messages=[{"role": "user", "content": "Generate 3 ideas:"}],
    stop_sequences=["\n4."]  # Stop before the 4th item
)
```

The model will generate "1. ... 2. ... 3. ..." and then stop when it was about to write "4."

**Common uses:**
- Enforce exact output length (stop after N items)
- Prevent the model from continuing past a delimiter ("END_OF_JSON", "```")
- Create structured conversation turns

**Learning priority: Medium-High.** Essential for structured output and agentic pipelines.

!!! example "Quick win"
    If you need exactly N list items, stop on the token that would start item N+1.

---

### Category 2: Decoding Strategy Parameters
*These control the algorithm used for selecting tokens — beyond simple sampling.*

!!! abstract "Category view"
    These techniques change *how* candidates are chosen, not just how strongly probabilities are scaled.

---

#### 2.1 Beam Search

**What it is**: Instead of sampling one token at a time, beam search keeps the top-B (beam width) *sequences* at each step, exploring B paths simultaneously, then returns the highest-scoring complete sequence.

**How it differs from sampling:**

```
Sampling (what you learned):
  Step 1: Sample token A or B or C (one is chosen randomly)
  Step 2: Given that choice, sample next token
  → Each run can produce a different sequence

Beam search (new concept):
  Step 1: Keep top-3 candidate sequences: [A...], [B...], [C...]
  Step 2: Extend each by one token, keep top-3 again
  → Final output is the single highest-scoring complete sequence
```

**Pros**: Tends to produce globally more coherent sequences; better for translation, structured outputs.

**Cons**: More compute (explores multiple paths); can produce generic, "safe" outputs; doesn't work well for creative tasks.

!!! question "When does beam search shine?"
    Prefer it when you want the single most likely complete sequence (translation, constrained summarization), not diverse ideation.

**Beam width tradeoff:**
- Beam width=1: greedy decoding (same as sampling with T near 0)
- Beam width=4: common for translation, summarization
- Beam width=10+: diminishing returns, much higher compute

**When to use over sampling**: Translation, document summarization where you want the single best output, structured extraction where you need the highest-confidence parse.

**Learning priority: Medium.** You'll encounter it in academic papers and some APIs (especially for NLP pipelines). Less relevant for LLM chat/completion APIs where sampling dominates.

---

#### 2.2 Min-p Sampling

**What it is**: A newer alternative to top-p. Instead of setting a fixed cumulative probability threshold, min-p sets a *minimum probability relative to the top token*. Any token with probability less than `min_p * P(top_token)` is excluded.

**Formula:**
```
If P(token_i) < min_p * P(top_token):
    exclude token_i
```

**Why this is interesting:**

In top-p, if the top token has 80% probability and p=0.9, you add the 80% token and then need 10% more — maybe including tokens with 5% and 3% probability. But if the top token has only 20% (uncertain model), you might need to include 15 tokens to reach 90%.

In min-p with min_p=0.05:
- Confident model (top=80%): threshold = 0.05 × 0.80 = 0.04 → excludes tokens below 4% → small nucleus
- Uncertain model (top=20%): threshold = 0.05 × 0.20 = 0.01 → excludes tokens below 1% → larger nucleus

The threshold *scales with the top token's probability*, so it's automatically more restrictive when the model is confident and more permissive when it's uncertain — but in a cleaner way than top-p.

**Current status**: Available in some libraries (llama.cpp, some Hugging Face configurations). Not yet in major APIs as of mid-2025, but gaining traction in the open-source community.

**Learning priority: Low-Medium.** Worth knowing exists; not yet widely deployed.

!!! note "Adoption snapshot"
    Min-p is especially common in local/open-source inference stacks and worth tracking for future API support.

---

#### 2.3 Typical Sampling

**What it is**: Instead of selecting the highest-probability tokens (top-p, top-k), typical sampling selects tokens that are "typical" — whose information content is closest to the expected entropy of the distribution. Outlier tokens (both surprisingly high and surprisingly low probability) can be excluded.

**The intuition**: In natural language, very high-probability tokens ("the", "is") and very surprising tokens (obscure words appearing out of nowhere) are both forms of non-typical behavior. Typical sampling tries to select tokens that reflect the model's genuine "expectation" about the next word.

**Current status**: Mainly a research technique. Rarely available in production APIs.

**Learning priority: Low.** Academic interest; understanding it deepens your intuition about entropy but you won't use it in production soon.

!!! info "Why still learn it"
    Typical sampling builds strong intuition for entropy-aware filtering even when your provider does not expose it directly.

---

#### 2.4 Contrastive Search

**What it is**: A decoding method that balances two goals at each step: (1) the token should have high probability according to the model, and (2) the token should be *different* from what was recently generated (measured by cosine similarity in embedding space).

**Why this is interesting**: It's like repetition penalty but operating at the semantic level, not the token level. It discourages repeating the same *meaning* even if different words are used.

**Current status**: Research technique, implemented in Hugging Face transformers. Not in standard APIs.

**Learning priority: Low.** Good conceptual knowledge; practical use is limited.

!!! tip "Mental model"
    Think of contrastive search as semantic anti-repetition: avoid saying the same thing, not just the same token.

---

### Category 3: Generation Quality Controls
*These affect the quality of the generation process itself.*

!!! info "Production relevance"
    These controls are often the difference between "works in demo" and "reliable under real traffic."

---

#### 3.1 Logit Bias

**What it is**: A dictionary mapping specific token IDs to logit adjustments. You can manually boost or suppress specific tokens.

**Example:**

```python
# Force the model to never say "I cannot" (suppress those tokens)
# Or always prefer a specific output format

logit_bias = {
    "1734": -100,   # token ID for "cannot" → effectively banned
    "198": 5,       # token ID for "\n" → boosted (force more newlines)
}
```

**Why it matters for engineers**: Lets you hard-wire constraints without changing the prompt. Useful for:
- Preventing specific words (competitor names, offensive terms)
- Forcing a specific format character (JSON braces, markdown symbols)
- Restricting output to a small vocabulary (for classification tasks where only "yes"/"no"/"maybe" should appear)

**The catch**: Token IDs are model-specific. The token ID for "cannot" in GPT-4 is different from Claude's tokenizer. You need to look up IDs for your specific model.

!!! warning "Portability risk"
    Logit-bias rules rarely transfer across model vendors without retokenizing and remapping IDs.

**Learning priority: Medium.** Very useful for production systems with strict output requirements.

---

#### 3.2 Seed (Reproducibility)

**What it is**: A random seed that makes sampling deterministic. Given the same seed, prompt, and parameters, the model produces the exact same output every time.

**Why it matters**: 
- Debugging: reproduce exact outputs that caused problems
- Testing: write unit tests for model outputs that need to be consistent
- Research: reproduce exact results from a paper or experiment

**The catch**: Seed-based reproducibility is only guaranteed within the same model version, on the same hardware. Model updates or infrastructure changes can break reproducibility even with the same seed.

!!! danger "Debugging pitfall"
    Treat seeds as environment-scoped reproducibility, not universal reproducibility across providers or model revisions.

**Example:**

```python
response1 = client.generate(prompt="Tell me a story", seed=42, temperature=0.9)
response2 = client.generate(prompt="Tell me a story", seed=42, temperature=0.9)
# response1 == response2  ← guaranteed (on same model/hardware)
```

**Learning priority: Medium-High.** Essential for testing and debugging.

---

#### 3.3 n (Number of Completions)

**What it is**: How many independent completions to generate for a single prompt. Instead of calling the API once, you get n different outputs in one call.

**Why it matters**: 
- pass@k evaluation: generate k code solutions, check if any pass test cases
- Best-of-n: generate n options, use a secondary model or rule to pick the best
- Diversity sampling: show users multiple options to choose from

**Example:**

```python
responses = client.generate(
    prompt="Write a headline for this article",
    n=5,          # Get 5 different headlines
    temperature=0.9
)
# Pick the best headline from the 5 options
```

**Cost**: Generating n=5 completions costs approximately 5× as much as n=1.

!!! warning "Cost guardrail"
    Best-of-n improves quality variance handling, but it scales spend quickly. Pair with strict token budgets and eval-based selection.

**Learning priority: Medium.** Important for evaluation pipelines and any "best of many" approach.

---

#### 3.4 Presence Penalty and Frequency Penalty (API-Level)

You covered these conceptually in Experiment 5 (presence vs frequency penalty). The API-level variants in OpenAI-style APIs use additive penalties on a scale of -2.0 to 2.0 (not multiplicative like the experiment used).

**Positive values**: Discourage repetition (as covered)  
**Negative values**: *Encourage* repetition — useful for tasks where you want the model to stay on a specific topic or use consistent terminology

**The negative value use case** (often missed):

If you're generating a technical document and want the model to consistently use the same terminology (e.g., always say "user" not "customer" or "client"), a small negative frequency penalty encourages the model to reuse words it has already used.

**Learning priority: Medium.** Extension of what you already know from Experiment 5.

!!! example "Underused trick"
    Slight negative frequency penalty can stabilize terminology in long-form technical writing.

---

### Category 4: Structured Output Parameters
*These are newer capabilities for getting reliably formatted outputs.*

!!! success "High leverage area"
    Structured output controls are among the biggest reliability upgrades available in modern LLM APIs.

---

#### 4.1 JSON Mode / Structured Outputs

**What it is**: A model-level constraint that forces every generated token to conform to valid JSON (or a specific JSON schema). The model literally cannot produce tokens that would break JSON syntax.

**How it works internally**: The allowed token set at each position is restricted to only tokens that are valid continuations of a valid JSON document. It's like top-k, but the "k" changes at every position based on the JSON grammar.

**Why this matters enormously**: The #1 reliability problem in LLM production systems is inconsistent output format. JSON mode eliminates this for structured data tasks.

**Example:**

```python
response = client.messages.create(
    model="claude-sonnet-4-6",
    messages=[...],
    response_format={"type": "json_object"}  # Forces valid JSON output
)
```

**Learning priority: Very High.** If you build production LLM systems, you will use this constantly.

!!! tip "Implementation pattern"
    Combine schema-constrained output with post-parse validation and retries for robust pipelines.

---

#### 4.2 Tool Use / Function Calling

**What it is**: Instead of generating free text, the model generates structured calls to predefined functions. The model says "I need to call `get_weather(city='London')` to answer this question" rather than hallucinating an answer.

**Why this matters**: Allows LLMs to interact with external systems — APIs, databases, search engines, code interpreters — in a reliable, structured way.

**The parameters involved:**
- Tool definitions (name, description, parameter schema)
- tool_choice: "auto" (model decides when to call tools), "required" (must call a tool), or a specific tool name

**Learning priority: Very High.** Tool use is the foundation of AI agents. Understanding this is essential for building anything beyond simple chatbots.

!!! info "Architecture shift"
    Tool calling changes LLMs from "text generators" into orchestrators that can read state, act, and recover from tool errors.

---

### Category 5: Advanced / Research Parameters
*These are less commonly tuned in practice but important to know about.*

!!! note "How to use this section"
    Learn these for conceptual range and ecosystem awareness; apply only when your stack explicitly supports them.

---

#### 5.1 Classifier-Free Guidance (CFG) Scale

**What it is**: Originally from image generation (Stable Diffusion), CFG scale amplifies how much the model follows your prompt vs. its unconditioned behavior.

**In language models**: Some research models use CFG to make the model more strictly follow the instruction. Higher CFG = more literal adherence to prompt; lower CFG = more natural but potentially off-topic.

**Current status**: Not available in standard LLM APIs. Common in image generation APIs (Midjourney, SDXL). Some research language models use it.

**Learning priority: Low for LLMs, High for image generation.** Important if you move into multimodal or image generation work.

---

#### 5.2 Mirostat Sampling

**What it is**: A method that dynamically adjusts temperature throughout generation to maintain a target entropy level. Instead of fixing T and watching entropy vary, Mirostat fixes target entropy and adjusts T accordingly.

**The idea**: Natural language has roughly consistent entropy at each position. Mirostat tries to maintain this consistency, avoiding the problem where a fixed temperature produces very high entropy in uncertain positions and very low entropy in certain positions.

**Parameters**:
- `mirostat_mode`: 0 (off), 1 (Mirostat), 2 (Mirostat 2.0)
- `mirostat_tau`: target entropy (typically 5.0)
- `mirostat_eta`: learning rate for adjustment (typically 0.1)

**Current status**: Available in llama.cpp and Ollama. Not in major cloud APIs.

**Learning priority: Low.** Interesting conceptually; niche in practice.

!!! abstract "Core idea"
    Fixed entropy target, adaptive temperature control.

---

#### 5.3 Speculative Decoding

**What it is**: Not a quality parameter — a speed optimization. A small "draft" model generates several tokens at once; the large target model verifies them in parallel. If the draft tokens are good, they're accepted. If not, the target model regenerates from the first bad token.

**Why it matters**: Can achieve 2–3× inference speedup with no quality loss (mathematically equivalent to running the target model directly).

**Parameters**: Usually handled automatically by inference infrastructure. No user-facing knobs.

**Learning priority: Low for application engineers, High for ML infrastructure engineers.** You need to know it exists and what it does; you probably won't configure it directly.

!!! info "Important distinction"
    Speculative decoding is primarily a latency/cost optimization, not a behavior-quality tuning knob.

---

#### 5.4 KV Cache Control

**What it is**: When processing long contexts, LLMs reuse computations from previous tokens via the "key-value cache." Some systems let you control cache behavior — whether to use a cached prefix, how long to retain the cache, etc.

**Why it matters for cost**: Processing the same system prompt repeatedly in every API call is expensive. With prefix caching, you pay full price for the system prompt once, then a fraction for subsequent calls with the same prefix.

**Learning priority: Medium for production engineers.** Directly affects cost and latency. Claude, GPT-4, and Gemini all offer some form of prompt caching.

---

### Category 6: Inference Infrastructure Parameters
*These don't change the model's outputs directly but matter enormously in production.*

!!! warning "Operational reality"
    Many production incidents are infra-parameter issues (throughput, memory, cache policy), not prompt issues.

---

#### 6.1 Batching and Throughput

**What it is**: Processing multiple requests simultaneously to improve throughput (requests per second). Individual latency may increase slightly, but total cost per request decreases.

**Learning priority: Medium.** Relevant when you're serving multiple users.

---

#### 6.2 Quantization

**What it is**: Reducing the numerical precision of model weights from 32-bit floats to 16-bit, 8-bit, or even 4-bit representations. This reduces memory requirements and increases speed at the cost of some quality.

**Common formats**: FP16, BF16, INT8, INT4, GGUF (for llama.cpp), AWQ, GPTQ

**Why it matters**: Running a 70B parameter model in full precision requires ~140GB of GPU memory. At 4-bit quantization, the same model fits in ~35GB — accessible to single-GPU setups.

**Learning priority: Medium-High if you run local/open-source models. Low if you only use cloud APIs.**

!!! note "Rule of thumb"
    Evaluate quality on task-specific benchmarks before rolling aggressive quantization into user-facing paths.

---

## The Parameters That Matter Most — Priority Ranking

If you're new and want to know where to focus your learning:

!!! success "Use this as your checklist"
    Tier 1 -> Tier 2 -> Tier 3 is designed to maximize practical impact per hour of study.

### Tier 1: Learn These Now (Immediate Practical Impact)
```
✅ Already covered: Temperature, Top-p, Top-k, Repetition penalty
🔲 Max tokens / max_new_tokens
🔲 System prompt design
🔲 JSON mode / Structured outputs
🔲 Tool use / Function calling
🔲 Stop sequences
🔲 Context window management
```

### Tier 2: Learn These Next (3–6 Months)
```
🔲 Seed and reproducibility
🔲 n completions (best-of-n, pass@k)
🔲 Logit bias
🔲 Prompt caching / KV cache
🔲 Beam search (conceptual)
🔲 Presence/frequency penalty (API variants)
```

### Tier 3: Learn These When Relevant (As Needed)
```
🔲 Min-p sampling
🔲 Mirostat
🔲 Quantization (if running local models)
🔲 Speculative decoding (if doing infrastructure work)
🔲 CFG scale (if doing image generation work)
🔲 Contrastive search
🔲 Typical sampling
```

---

## The Broader AI Engineering Skill Map

Parameters are just one piece. Here is the full picture of what a well-rounded AI engineer knows, organized by layer:

!!! abstract "Big picture"
    Sampling is one layer in a full system lifecycle: design, generation, evaluation, retrieval, adaptation, and operations.

```
┌──────────────────────────────────────────────────────┐
│                  APPLICATION LAYER                   │
│  System design, product thinking, user experience    │
├──────────────────────────────────────────────────────┤
│                  PROMPT ENGINEERING                  │
│  System prompts, few-shot examples, chain-of-thought │
│  ReAct patterns, structured output design            │
├──────────────────────────────────────────────────────┤
│              SAMPLING & GENERATION                   │  ← You are here
│  Temperature, top-p, top-k, rep penalty,             │
│  beam search, structured outputs, tool use           │
├──────────────────────────────────────────────────────┤
│               EVALUATION & METRICS                   │
│  ROUGE, BERTScore, pass@k, human eval,               │
│  LLM-as-judge, benchmark interpretation              │
├──────────────────────────────────────────────────────┤
│            RAG & RETRIEVAL SYSTEMS                   │
│  Vector databases, chunking, embedding models,       │
│  reranking, hybrid search, context construction      │
├──────────────────────────────────────────────────────┤
│              FINE-TUNING & ADAPTATION                │
│  LoRA, QLoRA, instruction tuning, RLHF concepts,     │
│  when fine-tuning beats prompting                    │
├──────────────────────────────────────────────────────┤
│            MODEL INTERNALS (OPTIONAL)                │
│  Transformers, attention, tokenization,              │
│  positional encoding, training objectives            │
└──────────────────────────────────────────────────────┘
```

---

## Suggested Learning Order After Experiment 6

!!! tip "Execution strategy"
    Build one small project per phase, and attach an eval harness early so each change is measurable.

### Month 1–2: Prompt Engineering (Highest ROI)

Prompt engineering improvements often outperform parameter tuning. Learn:

1. **Zero-shot vs few-shot prompting** — when to provide examples and when not to
2. **Chain-of-thought prompting** — asking the model to reason step by step before answering
3. **System prompt design** — persona, constraints, output format specification
4. **Structured output prompting** — getting reliable JSON/XML/Markdown without JSON mode
5. **ReAct pattern** — Reasoning + Acting, the foundation of tool-using agents

**Recommended resource**: Anthropic's prompt engineering guide at docs.anthropic.com

---

### Month 2–3: Tool Use and Agents (High Impact)

Tool use transforms LLMs from text generators into systems that can take actions:

1. **Function calling basics** — defining tools, parsing tool calls, returning results
2. **Multi-step agents** — models that call tools repeatedly to complete a task
3. **ReAct and plan-and-execute patterns** — structured approaches to multi-step reasoning
4. **Error handling in agents** — what happens when tools fail or return unexpected results

**Key concept to master**: The difference between a "single-turn" LLM call and a "multi-turn agent loop." Most production AI systems are the latter.

---

### Month 3–4: Evaluation (Underrated, Critical)

Most engineers skip evaluation and pay for it later:

1. **Task-specific metrics**: accuracy, F1, ROUGE, BLEU, pass@k, exact match
2. **LLM-as-judge**: using a model to score another model's outputs (faster than human eval)
3. **Evals design**: building test sets that are representative and hard to game
4. **A/B testing for LLM systems**: safely deploying parameter changes with measurement

**Key insight**: If you can't measure it, you can't improve it. Every parameter change you make should be validated against a metric — exactly what Experiment 6's checklist advised.

!!! danger "Skip this and you guess"
    Teams that skip evals often ship regressions they cannot detect or explain.

---

### Month 4–6: RAG Systems (Practical for Most Builders)

Retrieval-Augmented Generation is how you make LLMs work with your own data:

1. **Embeddings and vector search** — converting text to numbers that represent meaning
2. **Chunking strategies** — how to split long documents for retrieval
3. **Reranking** — improving retrieval quality after the initial search
4. **Context construction** — how to format retrieved documents in the prompt
5. **Hybrid search** — combining keyword search with semantic search

**Why this matters**: The context window size and prompt structure (things you now understand well from Experiment 6) are central to RAG system design.

---

### Month 6–12: Fine-Tuning (When You Need It)

Fine-tuning is often *not* the right answer, but when it is, it's powerful:

1. **When fine-tuning beats prompting**: consistent format requirements, domain-specific vocabulary, extreme latency or cost constraints
2. **LoRA and QLoRA**: efficient fine-tuning methods that work on consumer hardware
3. **Instruction tuning**: how models learn to follow instructions
4. **RLHF concepts**: why models are helpful and harmless (important for understanding model behavior)

**Important caveat**: Many engineers fine-tune too early. Exhaust prompt engineering and parameter tuning first. Fine-tuning is expensive and difficult to iterate on quickly.

!!! warning "Decision gate"
    Fine-tune only after prompt/system/tooling/eval baselines are strong and well-instrumented.

---

## The One-Page Summary: What Matters When

!!! info "Fast triage"
    Use the guide below as a first-pass diagnosis map before changing multiple knobs at once.

```
┌─────────────────────────────────────────────────────────────┐
│                    DEBUGGING GUIDE                          │
│                                                             │
│  Output is wrong / hallucinating:                           │
│    → Fix the prompt first (not the parameters)              │
│    → Add retrieval (RAG) for factual grounding              │
│    → Fine-tune only as a last resort                        │
│                                                             │
│  Output is too random / incoherent:                         │
│    → Lower temperature                                      │
│    → Tighten top-p                                          │
│    → Check for high repetition penalty (back off)           │
│                                                             │
│  Output is too repetitive / robotic:                        │
│    → Raise temperature slightly                             │
│    → Add light repetition penalty (start at 1.1)            │
│    → Check if top-k is too restrictive                      │
│                                                             │
│  Output format is inconsistent:                             │
│    → Use JSON mode / structured outputs                     │
│    → Add stop sequences                                     │
│    → Improve system prompt format instructions              │
│                                                             │
│  Output is too long / cuts off:                             │
│    → Adjust max_tokens                                      │
│    → Check context window usage                             │
│    → Use stop sequences to enforce length                   │
│                                                             │
│  Output is slow / expensive:                                │
│    → Reduce max_tokens                                      │
│    → Use prompt caching (same system prompt across calls)   │
│    → Consider a smaller model (check if quality holds)      │
└─────────────────────────────────────────────────────────────┘
```

---

## Honest Assessment: What Makes a Good AI Engineer

!!! success "Core principle"
    High-performing AI engineering is disciplined diagnosis + measurement + system-level thinking.

After studying parameters, the skill that separates good AI engineers from great ones is not knowing more parameters. It's:

**1. Diagnosis first, tuning second.**

Most problems in LLM systems are not parameter problems. They're prompt problems, data problems, or task-definition problems. The best AI engineers spend 80% of their time understanding *why* the model is failing before changing anything.

**2. Measurement discipline.**

Every change should be testable. Build evals before you build features. A system you can measure is a system you can improve; a system you can't measure is one you're guessing about.

**3. Understanding the full stack.**

The sampling parameters you've learned sit in a larger system: data → training → tokenization → inference → sampling → output → evaluation → feedback. Knowing where parameters fit in that stack helps you know which layer is actually causing your problem.

**4. Knowing when NOT to use LLMs.**

Some tasks are better solved with traditional software (rule-based systems, databases, deterministic algorithms). A great AI engineer knows when to use an LLM, when to use a smaller model, and when to use no model at all.

---

## What to Build to Learn Faster

Reading deepens knowledge; building ingrains it. Here are four projects that cover the most ground:

| Project                       | What you'll learn                                                                       |
|:------------------------------|:----------------------------------------------------------------------------------------|
| **Sentiment classifier**      | Prompt engineering, JSON mode, evaluation metrics, parameter tuning for precision tasks |
| **Document Q&A chatbot**      | RAG basics, context window management, retrieval, hallucination                         |
| **Code generation assistant** | Tool use, pass@k evaluation, diverse sampling, structured output                        |
| **Multi-step research agent** | ReAct pattern, tool chaining, error handling, agent reliability                         |

Build them in this order. Each one introduces new concepts while reusing what you already know.

!!! example "Portfolio strategy"
    Turn each project into a mini case study: problem, baseline, parameter choices, eval results, and lessons learned.

---

*You have completed the 6-Experiment LLM Parameter Learning Path.*  
*The roadmap above is your next map. Start with Tier 1 parameters and Month 1–2 prompt engineering.*  
*Everything else follows naturally from there.*