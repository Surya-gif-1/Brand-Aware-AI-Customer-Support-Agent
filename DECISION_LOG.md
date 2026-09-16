# Decision Log

## Engineering Decisions

---

### Decision 1: Brand Selection — AmazonHelp

**Decision:** Select AmazonHelp as the target brand.  
**Why:** AmazonHelp has the most conversation pairs in TWCS (>70,000 responses), covers a wide diversity of customer issues, and is a well-known brand that evaluation reviewers can contextualise.  
**Alternatives considered:** SpotifyCares, AppleSupport, XboxSupport.  
**Reason for final choice:** Volume and diversity. AmazonHelp covers shipping, payments, accounts, technical issues, returns, cancellations — mapping cleanly to 8 distinct intents.  
**Trade-off:** Amazon's support style (directing to DM / website) may limit response diversity in the retrieval pool.

---

### Decision 2: Intent Taxonomy Size — 8 Classes

**Decision:** Use 8 intent categories.  
**Why:** Enough granularity to be useful; small enough for high classifier precision. Derived from keyword frequency analysis of actual AmazonHelp conversations.  
**Alternatives considered:** 5 classes (too coarse), 15 classes (too fine-grained; sparse data per class).  
**Reason for final choice:** 8 classes strike a practical balance for a ~200-example golden set.  
**Trade-off:** Some messages genuinely span multiple intents (e.g. "I want to cancel and get a refund"). These are assigned to the first matching intent, which may be wrong.

---

### Decision 3: Keyword Heuristic Labelling for Golden Set

**Decision:** Use keyword-based heuristics (not model predictions) to assign initial labels.  
**Why:** Model predictions cannot be used as ground truth — this would circularly validate the same model. Keyword matching is transparent and auditable.  
**Alternatives considered:** Crowdsourcing labels (impractical in a take-home setting), zero-shot LLM labelling.  
**Reason for final choice:** Keyword matching is reproducible, explainable, and honest. Rows where heuristics are ambiguous are flagged `needs_review=True`.  
**Trade-off:** Keyword heuristics miss nuanced cases; the golden set needs human review for production use.

---

### Decision 4: Embedding Model — all-MiniLM-L6-v2

**Decision:** Use `all-MiniLM-L6-v2` from sentence-transformers.  
**Why:** Small (80MB), fast on CPU, achieves strong performance on semantic similarity benchmarks. No GPU required.  
**Alternatives considered:** `all-mpnet-base-v2` (better quality but 4x slower), `paraphrase-multilingual-MiniLM-L12-v2` (multilingual, unnecessary here).  
**Reason for final choice:** Speed vs. quality trade-off optimised for a laptop-runnable demo.  
**Trade-off:** Larger models would produce better embeddings but require GPU or long inference time.

---

### Decision 5: FAISS IndexFlatIP for Retrieval

**Decision:** Use FAISS `IndexFlatIP` (exact inner product = cosine for L2-normalised vectors).  
**Why:** Exact search is correct for datasets of this size (<100K vectors). No approximate search errors.  
**Alternatives considered:** `IndexIVFFlat` (approximate, faster for millions of vectors), BM25 keyword retrieval.  
**Reason for final choice:** Exact search is appropriate at this scale. Approximate indexes add complexity without benefit.  
**Trade-off:** Slower at very large scale (>1M vectors).

---

### Decision 6: Retrieval Top-K = 3

**Decision:** Default to 3 historical examples in the retrieval context.  
**Why:** 3 examples provide sufficient grounding without overloading the LLM prompt. Configurable in `config.py`.  
**Alternatives considered:** K=1 (too little context), K=5 (may dilute signal with less relevant examples).  
**Reason for final choice:** Empirically, 2-4 examples is the standard in RAG literature for short generation tasks.  
**Trade-off:** With K=3, less common intents may retrieve off-topic examples if the index is sparse.

---

### Decision 7: Escalation Confidence Threshold = 0.55

**Decision:** Escalate if classifier confidence < 0.55.  
**Why:** LR confidence scores are not calibrated. A threshold of 0.55 represents genuine uncertainty rather than a hard decision.  
**Alternatives considered:** 0.7 (too aggressive, many false escalations), 0.4 (too permissive).  
**Reason for final choice:** 0.55 is a reasonable engineering default. Explicitly documented as a threshold requiring calibration.  
**Trade-off:** This threshold was not optimised against a labelled escalation set; it is an engineering choice.

---

### Decision 8: High-Risk Intents Always Escalate

**Decision:** `payment_billing` and `complaint_other` always escalate regardless of confidence.  
**Why:** Billing errors and strong complaints carry financial and reputational risk that automation should not handle alone.  
**Alternatives considered:** Escalate only on very low confidence for these intents.  
**Reason for final choice:** Risk-averse engineering choice. False escalations cost human time but avoid financial/legal errors.  
**Trade-off:** High escalation rate for payment/billing conversations; may frustrate users waiting for human callback.

---

### Decision 9: LLM Provider Priority — Groq > Gemini > None

**Decision:** Use Groq as primary LLM, Gemini as fallback, graceful degradation if neither is available.  
**Why:** Groq offers extremely fast inference and a generous free tier. Gemini provides a reliable fallback.  
**Alternatives considered:** OpenAI GPT-4 (expensive), local LLM (Ollama) — requires local GPU.  
**Reason for final choice:** Free-tier accessibility is important for a take-home project that evaluators must run.  
**Trade-off:** Free-tier Groq has rate limits; heavy evaluation may hit them.

---

### Decision 10: Prompt Design — Grounded, No Fabrication

**Decision:** The LLM prompt explicitly forbids inventing policies, refund amounts, or order details.  
**Why:** Hallucinated customer-support responses can cause real harm (false promises, incorrect advice).  
**Alternatives considered:** Unconstrained generation (higher fluency, higher risk).  
**Reason for final choice:** Safety-first. The prompt instructs the model to acknowledge uncertainty when evidence is insufficient.  
**Trade-off:** Responses may be more conservative/generic when historical examples are insufficient.

---

### Decision 11: Train/Test Split Strategy

**Decision:** 80/20 stratified split of the golden set for classifier evaluation.  
**Why:** Maintains class distribution in both splits. The retriever is built ONLY on the training split to prevent self-retrieval.  
**Alternatives considered:** Cross-validation (impractical at 200 examples with embedding models due to speed).  
**Reason for final choice:** Simple, transparent, and standard. Fixed random seed for reproducibility.  
**Trade-off:** At ~200 examples, the test set (~40 examples) is small; metrics have high variance.

---

### Decision 12: LLM-as-Judge Rubric (5 Criteria)

**Decision:** Score responses on relevance, groundedness, factual safety, helpfulness, and brand tone.  
**Why:** A single overall score loses information. The 5-criterion rubric identifies which dimension fails.  
**Alternatives considered:** BLEU/ROUGE (reference-free generation evaluation; poor for open-ended support responses), human evaluation (impractical at scale).  
**Reason for final choice:** LLM-as-judge is the current state-of-the-art for evaluating open-ended generation without reference responses.  
**Trade-off:** The judge LLM may itself hallucinate scores. Judge scores are indicative, not ground truth.

---

### Decision 13: What Was Intentionally NOT Built

- **Multi-turn conversation context**: Added complexity; TWCS is predominantly single-turn.
- **Fine-tuned transformer**: Would require GPU and days of training; out of scope.
- **Production authentication / rate limiting**: This is a demo, not a production service.
- **Vector database (Pinecone, Weaviate)**: FAISS is sufficient for this scale.
- **A/B testing framework**: Out of scope for a take-home submission.
