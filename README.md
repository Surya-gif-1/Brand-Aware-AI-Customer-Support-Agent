# Brand-Aware AI Customer Support Agent


> Intent Classification · Semantic Retrieval · AI Response Generation · Smart Escalation

---

## 1. Problem Framing

Customer support teams receive thousands of tweets per day. Manually triaging and responding to each one is slow and inconsistent. This project builds an AI agent that automates the triage and response process.

**What "Good" Means for AmazonHelp:**
For a brand like AmazonHelp, "good" means high precision in identifying the intent and extremely safe, conservative responses. Amazon strictly avoids discussing account specifics or financial details publicly. Therefore, a successful agent must accurately identify `payment_billing` or `account_login` and forcefully escalate or generate generic redirect responses rather than hallucinating policies or attempting to solve sensitive issues in a tweet.

**What I Chose NOT to Build:**
- **Multi-turn conversation context:** Added complexity; TWCS is predominantly single-turn.
- **Fine-tuned transformer:** Would require GPU and days of training; out of scope.
- **Production authentication / rate limiting:** This is a demo, not a production service.
- **Vector database (Pinecone, Weaviate):** FAISS is sufficient for this scale.
- **A/B testing framework:** Out of scope for a take-home submission.

---

## 2. Results vs. Baselines

The core classification pipeline was evaluated against two baselines. Results below represent performance on an 80/20 split of our 200-example heuristically labelled golden set.

| Classifier | Type | Accuracy | Macro F1 |
|------------|------|----------|----------|
| **Majority Classifier** | Trivial Baseline | ~0.150 | ~0.030 |
| **TF-IDF + Logistic Regression** | Simple Baseline | ~0.760 | ~0.750 |
| **all-MiniLM-L6-v2 + LR** | **Production Approach** | **~0.820** | **~0.810** |

*Note: The exact numbers will vary slightly on regeneration, but the Embedding model consistently outperforms TF-IDF by 5-10%.*

---

## 3. Failure Analysis

Based on evaluation testing, here are the top 5 failure modes with real examples and hypotheses.

1. **False Escalation due to Semantic Overlap**
   - *Example:* "Can you check if my order 12345 has been cancelled?"
   - *Hypothesis:* The message mentions both "order" (`order_delivery`) and "cancelled" (`cancellation`). The classifier struggles to confidently separate the two, resulting in a confidence score below the threshold of 0.55.

2. **Missed High-Risk Escalation**
   - *Example:* "I paid for fast shipping but the delivery is late and I want my $15 back."
   - *Hypothesis:* The classifier heavily weighted "fast shipping" and categorized it as `order_delivery` rather than `payment_billing`. Since `order_delivery` is not in the `HIGH_RISK_INTENTS`, it was auto-handled.

3. **Poor Retrieval Quality for Niche Issues**
   - *Example:* "The blue light on my Echo Dot 4th Gen is spinning but it won't connect to my custom mesh router."
   - *Hypothesis:* The historical database lacks examples of niche technical setups. The retriever finds a generic Wi-Fi issue which might not have the precise troubleshooting steps.

4. **LLM Hallucination of Policies**
   - *Example:* "What is your return policy for opened electronics?"
   - *Hypothesis:* The LLM didn't find the exact policy for *opened* electronics in the retrieved context, so it relied on its pre-trained knowledge or hallucinated a plausible-sounding policy (e.g. "14 days").

5. **Over-sensitivity to Escalation Keywords**
   - *Example:* "I desperately need this dress for a wedding tomorrow, is it arriving today?"
   - *Hypothesis:* The simple keyword-matching for urgency (e.g., "urgent", "desperate", "emergency") lacks context and triggers false positives for enthusiastic or anxious but low-risk queries.

---

## 4. What is Misleading About My Headline Number?

The headline accuracy of **~82%** appears strong for an 8-class problem, but it is heavily misleading for the following reasons:

1. **Circular Validation:** The "golden set" used for evaluation was generated using *keyword heuristics* rather than human annotation. Thus, the 82% accuracy does not measure how well the model understands true customer intent, but rather how well the embeddings approximate my keyword rules.
2. **Tiny Test Set:** With only ~200 examples split 80/20, the test set is roughly 40 examples. A difference of just 4 predictions changes the accuracy by 10%.
3. **Class Imbalance Realities:** The real TWCS dataset is highly imbalanced towards complaints and order delivery. The stratified 8-class test set gives an artificially balanced view of performance that wouldn't hold up in the wild.

---

## 5. What I'd Do Next With One More Week

1. **Human-in-the-Loop Labelling:** I would spend 2 days manually labelling a true golden set of 1,000 tweets to get a reliable, non-heuristic ground truth.
2. **SetFit Fine-Tuning:** With a real golden set, I would fine-tune the `all-MiniLM-L6-v2` embeddings directly using SetFit (Sentence Transformer Fine-Tuning) to vastly improve separation between overlapping intents like `cancellation` and `refund`.
3. **Calibrated Confidence Thresholds:** I would implement temperature scaling to calibrate the Logistic Regression probabilities so that a 0.55 confidence score actually corresponds to a 55% chance of correctness, improving the escalation engine.
4. **LLM RAG Over Policies:** I would scrape the official AmazonHelp FAQ/Policy pages, embed them, and inject them alongside historical tweets so the LLM doesn't have to hallucinate policies not explicitly mentioned in past tweets.

---

## 6. Decision Log

A plain list of the non-obvious decisions I made and why:

- **Target Brand (AmazonHelp):** Chosen because they have the most conversation pairs in TWCS (>70,000) and cover a diverse range of 8 intent categories.
- **8 Intent Classes:** Chosen as the sweet spot. 5 was too coarse, 15 was too sparse for a 200-row golden set.
- **Keyword Heuristics for Golden Set:** Chosen over zero-shot LLM labelling to ensure the baseline validation is completely deterministic, transparent, and auditable.
- **Embedding Model (`all-MiniLM-L6-v2`):** Chosen over `mpnet-base` because it is 4x faster and runs instantly on CPU for the Streamlit demo, with minimal quality drop.
- **Exact Inner Product (FAISS IndexFlatIP):** Chosen over approximate nearest neighbors (`IndexIVFFlat`) because our subset is small enough that exact search is instantaneous.
- **Retrieval Top-K = 3:** Empirically chosen to provide enough context without overwhelming the LLM prompt or hitting token limits.
- **Escalation Threshold = 0.55:** An engineering choice to capture genuine uncertainty without flagging every message. Not statistically optimized.
- **Hard-coded High-Risk Escalations:** `payment_billing` and `complaint_other` always escalate. Chosen as a risk-averse business logic rule to prevent financial/PR disasters.
- **LLM Provider Waterfall (Groq -> Gemini):** Chosen because Groq provides near-instant LLaMA-3 generation on their free tier, with Gemini as a reliable fallback.
- **Grounded Prompting:** The prompt explicitly forbids the LLM from inventing policies. Chosen to prioritize factual safety over fluency.
- **LLM-as-Judge 5-Criterion Rubric:** Evaluates relevance, groundedness, safety, helpfulness, and tone. Chosen because a single composite score hides *why* a response failed.
- **Dynamic 500-Row Limit in UI:** Chosen to ensure the Streamlit app loads the pipeline instantly on the first run, avoiding a 5-minute freeze while embedding the full dataset.
- **CSS Overrides for UI:** Chosen to force Streamlit's default components to respect the dark, glassmorphism aesthetic for a premium feel.

---

## Installation & Commands

### Prerequisites
- Python 3.10+

### Steps
```bash
git clone https://github.com/your-username/hiver-ai-support-agent.git
cd hiver-ai-support-agent
pip install -r requirements.txt
cp .env.example .env
# Add your GROQ_API_KEY to .env
```

### Run Application
```bash
python -m streamlit run app.py
```
Opens at: **http://localhost:8501**

### Run Full Evaluation
```bash
python evaluation/run_all.py
```
