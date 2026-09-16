# Failure Analysis

This document outlines plausible failure modes for the Brand-Aware AI Customer Support Agent, based on evaluation testing.

## 1. False Escalation due to Semantic Overlap

**Customer Message:** "Can you check if my order 12345 has been cancelled?"
**Predicted Intent:** `cancellation`
**Confidence:** 0.48
**Expected Action:** `AUTO-HANDLE`
**Predicted Action:** `ESCALATE` (Rule: `low_confidence`)

**Hypothesis:** The message mentions both "order" (typical of `order_delivery`) and "cancelled" (typical of `cancellation`). The classifier struggles to confidently separate the two, resulting in a confidence score below the threshold of 0.55.
**Improvement:** Fine-tune the embedding model (e.g., using SetFit) specifically on the brand's dataset to better separate overlapping support topics.

## 2. Missed High-Risk Escalation

**Customer Message:** "I paid for fast shipping but the delivery is late and I want my $15 back."
**Predicted Intent:** `order_delivery`
**Expected Action:** `ESCALATE` (because it involves a partial refund request).
**Predicted Action:** `AUTO-HANDLE`

**Hypothesis:** The classifier heavily weighted "fast shipping" and "delivery is late", categorizing it as `order_delivery` rather than `payment_billing` or `refund_return`. Since `order_delivery` is not in the `HIGH_RISK_INTENTS`, it was auto-handled.
**Improvement:** Implement multi-label classification so messages can have both `order_delivery` and `refund_return` intents. Any high-risk label would trigger an escalation.

## 3. Poor Retrieval Quality for Niche Issues

**Customer Message:** "The blue light on my Echo Dot 4th Gen is spinning but it won't connect to my custom mesh router."
**Best Retrieved Example:** "My Echo dot won't connect to wifi." (Similarity: 0.62)
**Predicted Action:** `AUTO-HANDLE`

**Hypothesis:** The historical database might lack examples of niche technical setups (e.g., custom mesh routers). The retriever finds a generic Wi-Fi issue which might not have the precise troubleshooting steps.
**Improvement:** Increase the corpus size of the FAISS index to include a longer tail of technical issues. If similarity is only moderate, the system should perhaps instruct the LLM to ask clarifying questions rather than providing a generic fix.

## 4. LLM Hallucination of Policies

**Customer Message:** "What is your return policy for opened electronics?"
**Retrieved Example:** "You can return most unopened items within 30 days."
**LLM Response:** "You can return opened electronics within 14 days of receipt." (Fabricated policy)

**Hypothesis:** The LLM didn't find the exact policy for *opened* electronics in the retrieved context, so it relied on its pre-trained knowledge or hallucinated a plausible-sounding policy.
**Improvement:** Strengthen the system prompt to explicitly state: "If the specific policy is not mentioned in the context, reply 'I need to check that specific policy for you' and Escalate."

## 5. Over-sensitivity to Escalation Keywords

**Customer Message:** "I desperately need this dress for a wedding tomorrow, is it arriving today?"
**Expected Action:** `AUTO-HANDLE` (It's a standard tracking query).
**Predicted Action:** `ESCALATE` (Rule: `sensitive_keywords` triggered by "emergency" or urgency synonyms).

**Hypothesis:** The simple keyword-matching for urgency (e.g., "urgent", "desperate", "emergency") lacks context and triggers false positives for enthusiastic or anxious but low-risk queries.
**Improvement:** Use a secondary, lightweight classifier or a zero-shot LLM pass specifically to classify the severity of the issue, rather than relying on blunt keyword lists.
