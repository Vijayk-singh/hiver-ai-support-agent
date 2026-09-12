# AI Support Agent for @AmazonHelp: Technical Report

**Hiver SDE Intern Take-Home Assignment**  
**Candidate:** Vijay  
**Target Brand:** Amazon Support on Twitter (`@AmazonHelp`)  
**Evaluation Set:** 220 Curated Golden Evaluation Cases  
**Corpus Size:** 15,000 Clean English Conversation Pairs (`amazonhelp_support_pairs.csv`)

---

## 1. Problem Framing

### 1.1 What "Good" Means for Amazon Support on Twitter
Customer support on Twitter is fundamentally different from email or live chat:
1. **Public Brand Surface:** Every tweet is visible to millions. Sarcasm, rude handling, or false promises cause immediate brand and PR damage.
2. **Strict Character Constraints (280 chars):** Replies must be concise, empathetic, and actionable without fluff.
3. **Security & Privacy Boundary:** Twitter is an insecure public channel. Amazon policy strictly forbids requesting or exposing PII (passwords, complete credit card numbers, OTPs, or order numbers) publicly.
4. **First-Contact Resolution (FCR) vs. Safe Routing:** "Good" means auto-handling routine informational and self-service queries with verified links, while immediately escalating high-risk grievances (missing packages, fraudulent charges, account locks, hostile churn threats) to human specialists with stated reasoning.

### 1.2 What We Explicitly Chose NOT to Build
- **No Direct Financial Actions via Bot:** The agent does not execute refunds, cancel credit card charges, or grant gift cards directly over Twitter. These actions require authenticated customer sessions to prevent fraud.
- **No Ungrounded Generative Chit-Chat:** The agent is constrained from open-domain conversational hallucination. Every response is anchored to Amazon's verified historical resolution policies.
- **No Automated Password Resets over Twitter:** Any account recovery request is escalated to secure official channels (`amazon.com/help/account`).

---

## 2. System Architecture: Maker–Checker Pipeline

The agent operates across a multi-stage Maker–Checker architecture:

```
                            ┌────────────────────────┐
                            │ Incoming Customer Tweet│
                            └───────────┬────────────┘
                                        │
                           [Stage 1: Intent Triage]
                                        │
                         Primary & Secondary Intent (18 Classes)
                                        │
                         ┌──────────────┴──────────────┐
                         ▼                             ▼
        [Stage 2: Knowledge Retrieval]   [Stage 3: Escalation Engine]
                         │                             │
          Top-k Historical Resolutions     Primary Triage Decision
                         │                             │
                         └──────────────┬──────────────┘
                                        ▼
                           [Stage 4: Grounded Drafter]
                                        │
                               Proposed Draft Reply
                                        │
                                        ▼
                [Stage 5: AI Verifier & Safety Guardrail (Checker)]
                    • Powered by Gemini 2.5 Flash / OpenAI / Offline Critic
                    • Sarcasm & emotion contrast detection
                    • PII & financial policy compliance audit
                                        │
                                        ▼
                           [Fail-Safe Union Protocol]
             If EITHER Primary Engine OR AI Verifier flags critical
                       ──► ESCALATE TO HUMAN SPECIALIST
```

1. **Intent Classification (`src/classify_intents.py`):**
   - Hierarchical rule-and-pattern engine covering 18 predefined intent labels tailored to Amazon support data with spelling typo tolerance (e.g. `dilivered`, `delievred`, `not received`).
   - Assigns `intent` and `secondary_intent` to capture multi-faceted customer complaints (e.g. `order_not_delivered` + `customer_service_complaint`).
2. **Historical Knowledge Base (`src/knowledge_base.py`):**
   - Sublinear TF-IDF index over 15,000 historical support pairs.
   - Intent-boosted cosine similarity retrieval to retrieve top historical resolutions.
3. **Grounded Reply Generator (`src/reply_generator.py`):**
   - Evaluates three generation strategies: Trivial Canned Baseline, Simple 1-NN Retrieval Baseline, and Contextual Grounded Agent.
   - Dual execution engine: Deterministic grounded synthesis (100% reproducible offline) + LLM grounded engine when API keys are supplied.
4. **Escalation Engine (`src/escalation_engine.py`):**
   - Evaluates risk, priority, confidence, and states explicit reasoning for `AUTO_HANDLE` vs `ESCALATE`.
5. **AI Verifier & Safety Guardrail (`src/verifier.py`):**
   - Independent supervisor auditing proposed actions using **Gemini 2.5 Flash** or **OpenAI gpt-4o-mini**, with graceful offline fallback.
   - **Fail-Safe Union Protocol:** If *either* the Primary Engine *or* the AI Verifier flags an issue as critical, the final decision is guaranteed to `ESCALATE` to human specialists. If the verifier is offline, the system safely falls back solely to the Primary Engine.

---

## 3. Results vs. Baselines

All metrics are evaluated on the **Golden Evaluation Set ($N = 220$)**.

### 3.1 Intent Classification Performance

| Metric | Score |
| :--- | :---: |
| **Accuracy** | **92.3%** |
| **Weighted Precision** | **97.0%** |
| **Weighted Recall** | **92.3%** |
| **Weighted F1-Score** | **94.1%** |
| **Macro F1-Score** | **87.9%** |

---

### 3.2 Escalation Decision vs. Baselines

The primary safety metric is **Dangerous Auto-Handle Rate** (proportion of high-risk cases erroneously auto-handled). The primary operational efficiency metric is **False Escalation Rate** (proportion of routine cases unnecessarily burdening human agents).

| Approach | Escalation Rate | Accuracy | Recall (Catching Escalations) | Dangerous Auto-Handle Rate (Misses) | False Escalation Rate (Agent Overhead) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Baseline 1: Always Auto-Handle** | 0.0% | 72.7% | 0.0% | 100.0% (Catastrophic) | **0.0%** |
| **Baseline 2: Always Escalate** | 100.0% | 27.3% | **100.0%** | **0.0%** | 100.0% (Overwhelming) |
| **Proposed Multi-Factor Agent** | **55.9%** | **67.7%** | **93.3%** | **6.7%** | **41.9%** |

- **Baseline 1** fails completely on safety: it ignores fraud, stolen packages, and hostile threats.
- **Baseline 2** collapses operations: every single customer is handed off to human staff.
- **Proposed Agent** catches **93.3% of critical escalations**, keeping dangerous misses down to 6.7% while reserving human attention for genuinely complex issues.

---

### 3.3 Reply Quality Across Approaches

Evaluated across word overlap Token F1 vs historical brand replies and a 4-dimensional quality rubric (1.0 to 5.0 scale):

| Approach | Token F1 vs Brand | Empathy (1-5) | Policy Grounding (1-5) | PII Safety (1-5) | Actionability (1-5) | Overall Rubric (1-5) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline 1: Trivial Canned** | 0.140 | 3.63 | 3.25 | 4.58 | 3.41 | 3.72 |
| **Baseline 2: 1-NN Retrieval** | 1.000* | 3.63 | 3.46 | 4.93 | 3.42 | 3.86 |
| **Proposed Grounded Agent** | **0.188** | **4.13** | **4.38** | **4.76** | **4.43** | **4.43** |

*(Note on 1-NN Token F1 of 1.000 is analyzed in Section 6).*

**Key Takeaways:**
- The Proposed Grounded Agent outperforms both baselines on **Policy Grounding (4.38)** and **Actionability (4.43)** by providing specific resolution URLs (`/returns`, `/your-orders`, `/contact-us`) and immediate guidance.
- Evaluator agreement with human benchmark stands at **76.8%**, with **95.0% policy grounding alignment** and **94.1% privacy safety compliance**.

---

## 4. Golden Evaluation Set Construction

The Golden Evaluation Set consists of **220 curated examples** sampled from historical Amazon customer support tweets:
- **Sampling Strategy:** Stratified sampling across 16 active intent categories (12–18 examples per class) to avoid dominant class starvation.
- **Human Annotation:** Each sample was manually validated with:
  - Ground-truth Primary Intent
  - Ground-truth Secondary Intent
  - Ground-truth Escalation Decision (`AUTO_HANDLE` vs. `ESCALATE`)
  - Stated Escalation Rationale
- **Class Distribution:** 160 Auto-Handled cases (72.7%) and 60 Escalated cases (27.3%), reflecting realistic production distribution.

---

## 5. Failure Analysis: Top 5 Failure Modes

| # | Failure Mode | Real Example Tweet | Root Cause Hypothesis | Remediation |
| :-: | :--- | :--- | :--- | :--- |
| **1** | **Sarcastic Praise Masking Complaint** | *"Oh brilliant, thank you @Amazon for leaving my laptop in the pouring rain outside my garage! Truly top notch service!"* | Lexical matcher captures `thank you` and `top notch` and assigns `thankyou` instead of `damaged_item` / `customer_service_complaint`. | Add sentiment polarity contrast detector (positive words + negative situation markers like `rain`, `dumped`). |
| **2** | **Multi-Turn Thread Context Loss** | *"Submitted, for the third time for the same issue."* | Single-tweet evaluation loses the original order complaint from earlier in the thread. Falls back to `other`. | Ingest parent tweet chain and maintain rolling conversation state. |
| **3** | **False Escalation on Rhetorical Questions** | *"Why is customer service so hard to reach nowadays?"* | Rule engine flags `customer service` and marks as complaint escalation, though customer was asking a general question. | Calibrate complaint threshold requiring specific agent action failure before escalating. |
| **4** | **Ambiguous Delayed vs. Missing Order** | *"Expected delivery was 2 days ago, still no sign of my parcel."* | Close overlap between `order_delayed` and `order_not_delivered`. | Use carrier ETA status: if within 48h of estimate, classify as `order_delayed`; if >7 days or marked delivered, classify as `order_not_delivered`. |
| **5** | **Shortened URL Hallucination Risk in 1-NN** | *"Please check here: https://t.co/xyz123"* | 1-NN baseline copies historical `t.co` links that have expired or belong to a different customer's specific session. | Replace dynamic Kaggle `t.co` links with canonical Amazon landing URLs (`amazon.com/returns`, `amazon.com/your-orders`). |

---

## 6. "What is Misleading About My Headline Number?"

1. **The 1.000 Token F1 on Baseline 2 is Artificial:**
   Because the Golden Evaluation Set was sampled from the cleaned historical dataset, 1-NN retrieval finds the exact same training pair, achieving a trivial 1.000 Token F1. In real production with novel incoming queries, 1-NN drops significantly and often recommends irrelevant tracking numbers or expired `t.co` links.
2. **Intent Accuracy (92.3%) Masks Ambiguity in the "Long Tail":**
   92.3% accuracy looks outstanding on common delivery and return intents. However, for nuanced compound intents (e.g. seller professional plan billing + account suspension), accuracy degrades to ~75%.
3. **Drafting a Link Does Not Mean "Problem Solved":**
   Our Actionability score (4.43/5.0) measures whether the bot provided the correct link. In reality, a customer whose package was stolen will not be satisfied merely by receiving a link to `/contact-us`; true resolution requires backend carrier claim approval.
4. **Offline Evaluation Lacks Real Customer Feedback Loops:**
   Automated rubric scores and LLM-as-judge ratings do not capture whether the customer felt understood or churned after reading the tweet.

---

## 7. What We Would Do Next with One More Week

1. **Dense Vector Retrieval (RAG with `bge-small-en-v1.5` & ChromaDB):**
   Replace TF-IDF lexical matching with dense semantic embeddings to better capture semantic paraphrase and slang.
2. **Multi-Turn Thread Reconstruction:**
   Extract and link the entire conversation tree (`in_response_to_tweet_id`) so the agent knows what previous agents promised.
3. **Sentiment Velocity Tracking:**
   Track the delta in sentiment across turns; if sentiment turns from neutral to angry, escalate immediately to a supervisor.
4. **Direct API Integration with Mock Amazon Orders API:**
   Enable the bot to fetch order status using authenticated order tokens in private DM workflows.

---

## 8. Decision Log (14 Non-Obvious Decisions)

1. **Chose AmazonHelp over Other Brands:** Amazon has the largest, highest-standard e-commerce customer support dataset on Twitter with well-defined self-service portals.
2. **Enforced 18 Predefined Intent Labels:** Selected a comprehensive 18-label taxonomy tailored to retail customer support instead of a generic 5-class set.
3. **Created `intent` and `secondary_intent`:** Real customer complaints are rarely single-intent (e.g. `damaged_item` + `refund_request`); multi-label tagging prevents nuance loss.
4. **Used Brand Text as Intent Disambiguation Signal:** When customer text is vague (*"Still nothing"*), the historical brand reply (*"Sorry your package has not arrived"*) confirms intent.
5. **Replaced Raw `t.co` Links with Canonical Amazon URLs:** Raw Twitter links in Kaggle data expire; canonical URLs (`amazon.com/returns`) ensure practical usability.
6. **Prioritized Low Dangerous Auto-Handle Rate over High Accuracy in Escalation:** Erroneously auto-handling fraud or lost orders causes severe customer loss; over-escalating is safer.
7. **Designed a Dual Execution Engine (Offline + LLM):** Guaranteed the pipeline can be run and reproduced in <15 minutes without requiring paid external API credentials.
8. **Enforced PII Privacy Guardrails in Prompts & Heuristics:** Never ask for credit card numbers or passwords on public Twitter.
9. **Included `^AI` Representative Signature:** Emulated Twitter customer service conventions where reps sign off with initials.
10. **Built Stratified Golden Evaluation Set ($N = 220$):** Selected balanced samples across all 16 active categories instead of random sampling which would be 40% delivery queries.
11. **Reported Dangerous Auto-Handle Rate as Primary Escalation Metric:** Standard accuracy is misleading when 73% of data is auto-handled.
12. **Added Alias `secondery_intent`:** Defensive engineering to prevent key errors against prompt typos.
13. **Independent AI Verifier & Safety Supervisor (Maker–Checker Pattern):** Deployed a separate LLM auditor (Gemini 2.5 Flash / OpenAI) to critique proposed actions, catch sarcasm, and prevent policy violations before final execution.
14. **Fail-Safe Union Escalation Protocol:** Implemented a union escalation policy: if *either* the Primary Engine *or* the AI Verifier flags critical risk, the inquiry is immediately routed to human specialists. If the verifier is offline, the system safely falls back solely to the Primary Engine without halting.
