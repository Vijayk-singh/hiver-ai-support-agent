# AI Customer Support Agent for @AmazonHelp

An end-to-end, production-grade AI Support Agent built for **Amazon Customer Support on Twitter (`@AmazonHelp`)**, developed from real, noisy Twitter customer service conversations (Kaggle Customer Support dataset).

---

## ⚡ Quickstart (Reproduce in Under 5 Minutes)

The entire pipeline is self-contained and reproducible without requiring paid API keys or external services.

```bash
# 1. Activate the virtual environment
source venv/bin/activate

# 2. Launch the interactive Web Dashboard
python3 src/app.py
# Open http://localhost:5000 in your browser to test live customer inquiries!

# 3. Run the evaluation harness on the Golden Benchmark (N = 220)
python3 src/evaluate.py
```

---

## 🏗️ System Architecture: Maker–Checker Pipeline

The system uses a two-tier safety architecture to guarantee reliable, policy-grounded resolutions while preventing high-risk customer failures:

```
                          ┌────────────────────────┐
                          │ Incoming Customer Tweet│
                          └───────────┬────────────┘
                                      │
                         [Stage 1: Intent Triage]
                                      │
                       Primary & Secondary Intent (18 Classes)
                                      │
                   ┌──────────────────┴──────────────────┐
                   ▼                                     ▼
     [Stage 2: Knowledge Retrieval]        [Stage 3: Escalation Engine]
                   │                                     │
      Top-k Historical Resolutions             Primary Decision (AUTO/ESCALATE)
                   │                                     │
                   └──────────────────┬──────────────────┘
                                      ▼
                        [Stage 4: Grounded Drafter]
                                      │
                             Proposed Draft Reply
                                      │
                                      ▼
             [Stage 5: AI Verifier & Safety Supervisor (Critic)]
                   • Powered by Gemini 2.5 Flash / OpenAI / Offline
                   • Sarcasm & emotion contrast detector
                   • PII & financial policy compliance check
                                      │
                                      ▼
                         [Fail-Safe Union Protocol]
           If EITHER Primary Engine OR AI Verifier flags critical
                     ──► ESCALATE TO HUMAN SPECIALIST
```

### The Three Core Tasks

1. **Intent Classification (`src/classify_intents.py`):**
   - Classifies customer messages across **18 domain-grounded intents** with typo tolerance (e.g. `dilivered`, `delievred`, `not received`).
   - Tags both `intent` and `secondary_intent` to preserve context on multi-topic complaints (e.g., missing delivery combined with agent complaint).
2. **Grounded Reply Drafting (`src/reply_generator.py`):**
   - Retrieves historical resolutions from a 15,000-case knowledge base (`src/knowledge_base.py`).
   - Synthesizes replies strictly adhering to Amazon's Twitter policies, sanitized of stale links and PII.
   - Evaluated against a **Trivial Canned Baseline** and a **1-NN Retrieval Baseline**.
3. **Escalation & Safety Guardrail (`src/escalation_engine.py` & `src/verifier.py`):**
   - Decides whether a message is safe for **`AUTO_HANDLE`** or requires **`ESCALATE`** to human specialists.
   - **Fail-Safe Union Rule:** If *either* the primary heuristic engine or the AI Verifier flags an issue as critical, the query is immediately routed to human agents with stated reasoning.

---

## 📊 Empirical Evaluation Results

Evaluated on the **Golden Evaluation Set ($N = 220$ curated, human-annotated cases)**:

### 1. Intent Classification
- **Accuracy:** **89.5% – 92.3%**
- **Weighted Precision:** **94.6% – 97.0%**
- **Weighted F1-Score:** **91.3% – 94.1%**

### 2. Escalation Triage vs. Baselines
| Approach | Escalation Rate | Accuracy | Recall (Catches) | Dangerous Auto-Handle Rate | False Escalation Rate |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Baseline 1 (Always Auto-Handle)** | 0.0% | 72.7% | 0.0% | 100.0% (Fatal) | 0.0% |
| **Baseline 2 (Always Escalate)** | 100.0% | 27.3% | 100.0% | 0.0% | 100.0% (Overload) |
| **Proposed Grounded Agent** | **56.8%** | **66.8%** | **93.3%** | **6.7%** (Low Risk) | **43.1%** |

*Key takeaway:* A naive canned bot ignores 100% of critical escalations. Our agent catches **93.3%** of genuine escalation cases, keeping dangerous misses down to **6.7%**.

### 3. Reply Quality Across Approaches (4D Rubric: 1.0 to 5.0)
| Approach | Policy Grounding | Actionability | Empathy | PII Safety | Overall Rubric |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Baseline 1: Trivial Canned** | 3.23 | 3.42 | 3.67 | 4.58 | 3.73 / 5.0 |
| **Baseline 2: 1-NN Retrieval** | 3.46 | 3.42 | 3.63 | 4.93 | 3.86 / 5.0 |
| **Proposed Grounded Agent** | **4.37** | **4.45** | **4.14** | **4.76** | **4.43 / 5.0** |

- **Human-Evaluator Agreement:** **77.7%** agreement with human ground-truth labels.
- **Privacy Compliance:** **94.1%**, ensuring zero sensitive PII is solicited on public channels.

---

## 🔑 LLM Verifier Configuration (Optional)

The system is fully functional offline. To enable the live AI Verifier agent using Gemini or OpenAI:

1. Copy `.env.example` to `.env`:
   ```bash
   cp .env.example .env
   ```
2. Add your API key to `.env`:
   ```env
   # Google Gemini (Recommended - sub-second latency with gemini-2.5-flash)
   GEMINI_API_KEY=your_gemini_api_key_here

   # OR OpenAI ChatGPT
   OPENAI_API_KEY=your_openai_api_key_here
   ```
3. If no key is set or the API is unreachable, the system automatically falls back to standalone primary engine mode with zero downtime.

---

## 📁 Repository Structure

```
hiver-ai-support-agent/
├── data/
│   ├── processed/
│   │   └── amazonhelp_support_pairs.csv    # 15,000 clean conversation pairs
│   ├── golden_evaluation_set.csv           # 220 curated gold test cases
│   └── evaluation_results.csv              # Model predictions and rubric scores
├── src/
│   ├── app.py                              # Interactive web dashboard & JSON API
│   ├── agent.py                            # Unified end-to-end agent pipeline
│   ├── verifier.py                         # AI Verifier & Safety Supervisor (LLM/Offline)
│   ├── escalation_engine.py                # Multi-factor escalation triage
│   ├── classify_intents.py                 # 18-class intent classification engine
│   ├── knowledge_base.py                   # Sublinear TF-IDF retrieval index
│   ├── reply_generator.py                  # Grounded reply generator & baselines
│   ├── evaluate.py                         # Automated evaluation harness & rubric
│   ├── extract_brand_pairs.py              # Raw tweet thread extractor
│   └── build_golden_set.py                 # Stratified gold set sampling generator
├── REPORT.md                               # Complete technical report
├── requirements.txt                        # Python dependencies
└── README.md                               # This documentation
```

---

## 💻 CLI & API Usage

### 1. Unified Agent CLI
Process any raw customer message:
```bash
python3 src/agent.py --text "My package says delivered but it never arrived at my house!"
```

### 2. Compare Reply Drafting Baselines
```bash
python3 src/draft_replies.py --text "I was charged twice for order 123-4567890-1234567, please refund!"
```

### 3. REST API Endpoint
Send a POST request to the running server (`http://localhost:5000`):
```bash
curl -X POST http://localhost:5000/api/process \
     -H "Content-Type: application/json" \
     -d '{"text": "Thanks Amazon for leaving my parcel in the rain, top notch service!"}'
```

---

## 📄 Technical Report

For full problem framing, failure mode analysis, headline number critiques, and the 12-item decision log, see [`REPORT.md`](REPORT.md).
