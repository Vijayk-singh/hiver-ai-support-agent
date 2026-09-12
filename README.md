# AI Customer Support Agent for @AmazonHelp

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An end-to-end AI Support Agent built for **Amazon Customer Support on Twitter (`@AmazonHelp`)** from the Kaggle Customer Support dataset.

---

## 🚀 Quickstart: Reproduce Results in Under 5 Minutes

The entire pipeline is deterministic, self-contained, and runnable without paid API keys.

```bash
# 1. Activate virtual environment
source venv/bin/activate

# 2. Run full evaluation harness on the Golden Evaluation Set (N = 220)
python3 src/evaluate.py

# 3. Test interactive customer message processing
python3 src/agent.py --text "My package says delivered but I never received it! Where is it?"
```

---

## 📋 Core Capabilities (The Three Tasks)

1. **Intent Classification:** Classifies incoming customer tweets into primary and secondary intents across 18 predefined domain labels:
   - `order_not_delivered`, `order_delayed`, `order_status_inquiry`, `refund_request`, `return_request`, `wrong_item_received`, `damaged_item`, `cancellation_request`, `payment_issue`, `account_issue`, `customer_service_complaint`, `product_inquiry`, `other`, `Prime Membership`, `thankyou`, `Enquiry`, `unknown`, `Urgent`.
2. **Grounded Reply Drafting:** Retrieves top-k historical brand resolutions and drafts a response strictly grounded in official Amazon resolution policies, comparing against a **Trivial Canned Baseline** and a **1-NN Retrieval Baseline**.
3. **Escalation Decision Engine:** Evaluates safety, urgency, and customer sentiment to decide whether a query should be **`AUTO_HANDLE`** or **`ESCALATE`** to human agents, providing a **stated reason**, confidence score, risk level, and target operational queue.

---

## 📊 Headline Evaluation Results

Evaluated on the **Golden Evaluation Set ($N = 220$ curated human-annotated cases)**:

### 1. Intent Classification
- **Accuracy:** **92.3%**
- **Weighted Precision:** **97.0%**
- **Weighted F1-Score:** **94.1%**

### 2. Escalation Triage vs. Baselines
| Approach | Escalation Rate | Accuracy | Recall (Catches) | Dangerous Auto-Handle Rate | False Escalation Rate |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Baseline 1 (Always Auto-Handle)** | 0.0% | 72.7% | 0.0% | 100.0% (Fatal) | 0.0% |
| **Baseline 2 (Always Escalate)** | 100.0% | 27.3% | 100.0% | 0.0% | 100.0% (Overload) |
| **Proposed Grounded Agent** | **55.9%** | **67.7%** | **93.3%** | **6.7%** | **41.9%** |

### 3. Reply Quality across Approaches
| Approach | Policy Grounding (1-5) | Actionability (1-5) | Empathy (1-5) | PII Safety (1-5) | Overall Rubric |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Baseline 1: Trivial Canned** | 3.25 | 3.41 | 3.63 | 4.58 | 3.72 / 5.0 |
| **Baseline 2: 1-NN Retrieval** | 3.46 | 3.42 | 3.63 | 4.93 | 3.86 / 5.0 |
| **Proposed Grounded Agent** | **4.38** | **4.43** | **4.13** | **4.76** | **4.43 / 5.0** |

---

## 🛠️ Project Structure

```
hiver-ai-support-agent/
├── data/
│   ├── processed/
│   │   └── amazonhelp_support_pairs.csv    # 15,000 clean conversation pairs
│   ├── golden_evaluation_set.csv           # 220 curated gold test cases
│   └── evaluation_results.csv              # Full model inference predictions
├── src/
│   ├── extract_brand_pairs.py              # Raw dataset extractor & cleaner
│   ├── classify_intents.py                 # Hierarchical intent classifier
│   ├── knowledge_base.py                   # TF-IDF historical case retriever
│   ├── reply_generator.py                  # Grounded reply generator & baselines
│   ├── escalation_engine.py                # Auto-Handle vs Escalate triage engine
│   ├── agent.py                            # Unified end-to-end agent interface
│   ├── draft_replies.py                    # Interactive/batch reply drafting CLI
│   ├── build_golden_set.py                 # Stratified golden test set generator
│   └── evaluate.py                         # Evaluation harness & rubric
├── REPORT.md                               # Complete 6-page technical report
├── REQUIREMENTS.md                         # Assignment specification
├── requirements.txt                        # Python dependencies
└── README.md                               # Project documentation & guide
```

---

## 💻 CLI Usage Guide

### 1. Unified Agent Triage & Reply
Process any raw customer inquiry:
```bash
python3 src/agent.py --text "My order was supposed to arrive yesterday and your courier is refusing to deliver it!"
```

### 2. Intent Classification
Classify pairs into primary and secondary intents:
```bash
python3 src/classify_intents.py --input-csv data/processed/amazonhelp_support_pairs.csv --output-csv data/processed/amazonhelp_support_pairs.csv
```

### 3. Grounded Reply Drafting with Baselines
Compare Trivial Canned, 1-NN Retrieval, and Grounded Agent:
```bash
python3 src/draft_replies.py --text "I received the wrong item, ordered a shirt and got trousers. How do I exchange?"
```

### 4. Run Evaluation Suite
```bash
python3 src/evaluate.py --golden-set data/golden_evaluation_set.csv
```

---

## 📄 Technical Report

For full problem framing, failure mode analysis, headline number critiques, and decision logs, read [`REPORT.md`](REPORT.md).
