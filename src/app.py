"""Web Application for Amazon AI Support Agent.

Provides an interactive user interface to:
1. Enter customer support inquiries.
2. Define the problem (classify primary & secondary intents).
3. Decide whether the message is AUTO-HANDLED by system or ESCALATED to human (with stated reason).
4. Draft grounded reply based on historical brand resolutions, comparing against baselines.
"""

import os
import sys
from pathlib import Path
from flask import Flask, render_template_string, request, jsonify

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.agent import AmazonSupportAgent

app = Flask(__name__)

# Initialize agent once on startup
print("Initializing Amazon AI Support Agent for Web UI...")
agent = AmazonSupportAgent()
print("Agent initialized and ready to serve requests.")

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Amazon AI Support Agent — Live Dashboard</title>
  <script src="https://cdn.tailwindcss.com"></script>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
  <style>
    body { font-family: 'Plus Jakarta Sans', sans-serif; }
    code, pre { font-family: 'JetBrains Mono', monospace; }
  </style>
</head>
<body class="bg-slate-950 text-slate-100 min-h-screen antialiased flex flex-col">

  <!-- Header -->
  <header class="border-b border-slate-800 bg-slate-900/80 backdrop-blur sticky top-0 z-50">
    <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between">
      <div class="flex items-center space-x-3">
        <div class="w-10 h-10 rounded-xl bg-gradient-to-tr from-amber-500 to-orange-600 flex items-center justify-center font-black text-xl text-slate-950 shadow-lg shadow-orange-500/20">
          a
        </div>
        <div>
          <h1 class="text-base font-bold tracking-tight text-white flex items-center gap-2">
            Amazon Customer Support AI Agent
            <span class="text-xs font-semibold px-2 py-0.5 rounded-full bg-amber-500/10 text-amber-400 border border-amber-500/20">@AmazonHelp</span>
          </h1>
          <p class="text-xs text-slate-400">Hiver SDE Intern Take-Home System</p>
        </div>
      </div>
      <div class="flex items-center space-x-3 text-xs">
        <span class="flex items-center gap-1.5 px-3 py-1 rounded-full bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">
          <span class="w-2 h-2 rounded-full bg-emerald-400 animate-pulse"></span>
          Knowledge Base Active (15,000 pairs)
        </span>
      </div>
    </div>
  </header>

  <!-- Main Content -->
  <main class="flex-1 max-w-7xl w-full mx-auto px-4 sm:px-6 lg:px-8 py-8">
    <div class="grid grid-cols-1 lg:grid-cols-12 gap-8">

      <!-- Left Column: Input & Presets -->
      <div class="lg:col-span-5 space-y-6">
        <div class="bg-slate-900 border border-slate-800 rounded-2xl p-6 shadow-xl shadow-black/40">
          <div class="flex items-center justify-between mb-4">
            <h2 class="text-sm font-bold uppercase tracking-wider text-slate-400">Customer Tweet Inquiry</h2>
            <span class="text-xs text-slate-500">Live Inference</span>
          </div>

          <form id="inquiryForm" class="space-y-4">
            <div class="relative">
              <textarea
                id="customerText"
                rows="5"
                placeholder="Enter customer tweet inquiry here... (e.g. 'My package says delivered but I never received it! Where is my stuff?')"
                class="w-full bg-slate-950 border border-slate-800 rounded-xl p-4 text-sm text-slate-100 placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-amber-500 focus:border-transparent resize-none leading-relaxed"
                required
              ></textarea>
            </div>

            <button
              type="submit"
              id="submitBtn"
              class="w-full bg-gradient-to-r from-amber-500 to-orange-500 hover:from-amber-400 hover:to-orange-400 text-slate-950 font-bold py-3 px-6 rounded-xl shadow-lg shadow-orange-500/25 transition duration-200 flex items-center justify-center space-x-2"
            >
              <span id="btnText">Process Inquiry</span>
              <svg id="btnSpinner" class="hidden animate-spin h-5 w-5 text-slate-950" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle>
                <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8H4z"></path>
              </svg>
            </button>
          </form>

          <!-- Quick Sample Queries -->
          <div class="mt-6 pt-6 border-t border-slate-800">
            <p class="text-xs font-semibold text-slate-400 mb-3">Try Realistic Test Scenarios:</p>
            <div class="flex flex-wrap gap-2">
              <button class="preset-btn text-xs bg-slate-800 hover:bg-slate-700 text-slate-300 py-1.5 px-3 rounded-lg border border-slate-700/60 transition" data-text="3 different people have given 3 different answers and I still don't have my order. Says delivered Saturday, was not, I was home all day">
                📦 Missing / Stolen Package
              </button>
              <button class="preset-btn text-xs bg-slate-800 hover:bg-slate-700 text-slate-300 py-1.5 px-3 rounded-lg border border-slate-700/60 transition" data-text="Your representative was extremely rude on phone and hung up on me when I asked for a supervisor!">
                😤 Rude Agent Complaint
              </button>
              <button class="preset-btn text-xs bg-slate-800 hover:bg-slate-700 text-slate-300 py-1.5 px-3 rounded-lg border border-slate-700/60 transition" data-text="URGENT: I need my medicine order before 4pm because I am traveling tonight!">
                🚨 Time-Critical Emergency
              </button>
              <button class="preset-btn text-xs bg-slate-800 hover:bg-slate-700 text-slate-300 py-1.5 px-3 rounded-lg border border-slate-700/60 transition" data-text="I ordered red sneakers size 10, got women boots instead. How do I exchange this?">
                🔄 Wrong Item Received
              </button>
              <button class="preset-btn text-xs bg-slate-800 hover:bg-slate-700 text-slate-300 py-1.5 px-3 rounded-lg border border-slate-700/60 transition" data-text="You charged my credit card twice for order [ORDER_ID]. Need this refunded immediately!">
                💳 Double Payment Deduction
              </button>
              <button class="preset-btn text-xs bg-slate-800 hover:bg-slate-700 text-slate-300 py-1.5 px-3 rounded-lg border border-slate-700/60 transition" data-text="Where is my package? The tracking has not updated since yesterday morning.">
                📍 Routine Status Inquiry
              </button>
              <button class="preset-btn text-xs bg-slate-800 hover:bg-slate-700 text-slate-300 py-1.5 px-3 rounded-lg border border-slate-700/60 transition" data-text="Thank you so much Amazon, you guys resolved my refund in 5 minutes! Best support ever.">
                🙏 Praise & Gratitude
              </button>
            </div>
          </div>
        </div>

        <!-- Metric Card Banner -->
        <div class="bg-slate-900/60 border border-slate-800/80 rounded-2xl p-5 text-xs text-slate-400 space-y-2">
          <div class="flex items-center justify-between text-slate-300 font-semibold">
            <span>Golden Test Performance ($N=220$)</span>
            <a href="/report" class="text-amber-400 hover:underline">View Full Report</a>
          </div>
          <div class="grid grid-cols-3 gap-2 pt-2 text-center">
            <div class="bg-slate-950 p-2.5 rounded-lg border border-slate-800">
              <div class="text-slate-500">Intent Acc.</div>
              <div class="text-emerald-400 font-bold text-sm">92.3%</div>
            </div>
            <div class="bg-slate-950 p-2.5 rounded-lg border border-slate-800">
              <div class="text-slate-500">Dangerous Misses</div>
              <div class="text-emerald-400 font-bold text-sm">6.7%</div>
            </div>
            <div class="bg-slate-950 p-2.5 rounded-lg border border-slate-800">
              <div class="text-slate-500">Rubric Score</div>
              <div class="text-amber-400 font-bold text-sm">4.43 / 5.0</div>
            </div>
          </div>
        </div>
      </div>

      <!-- Right Column: Live Analysis Output -->
      <div class="lg:col-span-7 space-y-6">

        <!-- Welcome Placeholder -->
        <div id="placeholderView" class="bg-slate-900 border border-slate-800 border-dashed rounded-2xl p-12 text-center flex flex-col items-center justify-center min-h-[460px]">
          <div class="w-16 h-16 rounded-2xl bg-slate-800 flex items-center justify-center text-3xl mb-4 text-slate-400">
            ⚡
          </div>
          <h3 class="text-base font-bold text-slate-300">Ready to Triage Inquiries</h3>
          <p class="text-sm text-slate-500 max-w-md mt-1">
            Type any customer message or click one of the realistic test scenarios on the left to see intent analysis, human vs. system triage, and grounded reply drafting.
          </p>
        </div>

        <!-- Results Container (Hidden Initially) -->
        <div id="resultsView" class="hidden space-y-6">

          <!-- Card 1: Problem Definition & Escalation Triage -->
          <div class="bg-slate-900 border border-slate-800 rounded-2xl p-6 shadow-xl">
            <div class="flex items-center justify-between pb-4 border-b border-slate-800">
              <div>
                <span class="text-xs uppercase tracking-wider text-slate-400 font-bold">Step 1 & 3: Triage & Routing Decision</span>
                <h3 class="text-lg font-bold text-white mt-0.5">Problem Definition & Escalation</h3>
              </div>
              <div id="decisionBadge"></div>
            </div>

            <!-- Decision Box -->
            <div id="decisionBox" class="mt-4 p-4 rounded-xl border flex flex-col gap-2"></div>

            <!-- Intent Badges -->
            <div class="grid grid-cols-1 sm:grid-cols-3 gap-3 mt-4 pt-4 border-t border-slate-800/80 text-xs">
              <div class="bg-slate-950 p-3 rounded-xl border border-slate-800">
                <span class="text-slate-500 block mb-1">Primary Problem Intent</span>
                <span id="primaryIntentBadge" class="font-semibold text-purple-400"></span>
              </div>
              <div class="bg-slate-950 p-3 rounded-xl border border-slate-800">
                <span class="text-slate-500 block mb-1">Secondary Intent</span>
                <span id="secondaryIntentBadge" class="font-semibold text-amber-400"></span>
              </div>
              <div class="bg-slate-950 p-3 rounded-xl border border-slate-800">
                <span class="text-slate-500 block mb-1">Resolution Channel</span>
                <span id="targetChannelBadge" class="font-semibold text-blue-400"></span>
              </div>
            </div>
          </div>

          <!-- Card 2: Grounded Drafted Reply -->
          <div class="bg-slate-900 border border-slate-800 rounded-2xl p-6 shadow-xl">
            <div class="flex items-center justify-between mb-3">
              <div>
                <span class="text-xs uppercase tracking-wider text-slate-400 font-bold">Step 2: Grounded Generation</span>
                <h3 class="text-lg font-bold text-white mt-0.5">Drafted Support Reply</h3>
              </div>
              <span class="text-xs px-2.5 py-1 rounded-full bg-slate-800 text-slate-400 border border-slate-700">Official Brand Voice</span>
            </div>

            <div class="bg-slate-950 border border-slate-800 rounded-xl p-5 relative">
              <div class="flex items-start space-x-3 mb-3">
                <div class="w-8 h-8 rounded-full bg-amber-500 flex items-center justify-center text-slate-950 font-black text-sm">a</div>
                <div>
                  <div class="text-sm font-bold text-white flex items-center gap-1.5">
                    Amazon Help
                    <svg class="w-4 h-4 text-blue-400 fill-current" viewBox="0 0 24 24"><path d="M9 16.17L4.83 12l-1.42 1.41L9 19 21 7l-1.41-1.41L9 16.17z"/></svg>
                  </div>
                  <div class="text-xs text-slate-500">@AmazonHelp · Official Response</div>
                </div>
              </div>
              <p id="draftedReplyText" class="text-sm text-slate-200 leading-relaxed font-normal"></p>
            </div>
          </div>

          <!-- Card 3: Historical Grounding Evidence -->
          <div class="bg-slate-900 border border-slate-800 rounded-2xl p-6 shadow-xl">
            <div class="flex items-center justify-between mb-4">
              <h3 class="text-sm font-bold uppercase tracking-wider text-slate-400">Historical Grounding Evidence (Retrieved Cases)</h3>
              <span class="text-xs text-slate-500">Corpus Grounding</span>
            </div>
            <div id="historicalCasesContainer" class="space-y-3"></div>
          </div>

          <!-- Card 4: Baseline Comparison -->
          <div class="bg-slate-900 border border-slate-800 rounded-2xl p-6 shadow-xl">
            <h3 class="text-sm font-bold uppercase tracking-wider text-slate-400 mb-4">Baseline Model Comparisons</h3>
            <div class="space-y-3 text-xs">
              <div class="p-3 bg-slate-950 rounded-xl border border-slate-800">
                <span class="font-bold text-slate-400 block mb-1">Baseline 1: Trivial Canned Template</span>
                <p id="cannedBaselineText" class="text-slate-300"></p>
              </div>
              <div class="p-3 bg-slate-950 rounded-xl border border-slate-800">
                <span class="font-bold text-slate-400 block mb-1">Baseline 2: Simple 1-NN Verbatim Retrieval</span>
                <p id="retrievalBaselineText" class="text-slate-300"></p>
              </div>
            </div>
          </div>

        </div>

      </div>

    </div>
  </main>

  <!-- Footer -->
  <footer class="border-t border-slate-800 bg-slate-950 py-6 text-center text-xs text-slate-500 mt-auto">
    Hiver SDE Intern Take-Home Project · Built for Amazon Customer Support on Twitter (@AmazonHelp)
  </footer>

  <script>
    const form = document.getElementById('inquiryForm');
    const customerText = document.getElementById('customerText');
    const submitBtn = document.getElementById('submitBtn');
    const btnText = document.getElementById('btnText');
    const btnSpinner = document.getElementById('btnSpinner');
    const placeholderView = document.getElementById('placeholderView');
    const resultsView = document.getElementById('resultsView');

    // Preset buttons
    document.querySelectorAll('.preset-btn').forEach(btn => {
      btn.addEventListener('click', () => {
        customerText.value = btn.dataset.text;
        form.dispatchEvent(new Event('submit'));
      });
    });

    form.addEventListener('submit', async (e) => {
      e.preventDefault();
      const text = customerText.value.trim();
      if (!text) return;

      // Show loading state
      submitBtn.disabled = true;
      btnText.textContent = "Analyzing...";
      btnSpinner.classList.remove('hidden');

      try {
        const response = await fetch('/api/process', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ text })
        });

        if (!response.ok) throw new Error("Failed to process inquiry");
        const data = await response.json();

        // Render Results
        renderResults(data);

        placeholderView.classList.add('hidden');
        resultsView.classList.remove('hidden');
      } catch (err) {
        alert("Error analyzing inquiry: " + err.message);
      } finally {
        submitBtn.disabled = false;
        btnText.textContent = "Process Inquiry";
        btnSpinner.classList.add('hidden');
      }
    });

    function renderResults(res) {
      // 1. Badges & Intents
      document.getElementById('primaryIntentBadge').textContent = res.intent;
      document.getElementById('secondaryIntentBadge').textContent = res.secondary_intent || 'None';
      document.getElementById('targetChannelBadge').textContent = res.target_channel;

      // 2. Escalation Decision
      const esc = res.escalation;
      const isEscalate = esc.decision === 'ESCALATE';

      const badgeElem = document.getElementById('decisionBadge');
      if (isEscalate) {
        badgeElem.innerHTML = `<span class="px-3 py-1 rounded-full bg-red-500/10 text-red-400 border border-red-500/20 font-bold text-xs flex items-center gap-1.5">
          <span class="w-2 h-2 rounded-full bg-red-400 animate-ping"></span>
          ESCALATE TO HUMAN
        </span>`;
      } else {
        badgeElem.innerHTML = `<span class="px-3 py-1 rounded-full bg-emerald-500/10 text-emerald-400 border border-emerald-500/20 font-bold text-xs flex items-center gap-1.5">
          <span class="w-2 h-2 rounded-full bg-emerald-400"></span>
          SYSTEM AUTO-HANDLE
        </span>`;
      }

      const boxElem = document.getElementById('decisionBox');
      const boxColor = isEscalate ? 'border-red-500/30 bg-red-500/5' : 'border-emerald-500/30 bg-emerald-500/5';
      const headingColor = isEscalate ? 'text-red-400' : 'text-emerald-400';

      boxElem.className = `p-4 rounded-xl border ${boxColor}`;
      boxElem.innerHTML = `
        <div class="flex items-center justify-between">
          <span class="font-bold text-sm ${headingColor}">
            ${isEscalate ? '🚨 Requires Human Agent Escalation' : '🤖 Safe for Autonomous Handling'}
          </span>
          <span class="text-xs text-slate-400">Confidence: <strong>${Math.round(esc.confidence * 100)}%</strong> | Risk: <strong>${esc.risk_level}</strong> | Priority: <strong>${esc.priority}</strong></span>
        </div>
        <p class="text-xs text-slate-300 mt-1 leading-relaxed">
          <strong>Stated Reason:</strong> ${esc.reason}
        </p>
        <div class="text-xs text-slate-400 mt-1">
          <strong>Assigned Team:</strong> <span class="text-amber-400">${esc.suggested_team}</span>
        </div>
      `;

      // 3. Drafted Reply
      document.getElementById('draftedReplyText').textContent = res.drafted_reply;

      // 4. Historical Grounding Cases
      const casesContainer = document.getElementById('historicalCasesContainer');
      casesContainer.innerHTML = '';
      if (res.historical_cases && res.historical_cases.length > 0) {
        res.historical_cases.forEach((c, idx) => {
          const caseEl = document.createElement('div');
          caseEl.className = 'p-3 bg-slate-950 rounded-xl border border-slate-800 text-xs space-y-1.5';
          caseEl.innerHTML = `
            <div class="flex justify-between items-center text-slate-500">
              <span class="font-semibold text-slate-400">Historical Case #${idx + 1} (Intent: ${c.intent})</span>
              <span class="text-cyan-400 font-mono">Similarity: ${c.similarity.toFixed(3)}</span>
            </div>
            <div class="text-slate-300"><strong>Customer:</strong> "${c.customer_text}"</div>
            <div class="text-emerald-400"><strong>Amazon Reply:</strong> "${c.brand_text}"</div>
          `;
          casesContainer.appendChild(caseEl);
        });
      }

      // 5. Baselines
      document.getElementById('cannedBaselineText').textContent = res.trivial_baseline_reply;
      document.getElementById('retrievalBaselineText').textContent = res.retrieval_baseline_reply;
    }
  </script>
</body>
</html>
"""

@app.route("/")
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route("/api/process", methods=["POST"])
def process():
    data = request.get_json() or {}
    text = data.get("text", "").strip()
    if not text:
        return jsonify({"error": "No query text provided"}), 400

    result = agent.process_message(text)
    return jsonify(result.model_dump())

@app.route("/report")
def report():
    # Render REPORT.md as readable HTML
    report_path = PROJECT_ROOT / "REPORT.md"
    if not report_path.exists():
        return "Report not found", 404
    content = report_path.read_text(encoding="utf-8")
    return f"""
    <!DOCTYPE html>
    <html><head><title>Technical Report</title>
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <script src="https://cdn.tailwindcss.com"></script>
    </head>
    <body class="bg-slate-950 text-slate-100 p-8 max-w-4xl mx-auto font-sans leading-relaxed">
    <a href="/" class="text-amber-400 underline mb-4 inline-block">&larr; Back to Live Agent Dashboard</a>
    <pre class="whitespace-pre-wrap font-sans text-sm text-slate-300">{content}</pre>
    </body></html>
    """

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    print(f"\n========================================================")
    print(f"🚀 Amazon AI Support Agent Web UI running on:")
    print(f"   http://127.0.0.1:{port} or http://localhost:{port}")
    print(f"========================================================\n")
    app.run(host="0.0.0.0", port=port, debug=False)
