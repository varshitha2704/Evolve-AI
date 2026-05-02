# 🧬 Evolve Agent — CS5381 Analysis of Algorithms

> A Simplified Evolutionary Agent for Algorithm Discovery, inspired by [AlphaEvolve](https://arxiv.org/abs/2506.13131)

---

## 📌 Project Description

**Evolve Agent** is an evolutionary algorithm assistant that takes an initial code solution or algorithm description and iteratively improves it using a combination of:

- **LLM-guided mutation** (Google Gemini API)
- **Random / AST-based mutation**
- **Template-based candidate generation**
- **Human-in-the-loop refinement**

The system evaluates each candidate using a domain-specific fitness function and selects survivors via top-k selection — repeating across configurable generations to demonstrate measurable improvement.

**Use Cases implemented:**
1. 🎮 **Pacman Agent Optimization** — Maximize score, survival time, and minimize steps
2. 🔢 **3×3 Matrix Multiplication** *(Bonus)* — Minimize scalar operations while preserving correctness

---

## 🏗️ System Architecture

```
+-------------------------------------------------------------+
|                       Evolve Agent                          |
|                                                             |
|  +--------------+    +--------------+    +--------------+  |
|  |  Candidate   |--->|  Evaluator   |--->|  Selector    |  |
|  |  Generator   |    | (Fitness fn) |    |  (Top-k)     |  |
|  +--------------+    +--------------+    +--------------+  |
|         ^                                        |          |
|         |           Evolution Loop               |          |
|         +----------------------------------------+          |
|                                                             |
|  Strategies:                                                |
|    No-Evolution  |  Random (AST-based)  |  LLM-Guided      |
|                                                             |
|  Fitness (Pacman):  w1*score + w2*survival − w3*cost       |
|  Fitness (Matrix):  correctness(60) + efficiency(40)       |
|                                                             |
|  Acceleration:                                              |
|    Cache: hash(code) → fitness  (zero re-evaluation)       |
|    VDB:   FAISS / list  (top-k snippet retrieval)          |
+-------------------------------------------------------------+
```

### Component Responsibilities

| Component | Description |
|-----------|-------------|
| **Candidate Generator** | Produces mutated/refined code via LLM, random AST ops, or templates |
| **Evaluator** | Runs candidate in a sandboxed Pacman/Matrix environment, computes fitness |
| **Selector** | Retains the top-k candidates for the next generation |
| **Cache** | Hash-based memoization — avoids re-evaluating identical code |
| **Vector DB** | FAISS or list-based store of top-fitness snippets for LLM context |
| **UI (Streamlit)** | Visualizes fitness across generations, exports data and charts |

---

## 👥 Team & Contributions

| Member | Role & Contributions |
|--------|---------------------|
| **Siddhartha Bollam** | System architecture · Core evolutionary engine · Fitness computation module · LLM integration & caching · Backend–UI communication · Repo management |
| **Suraj Pandey** | LLM prompt engineering · LLM-guided & random mutation strategies · Template-based refinement modules · LLM API integration & validation · Prompt optimization |
| **Varshitha Chowdary Thella** | Automated evaluation framework · Performance logging & data collection · Selection utilities · Experimental datasets · Performance validation |
| **Kevin Sanchez** | UI design & implementation · Fitness visualization components · Comparative analysis plots · UI–backend configuration controls · Reporting & result export |

---

## ✅ Prerequisites

- Python **3.9+**
- A **Google Gemini API key** (free tier available at [Google AI Studio](https://aistudio.google.com/))
- Internet connection (for LLM API calls)

---

## 📦 Required Libraries

Install all dependencies with:

```bash
pip install -r requirements.txt
```

**Core dependencies (`requirements.txt`):**

```
streamlit>=1.35.0
matplotlib>=3.8.0
numpy>=1.26.0
google-generativeai>=0.7.0
```

**Optional (for Vector DB acceleration):**

```
faiss-cpu>=1.8.0
sentence-transformers>=3.0.0
```

> ⚠️ If `faiss` or `sentence-transformers` are unavailable, the system automatically falls back to a list-based Vector DB — full functionality is preserved.

---

## ⚙️ Environment Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/<your-org>/evolve-agent.git
   cd evolve-agent
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python -m venv venv
   # Windows:
   venv\Scripts\activate
   # macOS/Linux:
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up your Gemini API key** 
   - **Streamlit UI:** Paste your API key in the sidebar when the app loads.
---

## ▶️ Running the Application

```bash
streamlit run Evolve.py
```

The app will open at **http://localhost:8501** in your browser.

---

## 🔄 Flow of Execution

```
1. User provides:
   - Gemini API Key (sidebar)
   - Problem type: Pacman or Matrix
   - Input type: Python / Other Language / Pseudocode / Plain Text
   - Initial code / algorithm description
   - Number of generations
   - Evolution strategy: No Evolution | Random | LLM-Guided
   - Fitness weights (w1, w2, w3)

2. Click "Start Evolution"

3. For each generation:
   a. Candidate Generator produces N mutated candidates
      - Random: AST comparison swap, constant perturbation, statement swap, template replacement
      - LLM-Guided: Gemini API call with top-VDB snippets as context
      - No Evolution: Single LLM call, no further mutation
   b. Evaluator runs each candidate in the Pacman/Matrix sandbox
      - Cache check: if hash(code) is known → reuse fitness (skip re-evaluation)
      - Compute fitness score
      - Log runtime, steps, generation number
   c. Selector keeps top-k candidates
   d. VDB updated with high-fitness snippets
   e. UI refreshes: metric cards, generation progress, log

4. After all generations:
   - Best candidate displayed
   - Fitness line chart, strategy bar chart, runtime chart shown
   - Export: CSV, JSON, PNG charts available in Tab 4
```

---

## 📊 Fitness Functions

### Pacman
```
Fitness = w1 × score + w2 × survival_time − w3 × steps
          where w1 + w2 + w3 = 1
```
Default weights: `w1=0.5, w2=0.3, w3=0.2`

### Matrix Multiplication (Bonus)
```
Fitness = correctness_score (0–60) + efficiency_score (0–40)
efficiency_score = max(0, 40 × (1 − ops / 54))
```
Naive = 54 ops | Strassen ≈ 51 | Target ≤ 40 for maximum efficiency score.

---

## 🔍 Features

- **Multi-strategy evolution:** No Evolution, Random Mutation, LLM-Guided Mutation, Human-in-Loop
- **Flexible input:** Python code, other languages (Java/C++/JS), pseudocode, plain text description
- **Two problem domains:** Pacman optimization + Matrix multiplication (bonus)
- **Configurable fitness weights** via UI sliders
- **Hash-based memoization cache** — zero cost re-evaluation for duplicate candidates
- **Vector DB acceleration** — FAISS or list-based, top-fitness snippet retrieval for LLM prompts
- **Real-time visualization** — fitness line chart, strategy bar chart, runtime chart
- **Export suite** — CSV data, JSON run summary, PNG charts
- **Multi-run analysis** — mean, std, min, max across repeated runs
- **Benchmark comparison** — side-by-side strategy comparison plots

---

## 🐛 Known Issues & Solutions

| Issue | Cause | Solution |
|-------|-------|----------|
| `faiss` import fails | Not installed or CPU/GPU mismatch | Run `pip install faiss-cpu`; app auto-falls back to list VDB |
| `sentence_transformers` slow first load | Model download on first use | Wait ~30s on first run; cached afterward |
| Gemini API `429 Resource Exhausted` | Free-tier rate limit hit | Wait 60s and retry; switch to `gemini-1.5-flash-8b` (higher rate limits) |
| `ast.unparse` not available | Python < 3.9 | Upgrade to Python 3.9+ |
| LLM returns non-Python code | Gemini hallucination | System auto-falls back to random mutation; increase temperature diversity |
| Streamlit reruns mid-evolution | Button state reset | Use `st.session_state` — already handled in current implementation |
| Low fitness scores in early generations | Cold start with no VDB context | Run ≥5 generations; cache/VDB warms up and improves LLM context |

---

## 💡 Suggestions & Feedback

- **Increase generations** (8–15) for more visible fitness improvement trends
- **Use LLM-Guided strategy** for the most significant per-generation improvements
- **Enable FAISS** for large runs (10+ generations) — significantly reduces redundant evaluations
- **Export data after each run** — the CSV export in Tab 4 is required for Round 2 submission
- For the Matrix bonus task, start with the default naive code and observe convergence toward Strassen-like solutions

---

## 📹 Video Walkthrough

> 📎 https://texastechuniversity-my.sharepoint.com/personal/sibollam_ttu_edu/_layouts/15/stream.aspx?id=%2Fpersonal%2Fsibollam%5Fttu%5Fedu%2FDocuments%2FEvolve%5FVideo%2Emp4&nav=eyJyZWZlcnJhbEluZm8iOnsicmVmZXJyYWxBcHAiOiJPbmVEcml2ZUZvckJ1c2luZXNzIiwicmVmZXJyYWxBcHBQbGF0Zm9ybSI6IldlYiIsInJlZmVycmFsTW9kZSI6InZpZXciLCJyZWZlcnJhbFZpZXciOiJNeUZpbGVzTGlua0NvcHkifX0&ga=1&referrer=StreamWebApp%2EWeb&referrerScenario=AddressBarCopied%2Eview%2E53e770cf%2D1a9a%2D4423%2D8c39%2D1aa5b7b904ef

---

## 📖 References

1. A. Novikov *et al.*, "AlphaEvolve: A coding agent for scientific and algorithmic discovery," *arXiv*:2506.13131, Jun. 2025. https://doi.org/10.48550/arXiv.2506.13131
2. S. Tamilselvi, "Introduction to Evolutionary Algorithms," IntechOpen, 2022. https://doi.org/10.5772/intechopen.104198
3. H. Amit, "An Overview of Evolutionary Algorithms," *We Talk Data*, 2025.
4. C. Fernando *et al.*, "Promptbreeder: Self-Referential Self-Improvement via Prompt Evolution," *arXiv*:2309.16797, 2023.
5. UC Berkeley AI Group. *Pacman AI Projects*. http://ai.berkeley.edu/project_overview.html
6. Google DeepMind. *Gemini API Documentation*. https://ai.google.dev/gemini-api/docs
7. V. Strassen, "Gaussian elimination is not optimal," *Numer. Math.*, 13(4), 1969.
8. J. Johnson, M. Douze, H. Jegou, "Billion-scale similarity search with GPUs," *IEEE Transactions on Big Data*, 2019.
9. N. Reimers & I. Gurevych, "Sentence-BERT," *arXiv*:1908.10084, 2019.
10. Python Software Foundation. *ast — Abstract Syntax Trees*. https://docs.python.org/3/library/ast.html

