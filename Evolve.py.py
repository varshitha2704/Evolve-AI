# =============================================================================
# Evolve Agent  –  CS5381 Analysis of Algorithms
# A Simplified Evolutionary Agent for Algorithm Discovery (AlphaEvolve-inspired)
# =============================================================================

import ast
import csv
import hashlib
import io
import random
import time
import textwrap
import threading
from typing import Optional, List, Dict, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

try:
    import faiss
    from sentence_transformers import SentenceTransformer
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

# =============================================================================
# CONSTANTS
# =============================================================================

NO_EVOLUTION = "No Evolution (Single-Shot LLM)"
RANDOM_MUT   = "Random"
LLM_GUIDED   = "LLM-Guided"
HUMAN        = "Human-in-Loop"
STRATEGIES   = [NO_EVOLUTION, RANDOM_MUT, LLM_GUIDED]

# Input-type constants (Fix 1, 3, 4)
INPUT_PYTHON      = "Python"
INPUT_OTHER_LANG  = "Other Language (e.g. Java, C++, JS)"
INPUT_PSEUDOCODE  = "Pseudocode / Algorithm Description"
INPUT_TEXT        = "Plain Text Description"
INPUT_TYPES       = [INPUT_PYTHON, INPUT_OTHER_LANG, INPUT_PSEUDOCODE, INPUT_TEXT]

# Languages supported for display (won't be executed directly)
OTHER_LANGUAGES   = ["Java", "C++", "JavaScript", "C", "Rust", "Go", "Other"]

PROBLEM_PACMAN = "Pacman"
PROBLEM_MATRIX = "3x3 Matrix Multiply (Bonus)"

ALLOWED_IMPORTS = {
    "math", "random", "collections", "heapq",
    "itertools", "functools", "copy", "typing"
}
DANGEROUS_NAMES = {
    "open", "exec", "eval", "__import__", "compile",
    "subprocess", "socket", "shutil", "pickle"
}

# Gemini models (free tier available via Google AI Studio)
GEMINI_MODELS = [
    "gemini-2.0-flash",
    "gemini-1.5-flash",
    "gemini-1.5-flash-8b",
    "gemini-1.5-pro",
]

DIRECTIONS = {
    "UP":    (0, -1),
    "DOWN":  (0,  1),
    "LEFT":  (-1,  0),
    "RIGHT": ( 1,  0),
}

PACMAN_MAPS = [
    [
        "##########",
        "#........#",
        "#.##.##..#",
        "#........#",
        "#.##.##..#",
        "#........#",
        "##########",
    ],
    [
        "############",
        "#..........#",
        "#.####.###.#",
        "#.#....#...#",
        "#.#.##.#.#.#",
        "#......#.#.#",
        "########.#.#",
        "#..........#",
        "############",
    ],
]

# =============================================================================
# PAGE CONFIG
# =============================================================================

st.set_page_config(
    page_title="Evolve Agent - CS5381",
    page_icon="🧬",
    layout="wide",
)

st.markdown("""
<style>
/* ── Base ── */
.block-container { padding-top: 0.8rem; max-width: 1400px; }

/* ── Workflow header banner ── */
.workflow-banner {
    background: #1e293b;
    border: 1px solid #334155;
    border-radius: 12px;
    padding: 18px 22px;
    margin-bottom: 20px;
}
.workflow-banner h3 {
    color: #f1f5f9;
    font-size: 15px;
    margin: 0 0 14px 0;
    font-weight: 700;
    letter-spacing: 0.3px;
}
.workflow-steps {
    display: flex;
    align-items: center;
    gap: 0;
    margin-bottom: 14px;
}
.step-box {
    flex: 1;
    text-align: center;
    padding: 10px 6px;
    border-radius: 8px;
    font-size: 12px;
    font-weight: 700;
    line-height: 1.4;
}
.step-arrow {
    flex: 0 0 24px;
    text-align: center;
    font-size: 18px;
    color: #64748b;
    font-weight: 300;
}
.step-1 { background: #312e81; color: #c7d2fe; border: 1px solid #4338ca; }
.step-2 { background: #14532d; color: #86efac; border: 1px solid #16a34a; }
.step-3 { background: #713f12; color: #fde68a; border: 1px solid #d97706; }
.step-4 { background: #7c2d12; color: #fed7aa; border: 1px solid #ea580c; }
.step-5 { background: #581c87; color: #e9d5ff; border: 1px solid #9333ea; }
.workflow-ops {
    color: #94a3b8;
    font-size: 12px;
    line-height: 1.6;
    border-top: 1px solid #334155;
    padding-top: 10px;
}
.op-badge {
    display: inline-block;
    padding: 2px 8px;
    border-radius: 4px;
    font-size: 11px;
    font-weight: 600;
    margin: 0 2px;
}
.op-no-evo  { background: #312e81; color: #c7d2fe; }
.op-random  { background: #14532d; color: #86efac; }
.op-llm     { background: #7c2d12; color: #fed7aa; }

/* ── Metric cards ── */
.metric-row {
    display: flex; gap: 10px; margin-bottom: 14px;
}
.metric-card {
    flex: 1;
    background: #1e293b;
    border: 1px solid #334155;
    border-radius: 10px;
    padding: 12px 10px;
    text-align: center;
}
.metric-card .label {
    color: #64748b;
    font-size: 11px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin-bottom: 4px;
}
.metric-card .value {
    color: #f1f5f9;
    font-size: 22px;
    font-weight: 800;
    line-height: 1.2;
}
.metric-card .value.green { color: #4ade80; }
.metric-card .value.blue  { color: #60a5fa; }
.metric-card .value.amber { color: #fbbf24; }
.metric-card .value.purple{ color: #c084fc; }

/* ── Improvement banner ── */
.improve-box {
    background: #052e16;
    border: 1px solid #16a34a;
    border-left: 4px solid #22c55e;
    border-radius: 8px;
    padding: 10px 14px;
    color: #86efac;
    font-weight: 700;
    font-size: 14px;
    margin-bottom: 10px;
}
.stagnate-box {
    background: #431407;
    border: 1px solid #ea580c;
    border-left: 4px solid #f97316;
    border-radius: 8px;
    padding: 10px 14px;
    color: #fed7aa;
    font-size: 13px;
    margin-bottom: 10px;
}

/* ── Winner box ── */
.winner-box {
    background: #1c1917;
    border: 2px solid #fbbf24;
    border-radius: 10px;
    padding: 14px 18px;
    margin-bottom: 12px;
    color: #fde68a;
    font-size: 15px;
    font-weight: 700;
}

/* ── Best solution box ── */
.best-box {
    background: #052e16;
    border-left: 4px solid #22c55e;
    padding: 12px 16px;
    border-radius: 8px;
    margin-bottom: 10px;
    color: #86efac;
    font-weight: 600;
}

/* ── Warn box ── */
.warn-box {
    background: #431407;
    border-left: 4px solid #f97316;
    padding: 8px 12px;
    border-radius: 6px;
    margin-bottom: 6px;
    font-size: 13px;
    color: #fed7aa;
}

/* ── Log box ── */
.log-box {
    background: #0d1117;
    color: #7dd3fc;
    font-family: 'JetBrains Mono', 'Fira Code', monospace;
    font-size: 11.5px;
    padding: 12px 14px;
    border-radius: 8px;
    border: 1px solid #1e293b;
    max-height: 320px;
    overflow-y: auto;
    white-space: pre-wrap;
    line-height: 1.7;
}

/* ── Strategy badges ── */
.strategy-badge {
    display: inline-block; padding: 3px 10px;
    border-radius: 12px; font-size: 12px;
    font-weight: 600; margin: 2px;
}
.badge-no-evo  { background:#312e81; color:#c7d2fe; }
.badge-random  { background:#14532d; color:#86efac; }
.badge-llm     { background:#7c2d12; color:#fed7aa; }
.badge-human   { background:#4a044e; color:#f0abfc; }

/* ── Generation progress bar ── */
.gen-progress-row {
    display: flex; align-items: center; gap: 10px;
    margin-bottom: 6px; font-size: 13px;
}
.gen-bar-wrap {
    flex: 1; background: #1e293b;
    border-radius: 6px; height: 18px;
    overflow: hidden; border: 1px solid #334155;
}
.gen-bar-fill {
    height: 100%;
    border-radius: 6px;
    transition: width 0.3s;
}
</style>
""", unsafe_allow_html=True)

# =============================================================================
# DEFAULT CODE / PROBLEM TEXT
# =============================================================================

DEFAULT_PACMAN_CODE = textwrap.dedent("""\
    def pacman_agent(state):
        \"\"\"Basic Pacman agent: flees ghosts, then chases nearest food.\"\"\"
        pos    = state['pacman']
        food   = state['food']
        ghosts = state.get('ghosts', [])

        for gx, gy in ghosts:
            if abs(gx - pos[0]) + abs(gy - pos[1]) <= 2:
                dx = pos[0] - gx
                dy = pos[1] - gy
                if abs(dx) >= abs(dy):
                    return 'RIGHT' if dx >= 0 else 'LEFT'
                return 'DOWN' if dy >= 0 else 'UP'

        if food:
            nearest = min(food,
                key=lambda f: abs(f[0] - pos[0]) + abs(f[1] - pos[1]))
            dx = nearest[0] - pos[0]
            dy = nearest[1] - pos[1]
            if abs(dx) > abs(dy):
                return 'RIGHT' if dx > 0 else 'LEFT'
            return 'DOWN' if dy > 0 else 'UP'
        return 'RIGHT'
""")

DEFAULT_MATRIX_CODE = textwrap.dedent("""\
    def matmul_3x3(A, B):
        \"\"\"Naive 3x3 matrix multiplication: 27 multiplications, 27 additions.\"\"\"
        C = [[0]*3 for _ in range(3)]
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    C[i][j] += A[i][k] * B[k][j]
        return C
""")

DEFAULT_PACMAN_PROBLEM = (
    "Optimize a Pacman AI agent to maximize game score by collecting food "
    "pellets, surviving as long as possible, and minimizing steps taken."
)
DEFAULT_MATRIX_PROBLEM = (
    "Minimize the number of scalar multiplications and additions needed to "
    "compute the product of two 3x3 matrices while keeping results correct."
)

# =============================================================================
# PACMAN MUTATION TEMPLATES
# =============================================================================

_PACMAN_TEMPLATES = [
    textwrap.dedent("""\
        def pacman_agent(state):
            \"\"\"Ghost-flee + nearest-food agent (flee radius = 3).\"\"\"
            pos    = state['pacman']
            food   = state['food']
            ghosts = state.get('ghosts', [])
            FLEE_DIST = 3
            for gx, gy in ghosts:
                if abs(gx - pos[0]) + abs(gy - pos[1]) <= FLEE_DIST:
                    dx = pos[0] - gx
                    dy = pos[1] - gy
                    if abs(dx) >= abs(dy):
                        return 'RIGHT' if dx >= 0 else 'LEFT'
                    return 'DOWN' if dy >= 0 else 'UP'
            if food:
                nearest = min(food,
                    key=lambda f: abs(f[0]-pos[0]) + abs(f[1]-pos[1]))
                dx = nearest[0] - pos[0]
                dy = nearest[1] - pos[1]
                if abs(dx) > abs(dy):
                    return 'RIGHT' if dx > 0 else 'LEFT'
                return 'DOWN' if dy > 0 else 'UP'
            return 'RIGHT'
    """),
    textwrap.dedent("""\
        def pacman_agent(state):
            \"\"\"Power-pellet priority agent.\"\"\"
            pos    = state['pacman']
            food   = state['food']
            ghosts = state.get('ghosts', [])
            pellets = [f for f in food if f[0] % 5 == 0]
            targets = pellets if pellets else food
            if targets:
                nearest = min(targets,
                    key=lambda f: abs(f[0]-pos[0]) + abs(f[1]-pos[1]))
                dx = nearest[0] - pos[0]
                dy = nearest[1] - pos[1]
                if abs(dx) >= abs(dy):
                    return 'RIGHT' if dx > 0 else 'LEFT'
                return 'DOWN' if dy > 0 else 'UP'
            return 'UP'
    """),
    textwrap.dedent("""\
        def pacman_agent(state):
            \"\"\"Cluster-aware food agent.\"\"\"
            pos  = state['pacman']
            food = state['food']
            if not food:
                return 'RIGHT'
            def cluster_score(f):
                return sum(
                    1 for g in food
                    if abs(g[0]-f[0]) + abs(g[1]-f[1]) <= 3
                )
            target = max(food, key=cluster_score)
            dx = target[0] - pos[0]
            dy = target[1] - pos[1]
            if abs(dx) >= abs(dy):
                return 'RIGHT' if dx > 0 else 'LEFT'
            return 'DOWN' if dy > 0 else 'UP'
    """),
    textwrap.dedent("""\
        def pacman_agent(state):
            \"\"\"Safety-first move-scoring agent.\"\"\"
            pos    = state['pacman']
            food   = state['food']
            ghosts = state.get('ghosts', [])
            walls  = state.get('walls', set())
            FLEE = 4
            def move_score(action):
                d = {'UP':(0,-1),'DOWN':(0,1),'LEFT':(-1,0),'RIGHT':(1,0)}[action]
                nx, ny = pos[0]+d[0], pos[1]+d[1]
                if (nx, ny) in walls:
                    return -999
                ghost_risk = sum(
                    1 for gx,gy in ghosts
                    if abs(nx-gx)+abs(ny-gy) < FLEE
                )
                food_bonus = 5 if (nx, ny) in food else 0
                return food_bonus - ghost_risk * 10
            return max(['UP','DOWN','LEFT','RIGHT'], key=move_score)
    """),
    textwrap.dedent("""\
        def pacman_agent(state):
            \"\"\"Ghost-penalised food targeting.\"\"\"
            pos    = state['pacman']
            food   = state['food']
            ghosts = state.get('ghosts', [])
            if not food:
                return 'RIGHT'
            def food_priority(f):
                dist = abs(f[0]-pos[0]) + abs(f[1]-pos[1])
                ghost_penalty = sum(
                    10 for gx,gy in ghosts
                    if abs(gx-f[0])+abs(gy-f[1]) <= 3
                )
                return dist + ghost_penalty
            target = min(food, key=food_priority)
            dx = target[0] - pos[0]
            dy = target[1] - pos[1]
            if abs(dx) > abs(dy):
                return 'RIGHT' if dx > 0 else 'LEFT'
            return 'DOWN' if dy > 0 else 'UP'
    """),
]

# =============================================================================
# MATRIX MUTATION TEMPLATES  (used by random_mutate_matrix)
# =============================================================================

_MATRIX_TEMPLATES = [
    textwrap.dedent("""\
        def matmul_3x3(A, B):
            \"\"\"Unrolled 3x3 multiply - explicit sums.\"\"\"
            C = [[0]*3 for _ in range(3)]
            for i in range(3):
                for j in range(3):
                    s = A[i][0]*B[0][j] + A[i][1]*B[1][j] + A[i][2]*B[2][j]
                    C[i][j] = s
            return C
    """),
    textwrap.dedent("""\
        def matmul_3x3(A, B):
            \"\"\"Transpose B then dot rows.\"\"\"
            Bt = [[B[j][i] for j in range(3)] for i in range(3)]
            C = [[0]*3 for _ in range(3)]
            for i in range(3):
                for j in range(3):
                    C[i][j] = sum(A[i][k]*Bt[j][k] for k in range(3))
            return C
    """),
    textwrap.dedent("""\
        def matmul_3x3(A, B):
            \"\"\"Fully unrolled - 27 explicit multiplications.\"\"\"
            return [
                [A[0][0]*B[0][0]+A[0][1]*B[1][0]+A[0][2]*B[2][0],
                 A[0][0]*B[0][1]+A[0][1]*B[1][1]+A[0][2]*B[2][1],
                 A[0][0]*B[0][2]+A[0][1]*B[1][2]+A[0][2]*B[2][2]],
                [A[1][0]*B[0][0]+A[1][1]*B[1][0]+A[1][2]*B[2][0],
                 A[1][0]*B[0][1]+A[1][1]*B[1][1]+A[1][2]*B[2][1],
                 A[1][0]*B[0][2]+A[1][1]*B[1][2]+A[1][2]*B[2][2]],
                [A[2][0]*B[0][0]+A[2][1]*B[1][0]+A[2][2]*B[2][0],
                 A[2][0]*B[0][1]+A[2][1]*B[1][1]+A[2][2]*B[2][1],
                 A[2][0]*B[0][2]+A[2][1]*B[1][2]+A[2][2]*B[2][2]],
            ]
    """),
    textwrap.dedent("""\
        def matmul_3x3(A, B):
            \"\"\"Row-cached multiply - reuse row references.\"\"\"
            C = [[0]*3 for _ in range(3)]
            for i in range(3):
                Ai = A[i]
                for j in range(3):
                    C[i][j] = Ai[0]*B[0][j] + Ai[1]*B[1][j] + Ai[2]*B[2][j]
            return C
    """),
]

# =============================================================================
# STRUCTURED RANDOM MUTATION  –  AST-based + string ops (Pacman)
# =============================================================================

def _ast_swap_comparison(code: str) -> Tuple[str, str]:
    """AST mutation: swap > with >= in first comparison found."""
    try:
        tree = ast.parse(code)

        class _Swapper(ast.NodeTransformer):
            def __init__(self): self.changed = False
            def visit_Compare(self, node):
                self.generic_visit(node)
                new_ops = []
                for op in node.ops:
                    if isinstance(op, ast.Gt) and not self.changed:
                        new_ops.append(ast.GtE()); self.changed = True
                    elif isinstance(op, ast.GtE) and not self.changed:
                        new_ops.append(ast.Gt()); self.changed = True
                    else:
                        new_ops.append(op)
                node.ops = new_ops
                return node

        s = _Swapper()
        new_tree = s.visit(tree)
        ast.fix_missing_locations(new_tree)
        if s.changed:
            return ast.unparse(new_tree), "AST: swapped comparison operator (> ↔ >=)"
    except Exception:
        pass
    return code, "AST comparison swap – no effect"


def _ast_mutate_constant(code: str) -> Tuple[str, str]:
    """AST mutation: perturb one numeric constant by ±1."""
    try:
        tree = ast.parse(code)
        constants = [
            n for n in ast.walk(tree)
            if isinstance(n, ast.Constant)
            and isinstance(n.value, (int, float))
            and n.value not in (0, 1)
            and 1 < abs(n.value) < 20
        ]
        if constants:
            target = random.choice(constants)
            old_val = target.value
            target.value = type(target.value)(old_val + random.choice([-1, 1]))
            ast.fix_missing_locations(tree)
            return ast.unparse(tree), \
                "AST: constant {} -> {}".format(old_val, target.value)
    except Exception:
        pass
    return code, "AST constant mutate – no effect"


def _ast_swap_stmts(code: str) -> Tuple[str, str]:
    """AST mutation: swap two consecutive non-return statements."""
    try:
        tree = ast.parse(code)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and len(node.body) >= 4:
                idx = random.randint(0, len(node.body) - 2)
                a, b = node.body[idx], node.body[idx + 1]
                if not isinstance(a, (ast.FunctionDef, ast.ClassDef, ast.Return)) and \
                   not isinstance(b, (ast.FunctionDef, ast.ClassDef, ast.Return)):
                    node.body[idx], node.body[idx + 1] = b, a
                    ast.fix_missing_locations(tree)
                    return ast.unparse(tree), \
                        "AST: swapped statements at positions {}/{}".format(
                            idx + 1, idx + 2)
    except Exception:
        pass
    return code, "AST statement swap – no effect"


_STRING_OPS = [
    (lambda c: c.replace("return 'RIGHT'", "return 'UP'"),
     "String: fallback direction RIGHT -> UP"),
    (lambda c: c.replace("return 'UP'", "return 'RIGHT'"),
     "String: fallback direction UP -> RIGHT"),
    (lambda c: c.replace("<= 2", "<= 3"),
     "String: ghost detection radius 2 -> 3"),
    (lambda c: c.replace("<= 3", "<= 4"),
     "String: ghost detection radius 3 -> 4"),
    (lambda c: c.replace("abs(dx) > abs(dy)", "abs(dx) * 1.1 > abs(dy)"),
     "String: added 1.1 horizontal-bias multiplier"),
    (lambda c: c.replace("abs(dx) * 1.1 > abs(dy)", "abs(dx) > abs(dy)"),
     "String: removed horizontal-bias multiplier"),
]


def random_mutate(code: str, problem_type: str = PROBLEM_PACMAN) -> Tuple[str, str]:
    """
    Dispatch to the appropriate mutation strategy based on problem type.
    Pacman: AST + template mutations.
    Matrix: template substitution + constant perturbation.
    """
    if problem_type == PROBLEM_MATRIX:
        return random_mutate_matrix(code)
    return random_mutate_pacman(code)


def random_mutate_pacman(code: str) -> Tuple[str, str]:
    """
    Pacman structured mutation — four modes weighted by probability:
      25% template substitution
      20% AST comparison swap
      20% AST constant perturbation
      15% AST statement swap
      20% targeted string replacement
    """
    r = random.random()
    if r < 0.25:
        tpl = random.choice(_PACMAN_TEMPLATES)
        return tpl, "Template substitution from predefined library"
    elif r < 0.45:
        return _ast_swap_comparison(code)
    elif r < 0.65:
        return _ast_mutate_constant(code)
    elif r < 0.80:
        return _ast_swap_stmts(code)
    else:
        fn, desc = random.choice(_STRING_OPS)
        try:
            return fn(code), desc
        except Exception:
            return code, "String op – no effect"


def random_mutate_matrix(code: str) -> Tuple[str, str]:
    """
    Matrix structured mutation:
      40% template substitution (known correct variants)
      30% AST constant perturbation (tweak loop bounds, indices)
      30% AST comparison / structural swap
    """
    r = random.random()
    if r < 0.40:
        tpl = random.choice(_MATRIX_TEMPLATES)
        return tpl, "Matrix template substitution"
    elif r < 0.70:
        return _ast_mutate_constant(code)
    else:
        return _ast_swap_comparison(code)


# =============================================================================
# DEMO LLM MUTATION FOR MATRIX (no API key)
# =============================================================================

def llm_mutate_demo_matrix(code: str, gen: int, context_snippets: list) -> Tuple[str, str]:
    """
    Deterministic demo for matrix mode: cycles through templates that are
    genuinely different from each other so fitness can improve over gens.
    """
    # Prefer high-fitness VDB entry first
    if context_snippets:
        best = max(context_snippets, key=lambda r: r["fitness"])
        if best["fitness"] > 0 and best["code"] != code:
            return best["code"], "[DEMO LLM] VDB retrieval fitness={:.1f}".format(best["fitness"])

    # Cycle through matrix templates, then try constant perturbation
    all_candidates = list(_MATRIX_TEMPLATES)
    idx = (gen - 1) % len(all_candidates)
    candidate = all_candidates[idx]
    if candidate.strip() != code.strip():
        return candidate, "[DEMO LLM] Matrix template gen {} idx {}".format(gen, idx)

    # Fallback: perturb a constant
    mutated, desc = _ast_mutate_constant(code)
    return mutated, "[DEMO LLM] Matrix constant perturb: " + desc

# =============================================================================
# INPUT NORMALIZATION  (Fix 2, 4)
# Convert non-Python input into a runnable Python stub so the rest of the
# pipeline (mutation, evaluation, validation) can operate normally.
# =============================================================================

def normalize_to_python(code: str, input_type: str, problem_type: str,
                        api_key: str = "", model_name: str = "",
                        language: str = "") -> Tuple[str, str]:
    """
    Converts pseudocode / text descriptions / foreign-language code into a
    Python stub that the evaluator can run.  Returns (python_code, note).

    Strategy:
      - INPUT_PYTHON     → returned as-is.
      - INPUT_OTHER_LANG → if API key available, ask LLM to translate to Python;
                           otherwise wrap in a docstring comment stub.
      - INPUT_PSEUDOCODE → if API key available, ask LLM to implement in Python;
                           otherwise return appropriate default template.
      - INPUT_TEXT       → same as pseudocode.
    """
    if input_type == INPUT_PYTHON:
        return code, "Input is Python — used directly."

    note_prefix = {
        INPUT_OTHER_LANG: "Translated from {} to Python".format(language or "other language"),
        INPUT_PSEUDOCODE: "Implemented from pseudocode",
        INPUT_TEXT:       "Implemented from text description",
    }.get(input_type, "Converted to Python")

    # If an API key is available, use the LLM to translate / implement
    if api_key.strip():
        if problem_type == PROBLEM_MATRIX:
            fn_sig = "def matmul_3x3(A, B):"
            fn_req = ("Return ONLY a valid Python function `def matmul_3x3(A, B):` "
                      "that multiplies two 3x3 matrices stored as list-of-lists. "
                      "No imports, no markdown fences.")
        else:
            fn_sig = "def pacman_agent(state):"
            fn_req = ("Return ONLY a valid Python function `def pacman_agent(state):` "
                      "that reads state['pacman'], state['food'], state['ghosts'], state['walls'] "
                      "and returns one of 'UP','DOWN','LEFT','RIGHT'. "
                      "No imports, no markdown fences.")

        if input_type == INPUT_OTHER_LANG:
            prompt = (
                "Translate the following {} code to Python.\n"
                "{}\n\n"
                "Source code:\n```\n{}\n```\n\n{}".format(
                    language or "foreign-language", fn_req, code, fn_req)
            )
        else:
            prompt = (
                "Implement the following algorithm description as Python code.\n"
                "Description:\n{}\n\n{}".format(code, fn_req)
            )

        try:
            if not GEMINI_AVAILABLE:
                raise ImportError("google-generativeai not installed. Run: pip install google-generativeai")
            genai.configure(api_key=api_key)
            _gmodel = genai.GenerativeModel(model_name or "gemini-2.0-flash")
            resp = _gmodel.generate_content(prompt)
            translated = _strip_fences(resp.text)
            return translated, "[Gemini] {} via API.".format(note_prefix)
        except Exception as exc:
            _fallback_reason = str(exc)[:200]
            # No API key (or LLM failed) — return appropriate default template
            if problem_type == PROBLEM_MATRIX:
                stub = DEFAULT_MATRIX_CODE
            else:
                stub = DEFAULT_PACMAN_CODE
            note = (
                "⚠️ LLM translation failed for {} — falling back to default template.\n"
                "**Reason:** `{}`\n"
                "Fix: check your API key, quota, or enter Python code directly.".format(
                    note_prefix, _fallback_reason)
            )
            return stub, note

    # No API key — return appropriate default template
    if problem_type == PROBLEM_MATRIX:
        stub = DEFAULT_MATRIX_CODE
    else:
        stub = DEFAULT_PACMAN_CODE

    note = (
        "⚠️ No API key provided — cannot translate {} input automatically.\n"
        "Using the default Python template as a starting point.\n"
        "To enable automatic translation: enter your Google Gemini API key in the sidebar, "
        "or paste Python code directly.".format(input_type)
    )
    return stub, note


# =============================================================================
# UTILITY
# =============================================================================

def code_hash(code: str) -> str:
    return hashlib.md5(code.encode("utf-8", errors="replace")).hexdigest()[:12]

# =============================================================================
# SESSION STATE
# =============================================================================

def _init_state():
    defaults = {
        "history":           [],
        "best_code":         "",
        "best_fitness":      0.0,
        "best_strategy":     NO_EVOLUTION,
        "log_lines":         [],
        "cache":             {},
        "vdb":               None,
        "ran_once":          False,
        "human_code":        "",
        "runtime_total":     0.0,
        "multi_run_data":    [],
        "benchmark_results": {},
        "accel_cache_hits":       0,
        "accel_unique_evals":     0,
        "accel_reused_candidates":0,
        "accel_total_eval_time":  0.0,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

_init_state()

# =============================================================================
# LOGGING
# =============================================================================

def log(msg: str):
    st.session_state.log_lines.append(str(msg))

def log_sep(gen: int):
    log("-- Generation {} {}".format(gen, "-" * 26))

# =============================================================================
# SAFE EXECUTION
# =============================================================================

class _ExecTimeoutError(Exception): pass


def _run_with_timeout(fn, timeout_s: float = 2.0):
    result_holder = [None]
    error_holder  = [None]
    def _target():
        try:    result_holder[0] = fn()
        except Exception as exc: error_holder[0] = exc
    t = threading.Thread(target=_target, daemon=True)
    t.start(); t.join(timeout_s)
    if t.is_alive():
        raise _ExecTimeoutError("execution exceeded {:.1f}s".format(timeout_s))
    if error_holder[0] is not None:
        raise error_holder[0]
    return result_holder[0]

# =============================================================================
# VALIDATION
# =============================================================================

def validate_candidate(code: str, input_type: str = INPUT_PYTHON) -> Tuple[bool, str]:
    """
    Validates a candidate code string.
    - For INPUT_PYTHON: full AST parse + import/call safety checks.
    - For non-Python inputs: only checks non-empty (actual execution validation
      happens after normalization to Python).
    """
    if not code or not code.strip():
        return False, "empty code"

    if input_type != INPUT_PYTHON:
        # Non-Python / pseudocode are valid as long as non-empty.
        # They will be converted to Python before execution.
        return True, "non-Python input accepted (will be normalized before evaluation)"

    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        # Fix 8: return a clear, user-visible syntax error message
        return False, "SyntaxError at line {}: {}".format(
            getattr(exc, 'lineno', '?'), exc.msg if hasattr(exc, 'msg') else str(exc))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                if root not in ALLOWED_IMPORTS:
                    return False, "disallowed import: '{}'. Allowed: {}".format(
                        root, ", ".join(sorted(ALLOWED_IMPORTS)))
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                root = node.module.split(".")[0]
                if root not in ALLOWED_IMPORTS:
                    return False, "disallowed import: '{}'. Allowed: {}".format(
                        root, ", ".join(sorted(ALLOWED_IMPORTS)))
        elif isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name) and func.id in DANGEROUS_NAMES:
                return False, "disallowed call: '{}' is not permitted for safety reasons.".format(func.id)
            if isinstance(func, ast.Attribute) and func.attr in DANGEROUS_NAMES:
                return False, "disallowed call: '{}' is not permitted for safety reasons.".format(func.attr)
    return True, "ok"

# =============================================================================
# VECTOR DB
# =============================================================================

class VectorDB:
    def __init__(self):
        self.records = []
        self._index = self._model = None
        self._dim = 384; self._ready = False
        if FAISS_AVAILABLE:
            try:
                self._model = SentenceTransformer("all-MiniLM-L6-v2")
                self._index = faiss.IndexFlatL2(self._dim)
                self._ready = True
            except Exception: pass

    def add(self, code: str, fitness: float, op: str):
        self.records.append({"code": code, "fitness": fitness, "op": op})
        if self._ready:
            try:
                emb = self._model.encode([code], normalize_embeddings=True)
                self._index.add(emb.astype("float32"))
            except Exception: pass

    def retrieve(self, query_code: str, k: int = 2) -> List[Dict]:
        if not self.records: return []
        if self._ready and len(self.records) >= k:
            try:
                emb = self._model.encode([query_code], normalize_embeddings=True)
                _, idxs = self._index.search(emb.astype("float32"), k)
                return [self.records[i] for i in idxs[0] if 0 <= i < len(self.records)]
            except Exception: pass
        return sorted(self.records, key=lambda r: r["fitness"], reverse=True)[:k]

    @property
    def size(self) -> int: return len(self.records)
    @property
    def backend(self) -> str:
        return "FAISS + sentence-transformers" if self._ready \
               else "list fallback (pip install faiss-cpu sentence-transformers)"

def _get_vdb() -> VectorDB:
    if st.session_state.vdb is None:
        st.session_state.vdb = VectorDB()
    return st.session_state.vdb

# =============================================================================
# PACMAN SIMULATOR
# =============================================================================

def _parse_map(grid):
    walls, food, ghosts, pacman = set(), set(), [], (1, 1)
    for y, row in enumerate(grid):
        for x, ch in enumerate(row):
            if   ch == "#": walls.add((x, y))
            elif ch == ".": food.add((x, y))
            elif ch == "P": pacman = (x, y)
            elif ch == "G": ghosts.append((x, y))
    return walls, food, pacman, ghosts


def _ghost_move(ghost, pacman, walls, grid_w, grid_h):
    gx, gy = ghost; px, py = pacman
    best = ghost; best_dist = abs(gx-px)+abs(gy-py)
    for dx, dy in DIRECTIONS.values():
        nx, ny = gx+dx, gy+dy
        if (nx,ny) not in walls and 0<=nx<grid_w and 0<=ny<grid_h:
            d = abs(nx-px)+abs(ny-py)
            if d < best_dist: best_dist = d; best = (nx, ny)
    return best


def run_pacman_game(agent_code: str, map_idx: int = 0,
                    max_steps: int = 200, seed: int = 42) -> Dict:
    ok, reason = validate_candidate(agent_code, INPUT_PYTHON)
    if not ok:
        return {"score":0,"survival":0,"steps":0,"ate_all":False,"error":reason}
    grid = PACMAN_MAPS[map_idx % len(PACMAN_MAPS)]
    grid_h, grid_w = len(grid), len(grid[0])
    walls, food_set, pacman, ghost_starts = _parse_map(grid)
    if not ghost_starts: ghost_starts = [(grid_w-2, grid_h-2)]
    ghosts = list(ghost_starts); food = set(food_set)
    ns = {}
    try:
        def _compile(): exec(compile(agent_code, "<agent>", "exec"), ns)
        _run_with_timeout(_compile, timeout_s=2.0)
    except Exception as exc:
        return {"score":0,"survival":0,"steps":0,"ate_all":False,"error":str(exc)}
    agent_fn = ns.get("pacman_agent")
    if agent_fn is None:
        return {"score":0,"survival":0,"steps":0,"ate_all":False,
                "error": (
                    "❌ Required function 'pacman_agent' not found. "
                    "Your code MUST define: def pacman_agent(state): "
                    "that returns one of 'UP', 'DOWN', 'LEFT', 'RIGHT'. "
                    "Check that the function name is spelled exactly as 'pacman_agent' "
                    "(case-sensitive) and is at the top level (not nested inside another function or class)."
                )}
    score = 0; step = 0; rng = random.Random(seed)
    for step in range(1, max_steps+1):
        state = {"pacman":pacman,"food":list(food),"ghosts":list(ghosts),"walls":walls}
        try:
            def _call(): return agent_fn(state)
            action = _run_with_timeout(_call, timeout_s=0.3)
        except Exception: action = "RIGHT"
        if action not in DIRECTIONS: action = "RIGHT"
        dx, dy = DIRECTIONS[action]; nx, ny = pacman[0]+dx, pacman[1]+dy
        if (nx,ny) in walls: score -= 1
        else: pacman = (nx, ny)
        if pacman in food: food.discard(pacman); score += 10
        ghosts = [_ghost_move(g, pacman, walls, grid_w, grid_h) for g in ghosts]
        if pacman in ghosts: score -= 50; break
        if not food: score += 100; break
    return {"score":max(0,score),"survival":step,"steps":step,
            "ate_all":len(food)==0,"error":None}


def evaluate_pacman(agent_code: str, runs: int = 2) -> Tuple[float,float,float]:
    scores, survivals, costs = [], [], []
    for seed in range(runs):
        for map_idx in range(len(PACMAN_MAPS)):
            res = run_pacman_game(agent_code, map_idx=map_idx, seed=seed)
            if res["error"]:
                scores.append(0.0); survivals.append(0.0); costs.append(100.0)
            else:
                scores.append(   min(100.0, res["score"]    / 4.0))
                survivals.append(min(100.0, res["survival"] / 2.0))
                costs.append(    min(100.0, res["steps"]    / 2.0))
    return (round(float(np.mean(scores)),1),
            round(float(np.mean(survivals)),1),
            round(float(np.mean(costs)),1))

# =============================================================================
# MATRIX FITNESS  –  executed operation counting via _CountedNum
# =============================================================================

_MATRIX_TESTS = [
    ([[1,2,3],[4,5,6],[7,8,9]], [[9,8,7],[6,5,4],[3,2,1]],
     [[30,24,18],[84,69,54],[138,114,90]]),
    ([[1,0,0],[0,1,0],[0,0,1]], [[5,6,7],[8,9,10],[11,12,13]],
     [[5,6,7],[8,9,10],[11,12,13]]),
    ([[2,2,2],[2,2,2],[2,2,2]], [[1,1,1],[1,1,1],[1,1,1]],
     [[6,6,6],[6,6,6],[6,6,6]]),
]


class _CountedNum:
    """
    Wraps a numeric value and intercepts * / + / - so that matrix code
    evaluated at runtime produces real operation counts instead of AST estimates.
    """
    _mul_count = 0
    _add_count = 0

    @classmethod
    def reset(cls):
        cls._mul_count = 0
        cls._add_count = 0

    def __init__(self, v): self.v = v
    def __mul__(self, other):
        _CountedNum._mul_count += 1
        return _CountedNum(self.v * (other.v if isinstance(other, _CountedNum) else other))
    def __rmul__(self, other):
        _CountedNum._mul_count += 1
        return _CountedNum((other.v if isinstance(other, _CountedNum) else other) * self.v)
    def __add__(self, other):
        _CountedNum._add_count += 1
        return _CountedNum(self.v + (other.v if isinstance(other, _CountedNum) else other))
    def __radd__(self, other):
        _CountedNum._add_count += 1
        return _CountedNum((other.v if isinstance(other, _CountedNum) else other) + self.v)
    def __sub__(self, other):
        _CountedNum._add_count += 1
        return _CountedNum(self.v - (other.v if isinstance(other, _CountedNum) else other))
    def __rsub__(self, other):
        _CountedNum._add_count += 1
        return _CountedNum((other.v if isinstance(other, _CountedNum) else other) - self.v)
    def __eq__(self, other):
        return self.v == (other.v if isinstance(other, _CountedNum) else other)
    def __repr__(self): return repr(self.v)


def _wrap_matrix(M):
    return [[_CountedNum(v) for v in row] for row in M]


def _unwrap_matrix(M):
    return [[v.v if isinstance(v, _CountedNum) else v for v in row] for row in M]


def compute_fitness_matrix(code: str) -> Tuple[float, int, int]:
    """
    Returns (fitness, avg_muls, avg_adds).
    Operation counts come from executed instrumented arithmetic.
    Naive = 27 muls + 27 adds = 54 total ops (executed).
    Target <= 40 total ops for maximum efficiency score.
    """
    ok, _ = validate_candidate(code, INPUT_PYTHON)
    if not ok: return 0.0, 0, 0
    ns = {}
    try:
        def _exec(): exec(compile(code, "<matrix>", "exec"), ns)
        _run_with_timeout(_exec, timeout_s=2.0)
        fn = ns.get("matmul_3x3")
        if fn is None: 
            return (0.0, 0, 0)  # error logged via caller; clear message shown in UI

        total_muls = 0; total_adds = 0
        for A, B, expected in _MATRIX_TESTS:
            _CountedNum.reset()
            A_w = _wrap_matrix(A); B_w = _wrap_matrix(B)
            try:
                def _call(): return fn(A_w, B_w)
                result_w = _run_with_timeout(_call, timeout_s=1.0)
            except Exception:
                return 0.0, 0, 0
            result = _unwrap_matrix(result_w)
            if result != expected: return 0.0, 0, 0
            total_muls += _CountedNum._mul_count
            total_adds += _CountedNum._add_count

        avg_muls = total_muls // len(_MATRIX_TESTS)
        avg_adds = total_adds // len(_MATRIX_TESTS)
        correctness = 60.0
    except Exception:
        return 0.0, 0, 0

    total_ops = avg_muls + avg_adds
    # Naive = 54 ops, Target <= 40 for full efficiency bonus
    efficiency = max(0.0, 40.0 - max(0, total_ops - 40) * 0.5)
    fitness = round(correctness + efficiency, 2)
    return fitness, avg_muls, avg_adds


def compute_fitness_pacman(score_n, survival_n, cost_n, w1, w2, w3) -> float:
    return round(w1*score_n + w2*survival_n - w3*cost_n, 2)

# =============================================================================
# MEMOIZED EVALUATION  –  hash-keyed, never re-evaluates identical code
# =============================================================================

def evaluate_candidate_cached(
    code: str, problem_type: str, eval_runs: int,
    w1: float, w2: float, w3: float,
) -> Tuple[float, dict, bool]:
    """
    Returns (fitness, metrics_dict, was_cache_hit).

    Pacman  metrics: score, survival, cost
    Matrix  metrics: correctness, op_count, mul_count, add_count, matrix_fitness
    """
    h = code_hash(code)
    entry = st.session_state.cache.get(h, {})
    if "fitness" in entry:
        st.session_state.accel_cache_hits        += 1
        st.session_state.accel_reused_candidates += 1
        return entry["fitness"], entry["metrics"], True

    t0 = time.time()
    if problem_type == PROBLEM_MATRIX:
        fitness, avg_muls, avg_adds = compute_fitness_matrix(code)
        correctness = 60.0 if fitness > 0 else 0.0
        op_count    = avg_muls + avg_adds
        metrics = {
            "correctness":    correctness,
            "op_count":       op_count,
            "mul_count":      avg_muls,
            "add_count":      avg_adds,
            "matrix_fitness": fitness,
            # Neutral aliases kept for chart/CSV code
            "score":    fitness,
            "survival": float(avg_muls),
            "cost":     float(avg_adds),
        }
    else:
        score_n, survival_n, cost_n = evaluate_pacman(code, runs=eval_runs)
        fitness = compute_fitness_pacman(score_n, survival_n, cost_n, w1, w2, w3)
        metrics = {
            "score":    score_n,
            "survival": survival_n,
            "cost":     cost_n,
        }

    st.session_state.accel_total_eval_time += round(time.time() - t0, 4)
    st.session_state.accel_unique_evals    += 1

    entry.update({"fitness": fitness, "metrics": metrics})
    st.session_state.cache[h] = entry
    return fitness, metrics, False

# =============================================================================
# LLM MUTATION
# =============================================================================

def _build_prompt(code, problem, gen, problem_type, context_snippets):
    ctx = ""
    if context_snippets:
        top = context_snippets[0]
        ctx = ("\nHigh-fitness reference (fitness={:.1f}):\n"
               "```python\n{}\n```\n").format(top["fitness"], top["code"][:600])
    if problem_type == PROBLEM_MATRIX:
        return (
            "Problem: {}\n{}\n"
            "Current implementation (Generation {}):\n```python\n{}\n```\n\n"
            "Reduce scalar multiplications and additions while keeping correctness. "
            "Return ONLY the improved Python function def matmul_3x3(A, B): "
            "with no markdown fences."
        ).format(problem, ctx, gen, code)
    return (
        "Problem: {}\n{}\n"
        "Current Pacman agent (Generation {}):\n```python\n{}\n```\n\n"
        "Improve this Pacman agent. Rules:\n"
        "1. Keep signature: def pacman_agent(state):\n"
        "2. state keys: 'pacman'=(x,y), 'food'=[(x,y),...], "
        "'ghosts'=[(x,y),...], 'walls'=set\n"
        "3. Return exactly one of: 'UP' 'DOWN' 'LEFT' 'RIGHT'\n"
        "4. No import statements.\n"
        "Add one inline comment. Return ONLY the function, no markdown fences."
    ).format(problem, ctx, gen, code)


def _strip_fences(raw: str) -> str:
    for fence in ["```python", "```"]:
        if fence in raw:
            parts = raw.split(fence)
            raw = parts[1].split("```")[0] if len(parts) >= 3 else parts[1]
    return raw.strip()


def llm_mutate_real(code, problem, gen, api_key, problem_type,
                    context_snippets, model_name="gemini-2.0-flash"):
    """Call Google Gemini API for LLM-guided mutation."""
    if not GEMINI_AVAILABLE:
        raise RuntimeError(
            "google-generativeai not installed. "
            "Run: pip install google-generativeai"
        )
    genai.configure(api_key=api_key)
    model  = genai.GenerativeModel(model_name)
    prompt = _build_prompt(code, problem, gen, problem_type, context_snippets)
    backoff = 1.0
    last_err = "unknown"
    for attempt in range(3):
        try:
            resp  = model.generate_content(prompt)
            raw   = _strip_fences(resp.text)
            label = "[GEMINI] {} (gen {}, attempt {})".format(
                model_name, gen, attempt + 1)
            return raw, label
        except Exception as exc:
            last_err = str(exc)[:120]
            # Back off on quota / rate-limit errors
            if "quota" in last_err.lower() or "429" in last_err:
                time.sleep(backoff); backoff *= 2
            else:
                break
    raise RuntimeError("Gemini API failed: {}".format(last_err))


def llm_mutate_demo(code, gen, context_snippets, problem_type=PROBLEM_PACMAN):
    """Demo fallback when no API key. Routes to problem-specific demo mutator."""
    if problem_type == PROBLEM_MATRIX:
        return llm_mutate_demo_matrix(code, gen, context_snippets)

    # Pacman demo
    if context_snippets:
        best = max(context_snippets, key=lambda r: r["fitness"])
        if best["fitness"] > 0 and best["code"] != code:
            return (best["code"],
                    "[DEMO LLM] VDB retrieval fitness={:.1f}: {}".format(
                        best["fitness"], best["op"]))
    ideas = [
        ("flee radius 3 + nearest food",    _PACMAN_TEMPLATES[0]),
        ("food cluster targeting",           _PACMAN_TEMPLATES[2]),
        ("power-pellet priority",            _PACMAN_TEMPLATES[1]),
        ("safety-first move scoring",        _PACMAN_TEMPLATES[3]),
        ("ghost-penalised food targeting",   _PACMAN_TEMPLATES[4]),
        ("horizontal bias 1.15x",
         code.replace("abs(dx) > abs(dy)", "abs(dx) * 1.15 > abs(dy)")),
        ("wider ghost flee radius",
         code.replace("<= 2", "<= 4")),
    ]
    label, new_code = ideas[(gen-1) % len(ideas)]
    return new_code, "[DEMO LLM] Gen {}: {}".format(gen, label)

# =============================================================================
# SINGLE-SHOT LLM BASELINE  (true No-Evolution mode)
# =============================================================================

def single_shot_llm(initial_code, problem, api_key, problem_type, model_name):
    """One LLM call, no iteration, no selection — true baseline."""
    if api_key.strip():
        try:
            code, op = llm_mutate_real(
                initial_code, problem, 0, api_key, problem_type, [], model_name)
            return code, "[SINGLE-SHOT LLM] " + op
        except Exception:
            pass
    return initial_code, "[DEMO FALLBACK] No API key - initial code returned unmodified"

# =============================================================================
# MAIN EVOLUTION LOOP
# =============================================================================

def run_evolution(
    initial_code, problem, api_key, n_gen,
    w1, w2, w3, top_k, human_code, problem_type,
    fixed_seed, eval_runs, model_name, pop_size,
):
    if fixed_seed is not None:
        random.seed(fixed_seed); np.random.seed(fixed_seed)

    st.session_state.history   = []
    st.session_state.log_lines = []
    st.session_state.cache     = {}
    st.session_state.vdb       = VectorDB()
    st.session_state.accel_cache_hits        = 0
    st.session_state.accel_unique_evals      = 0
    st.session_state.accel_reused_candidates = 0
    st.session_state.accel_total_eval_time   = 0.0
    vdb = st.session_state.vdb

    active_strategies = list(STRATEGIES)
    if human_code and human_code.strip():
        v, r = validate_candidate(human_code, INPUT_PYTHON)
        if v:   active_strategies.append(HUMAN);  log("Human code validated OK")
        else:   log("Human code rejected ({}), excluded".format(r))

    best_code     = initial_code
    best_fitness  = 0.0
    best_strategy = NO_EVOLUTION
    current_pool  = [initial_code] * pop_size
    total_start   = time.time()

    # Per-strategy running best — ensures cumulative improvement curve
    strategy_best_code    = {s: initial_code for s in STRATEGIES + [HUMAN]}
    strategy_best_fitness = {s: 0.0          for s in STRATEGIES + [HUMAN]}

    log("Evolution started")
    log("  Problem     : {}".format(problem_type))
    log("  Generations : {}   top_k={}   eval_runs={}   pop_size={}".format(
        n_gen, top_k, eval_runs, pop_size))
    log("  Weights     : w1={} w2={} w3={}".format(w1, w2, w3))
    log("  LLM mode    : {}".format(
        "REAL API - model: {}".format(model_name) if api_key.strip() else "DEMO (no key)"))
    log("  Seed        : {}".format(fixed_seed if fixed_seed is not None else "random"))
    log("  VDB backend : {}".format(vdb.backend))
    log("")

    for gen in range(1, n_gen+1):
        gen_start   = time.time()
        steps_count = [0]
        log_sep(gen)
        gen_row        = {"gen": gen}
        gen_candidates = []

        for strategy in active_strategies:
            s_best_fit     = strategy_best_fitness[strategy]
            s_best_code    = strategy_best_code[strategy]
            s_best_op      = ""
            s_best_metrics = {}

            # No-evolution: single shot only (no pop loop)
            n_trials = 1 if strategy == NO_EVOLUTION else pop_size

            for _trial in range(n_trials):
                parent = strategy_best_code[strategy] if strategy != NO_EVOLUTION \
                         else initial_code

                # ── Generate ──────────────────────────────────────────────────
                if strategy == NO_EVOLUTION:
                    candidate, op = single_shot_llm(
                        initial_code, problem, api_key, problem_type, model_name)

                elif strategy == RANDOM_MUT:
                    # FIX: pass problem_type so matrix gets matrix mutations
                    candidate, op = random_mutate(parent, problem_type)
                    ok, reason = validate_candidate(candidate, INPUT_PYTHON)
                    if not ok:
                        log("  [Random] Invalid -> reverted ({})".format(reason))
                        candidate = parent; op += " | reverted"

                elif strategy == LLM_GUIDED:
                    h   = code_hash(parent)
                    ctx = vdb.retrieve(parent, k=2)
                    if h in st.session_state.cache and \
                       "llm_code" in st.session_state.cache[h]:
                        candidate = st.session_state.cache[h]["llm_code"]
                        op = "[CACHE HIT] Reused prior LLM mutation"
                        log("  [LLM] Cache hit for code hash {}".format(h))
                    elif api_key.strip():
                        try:
                            candidate, op = llm_mutate_real(
                                parent, problem, gen, api_key,
                                problem_type, ctx, model_name)
                            st.session_state.cache.setdefault(h, {})["llm_code"] = candidate
                            log("  [LLM] Real API mutation succeeded")
                        except Exception as exc:
                            _err = str(exc)[:80]
                            log("  [LLM] ⚠ API call failed — reason: {}".format(_err))
                            log("  [LLM] Falling back to demo template mutation")
                            # FIX: pass problem_type to demo fallback
                            candidate, op = llm_mutate_demo(parent, gen, ctx, problem_type)
                            op += " | LLM fallback (reason: {})".format(_err)
                    else:
                        # FIX: pass problem_type to demo fallback
                        candidate, op = llm_mutate_demo(parent, gen, ctx, problem_type)
                        st.session_state.cache.setdefault(h, {})["llm_code"] = candidate

                    ok, reason = validate_candidate(candidate, INPUT_PYTHON)
                    if not ok:
                        log("  [LLM] Invalid -> reverted ({})".format(reason))
                        candidate = parent; op += " | reverted"

                elif strategy == HUMAN:
                    candidate = human_code
                    op = "Human-in-the-loop: manually supplied code"

                else:
                    continue

                # ── Evaluate (memoized) ───────────────────────────────────────
                steps_count[0] += 1
                fitness, metrics, hit = evaluate_candidate_cached(
                    candidate, problem_type, eval_runs, w1, w2, w3)

                if hit:
                    log("  [{:<14s}] (cached) fitness={:.2f}".format(strategy, fitness))
                elif problem_type == PROBLEM_MATRIX:
                    log("  [{:<14s}] correctness={:.0f}  muls={}  adds={}  matrix_fitness={:.2f}".format(
                        strategy,
                        metrics.get("correctness", 0),
                        metrics.get("mul_count", 0),
                        metrics.get("add_count", 0),
                        fitness))
                else:
                    log("  [{:<14s}] score={:5.1f}  surv={:5.1f}  cost={:5.1f}  fitness={:.2f}".format(
                        strategy,
                        metrics.get("score", 0.0),
                        metrics.get("survival", 0.0),
                        metrics.get("cost", 0.0),
                        fitness))

                if fitness > 0:
                    vdb.add(candidate, fitness, op)

                if fitness > s_best_fit:
                    s_best_fit     = fitness
                    s_best_code    = candidate
                    s_best_op      = op
                    s_best_metrics = metrics

            # Update running best for this strategy
            if s_best_fit > strategy_best_fitness[strategy]:
                strategy_best_fitness[strategy] = s_best_fit
                strategy_best_code[strategy]    = s_best_code

            gen_row[strategy] = {
                "fitness":        s_best_fit,
                "cumulative_best": strategy_best_fitness[strategy],
                # Pacman metrics
                "score":    s_best_metrics.get("score",    0.0),
                "survival": s_best_metrics.get("survival", 0.0),
                "cost":     s_best_metrics.get("cost",     0.0),
                # Matrix metrics (empty string when Pacman)
                "correctness":    s_best_metrics.get("correctness",    ""),
                "op_count":       s_best_metrics.get("op_count",       ""),
                "mul_count":      s_best_metrics.get("mul_count",      ""),
                "add_count":      s_best_metrics.get("add_count",      ""),
                "matrix_fitness": s_best_metrics.get("matrix_fitness", ""),
                "op":   s_best_op,
                "code": s_best_code,
            }
            gen_candidates.append((strategy_best_fitness[strategy],
                                   strategy_best_code[strategy], strategy))

        # ── Top-k selection ───────────────────────────────────────────────────
        gen_candidates.sort(key=lambda x: x[0], reverse=True)
        top_selected = gen_candidates[:top_k]
        if top_selected:
            pool_codes = [c[1] for c in top_selected]
            while len(pool_codes) < pop_size:
                pool_codes.append(pool_codes[0])
            current_pool = pool_codes
        else:
            current_pool = [best_code] * pop_size

        if top_selected:
            gf, gc, gs = top_selected[0]
            log("[Gen {}] Best candidate: {} (fitness={:.2f})".format(gen, gs, gf))
            if gf > best_fitness:
                best_fitness = gf; best_code = gc; best_strategy = gs
                log("[Gen {}] ** New overall best: {:.2f}".format(gen, best_fitness))

        gen_time = round(time.time()-gen_start, 3)
        gen_row["runtime_s"] = gen_time
        gen_row["steps"]     = steps_count[0]
        log("[Gen {}] {}s  steps={}  cache={}".format(
            gen, gen_time, steps_count[0], len(st.session_state.cache)))
        st.session_state.history.append(gen_row)

    st.session_state.best_code     = best_code
    st.session_state.best_fitness  = best_fitness
    st.session_state.best_strategy = best_strategy
    st.session_state.runtime_total = round(time.time()-total_start, 2)
    st.session_state.ran_once      = True
    st.session_state.multi_run_data.append(best_fitness)

    log("")
    log("Done.  Best fitness={:.2f}  via {}".format(best_fitness, best_strategy))
    log("Runtime={}s  Cache={}  VDB={} ({})".format(
        st.session_state.runtime_total,
        len(st.session_state.cache), vdb.size, vdb.backend))

# =============================================================================
# BENCHMARK MODE – each strategy in isolation, same budget + seed
# =============================================================================

def run_benchmark(initial_code, problem, api_key, n_gen, w1, w2, w3,
                  problem_type, seed, eval_runs, model_name, pop_size):
    """
    Runs each strategy in strict isolation with an identical budget and seed.
    Mirrors the logic of run_evolution for a fair, consistent comparison.
    """
    results = {}
    for strategy in [NO_EVOLUTION, RANDOM_MUT, LLM_GUIDED]:
        random.seed(seed); np.random.seed(seed)
        cache   = {}; vdb_tmp = VectorDB()
        curve   = []; best_f  = 0.0
        strategy_best_code = initial_code

        for gen in range(1, n_gen+1):
            gen_cands = []
            n_trials  = 1 if strategy == NO_EVOLUTION else pop_size

            for _trial in range(n_trials):
                parent = strategy_best_code if strategy != NO_EVOLUTION else initial_code

                if strategy == NO_EVOLUTION:
                    if api_key.strip():
                        try:
                            candidate, op = llm_mutate_real(
                                initial_code, problem, 0, api_key,
                                problem_type, [], model_name)
                        except Exception:
                            candidate = initial_code; op = "baseline"
                    else:
                        candidate = initial_code
                        op = "[DEMO FALLBACK] no API key"

                elif strategy == RANDOM_MUT:
                    # FIX: pass problem_type
                    candidate, op = random_mutate(parent, problem_type)
                    ok, _ = validate_candidate(candidate, INPUT_PYTHON)
                    if not ok: candidate = parent; op = "reverted"

                elif strategy == LLM_GUIDED:
                    h   = code_hash(parent)
                    ctx = vdb_tmp.retrieve(parent, k=2)
                    if h in cache and "llm_code" in cache[h]:
                        candidate = cache[h]["llm_code"]; op = "cache_hit"
                    elif api_key.strip():
                        try:
                            candidate, op = llm_mutate_real(
                                parent, problem, gen, api_key, problem_type, ctx, model_name)
                            cache.setdefault(h, {})["llm_code"] = candidate
                        except Exception:
                            # FIX: pass problem_type
                            candidate, op = llm_mutate_demo(parent, gen, ctx, problem_type)
                    else:
                        # FIX: pass problem_type
                        candidate, op = llm_mutate_demo(parent, gen, ctx, problem_type)
                        cache.setdefault(h, {})["llm_code"] = candidate
                    ok, _ = validate_candidate(candidate, INPUT_PYTHON)
                    if not ok: candidate = parent

                # Evaluate with local cache
                h = code_hash(candidate)
                if h in cache and "fitness" in cache[h]:
                    fitness = cache[h]["fitness"]
                else:
                    if problem_type == PROBLEM_MATRIX:
                        fitness, _, _ = compute_fitness_matrix(candidate)
                    else:
                        scr, srv, cst = evaluate_pacman(candidate, runs=eval_runs)
                        fitness = compute_fitness_pacman(scr, srv, cst, w1, w2, w3)
                    cache[h] = {"fitness": fitness}

                if fitness > 0: vdb_tmp.add(candidate, fitness, op)
                gen_cands.append((fitness, candidate))

            gen_cands.sort(reverse=True, key=lambda x: x[0])

            top = gen_cands[:max(1, pop_size // 2)]
            if top and top[0][0] > best_f:
                best_f = top[0][0]
                strategy_best_code = top[0][1]

            curve.append(best_f)

        results[strategy] = curve
    return results

# =============================================================================
# CHARTS
# =============================================================================

_CHART_STYLE = {
    NO_EVOLUTION: ("#8884d8", "^", ":"),
    RANDOM_MUT:   ("#22c55e", "s", "--"),
    LLM_GUIDED:   ("#f97316", "o", "-"),
    HUMAN:        ("#e11d48", "D", "-."),
}


def _active_strategies(history):
    return [s for s in STRATEGIES + [HUMAN] if any(s in h for h in history)]


def make_line_chart(history):
    gens  = [h["gen"] for h in history]
    strats = _active_strategies(history)
    fig, ax = plt.subplots(figsize=(8, 4.2))
    fig.patch.set_facecolor("#0f172a"); ax.set_facecolor("#0f172a")
    ax.tick_params(colors="#94a3b8"); ax.yaxis.label.set_color("#94a3b8")
    ax.xaxis.label.set_color("#94a3b8"); ax.title.set_color("#f1f5f9")
    for spine in ax.spines.values(): spine.set_edgecolor("#334155")
    for s in strats:
        color, marker, ls = _CHART_STYLE.get(s, ("#888","o","-"))
        vals = [h[s].get("cumulative_best", h[s]["fitness"])
                if s in h else float("nan") for h in history]
        ax.plot(gens, vals, color=color, marker=marker, linestyle=ls,
                linewidth=2.5, markersize=7, label=s, zorder=3)
        ax.fill_between(gens, vals, alpha=0.10, color=color)
    ax.set_xlabel("Generation", fontsize=11)
    ax.set_ylabel("Cumulative Best Fitness", fontsize=11)
    ax.set_title("Fitness Improvement Across Generations",
                 fontsize=13, fontweight="bold", color="#f1f5f9")
    ax.set_ylim(0, 112); ax.legend(fontsize=10, loc="lower right",
                                    facecolor="#1e293b", labelcolor="#f1f5f9",
                                    edgecolor="#334155")
    ax.grid(True, alpha=0.18, linestyle="--", color="#334155")
    ax.spines[["top","right"]].set_visible(False)
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    plt.tight_layout()
    return fig


def make_bar_chart(history):
    strats = _active_strategies(history)
    avgs = {}; maxs = {}
    for s in strats:
        vals = [h[s].get("cumulative_best", h[s]["fitness"]) for h in history if s in h]
        avgs[s] = round(float(np.mean(vals)), 2) if vals else 0.0
        maxs[s] = round(float(max(vals)), 2) if vals else 0.0
    labels = list(avgs.keys())
    avg_h  = list(avgs.values())
    max_h  = [maxs[l] for l in labels]
    colors = [_CHART_STYLE.get(l, ("#888",))[0] for l in labels]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(6, 4.0))
    fig.patch.set_facecolor("#0f172a"); ax.set_facecolor("#0f172a")
    ax.tick_params(colors="#94a3b8"); ax.yaxis.label.set_color("#94a3b8")
    ax.title.set_color("#f1f5f9")
    for spine in ax.spines.values(): spine.set_edgecolor("#334155")
    bars1 = ax.bar(x-0.2, avg_h, 0.35, color=colors, edgecolor="#0f172a",
                   linewidth=1.2, alpha=0.85, label="Avg Fitness")
    bars2 = ax.bar(x+0.2, max_h, 0.35, color=colors, edgecolor="#0f172a",
                   linewidth=1.2, alpha=0.45, label="Max Fitness", hatch="///")
    for bar, h in zip(bars1, avg_h):
        ax.text(bar.get_x()+bar.get_width()/2, h+0.8,
                "{:.1f}".format(h), ha="center", va="bottom", fontsize=9, color="#f1f5f9")
    for bar, h in zip(bars2, max_h):
        ax.text(bar.get_x()+bar.get_width()/2, h+0.8,
                "{:.1f}".format(h), ha="center", va="bottom", fontsize=9, color="#f1f5f9")
    short_labels = [l.replace(" (Single-Shot LLM)", "") for l in labels]
    ax.set_xticks(x); ax.set_xticklabels(short_labels, fontsize=9, color="#94a3b8")
    ax.set_ylabel("Fitness", fontsize=10)
    ax.set_title("Strategy Comparison (Avg vs Max)", fontsize=11, fontweight="bold")
    ax.set_ylim(0, 115)
    ax.legend(fontsize=9, facecolor="#1e293b", labelcolor="#f1f5f9", edgecolor="#334155")
    ax.spines[["top","right"]].set_visible(False)
    plt.tight_layout()
    return fig


def make_runtime_chart(history):
    gens  = [h["gen"] for h in history]
    times = [h.get("runtime_s", 0.0) for h in history]
    fig, ax = plt.subplots(figsize=(5, 3.2))
    fig.patch.set_facecolor("#0f172a"); ax.set_facecolor("#0f172a")
    ax.tick_params(colors="#94a3b8"); ax.yaxis.label.set_color("#94a3b8")
    ax.xaxis.label.set_color("#94a3b8"); ax.title.set_color("#f1f5f9")
    for spine in ax.spines.values(): spine.set_edgecolor("#334155")
    ax.bar(gens, times, color="#38bdf8", edgecolor="#0f172a", alpha=0.85)
    ax.set_xlabel("Generation", fontsize=9)
    ax.set_ylabel("Wall-clock (s)", fontsize=9)
    ax.set_title("Runtime per Generation", fontsize=11, fontweight="bold")
    ax.spines[["top","right"]].set_visible(False)
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    plt.tight_layout()
    return fig


def make_multirun_chart(data):
    mean_v = float(np.mean(data)); std_v = float(np.std(data))
    fig, ax = plt.subplots(figsize=(5, 3.2))
    fig.patch.set_facecolor("#fafafa"); ax.set_facecolor("#fafafa")
    ax.plot(range(1, len(data)+1), data, "o-", color="#6366f1", linewidth=2)
    ax.axhline(mean_v, color="#e11d48", linestyle="--",
               label="Mean={:.2f} +/- {:.2f}".format(mean_v, std_v))
    ax.fill_between(range(1,len(data)+1), data, alpha=0.08, color="#6366f1")
    ax.set_xlabel("Run #", fontsize=9); ax.set_ylabel("Best Fitness", fontsize=9)
    ax.set_title("Multi-Run Reproducibility", fontsize=11, fontweight="bold")
    ax.legend(fontsize=9); ax.spines[["top","right"]].set_visible(False)
    plt.tight_layout()
    return fig


def make_benchmark_chart(bench_results: Dict):
    if not bench_results: return None
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    fig.patch.set_facecolor("#fafafa")
    ax = axes[0]; ax.set_facecolor("#fafafa")
    for strat, curve in bench_results.items():
        color, marker, ls = _CHART_STYLE.get(strat, ("#888","o","-"))
        gens = list(range(1, len(curve)+1))
        ax.plot(gens, curve, color=color, marker=marker, linestyle=ls,
                linewidth=2.5, markersize=7, label=strat, zorder=3)
        ax.fill_between(gens, curve, alpha=0.07, color=color)
    ax.set_xlabel("Generation", fontsize=11)
    ax.set_ylabel("Best Fitness (cumulative)", fontsize=11)
    ax.set_title("Isolated Strategy Benchmark", fontsize=12, fontweight="bold")
    ax.set_ylim(0, 110); ax.legend(fontsize=10)
    ax.grid(True, alpha=0.22, linestyle="--")
    ax.spines[["top","right"]].set_visible(False)
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax2 = axes[1]; ax2.set_facecolor("#fafafa")
    labels = list(bench_results.keys())
    finals = [bench_results[s][-1] if bench_results[s] else 0 for s in labels]
    colors = [_CHART_STYLE.get(s, ("#888",))[0] for s in labels]
    bars = ax2.bar(labels, finals, color=colors, edgecolor="white",
                   linewidth=1.5, alpha=0.9)
    for bar, h in zip(bars, finals):
        ax2.text(bar.get_x()+bar.get_width()/2, h+0.8,
                 "{:.1f}".format(h), ha="center", va="bottom",
                 fontsize=12, fontweight="bold")
    ax2.set_ylabel("Final Best Fitness", fontsize=11)
    ax2.set_title("Final Score by Strategy", fontsize=12, fontweight="bold")
    ax2.set_ylim(0, 115); ax2.spines[["top","right"]].set_visible(False)
    plt.tight_layout()
    return fig

# =============================================================================
# CSV EXPORT
# =============================================================================

def build_csv(history):
    strats = _active_strategies(history)
    buf    = io.StringIO()
    wr = csv.writer(buf)
    is_matrix = any(
        h[s].get("matrix_fitness", "") != ""
        for h in history for s in strats if s in h
    )
    if is_matrix:
        wr.writerow(["Generation","Strategy","Matrix_Fitness","Correctness",
                     "Mul_Count","Add_Count","Op_Count","Runtime_s","Steps","Operation"])
        for h in history:
            for s in strats:
                if s in h:
                    d = h[s]
                    wr.writerow([h["gen"], s,
                                 d.get("matrix_fitness", d["fitness"]),
                                 d.get("correctness", ""),
                                 d.get("mul_count", ""),
                                 d.get("add_count", ""),
                                 d.get("op_count", ""),
                                 h.get("runtime_s",""), h.get("steps",""), d["op"]])
    else:
        wr.writerow(["Generation","Strategy","Fitness","Score",
                     "Survival","Cost","Runtime_s","Steps","Operation"])
        for h in history:
            for s in strats:
                if s in h:
                    d = h[s]
                    wr.writerow([h["gen"], s, d["fitness"], d["score"],
                                 d["survival"], d["cost"],
                                 h.get("runtime_s",""), h.get("steps",""), d["op"]])
    return buf.getvalue()

# =============================================================================
# SIDEBAR
# =============================================================================

with st.sidebar:
    st.header("Settings")

    problem_type = st.selectbox(
        "Problem Type", [PROBLEM_PACMAN, PROBLEM_MATRIX],
        help="Pacman agent optimisation or bonus 3x3 matrix multiply task.")

    # ── API Key  (Google Gemini) ──────────────────────────────────────────────
    st.subheader("API Key & Mode")

    # Priority: st.secrets > environment variable > manual entry
    _secret_key = ""
    _key_source  = ""
    try:
        _secret_key = st.secrets.get("GOOGLE_API_KEY", "")
        if _secret_key:
            _key_source = "st.secrets"
    except Exception:
        pass

    if not _secret_key:
        import os as _os
        _secret_key = _os.environ.get("GOOGLE_API_KEY", "")
        if _secret_key:
            _key_source = "environment variable"

    if _secret_key:
        st.success("🔑 Google API key loaded from {} — **Real Gemini mode active**".format(_key_source))
        api_key = _secret_key
        _override = st.text_input(
            "Override API Key (optional)", type="password",
            placeholder="Leave blank to use auto-loaded key",
            help="Overrides the auto-detected key for this session only.")
        if _override.strip():
            api_key = _override.strip()
            st.info("Using manually entered key (override active).")
    else:
        api_key = st.text_input(
            "Google Gemini API Key", type="password",
            placeholder="AIza...",
            help=(
                "Get a FREE key at aistudio.google.com → 'Get API key'.\n"
                "Alternatively set GOOGLE_API_KEY as an environment variable "
                "or add it to .streamlit/secrets.toml as:\n"
                "GOOGLE_API_KEY = \"AIza...\""
            ))
        if api_key.strip():
            if not api_key.strip().startswith("AIza"):
                st.warning("⚠️ Key format looks unusual — Google API keys usually start with 'AIza'. "
                           "Double-check your key.")
            else:
                st.success("🔑 Google API key entered — **Real Gemini mode active**")
        else:
            st.info(
                "ℹ️ **Demo mode active** — No API key detected.\n\n"
                "Get a **free** Google Gemini key at "
                "[aistudio.google.com](https://aistudio.google.com) → **Get API key**.\n\n"
                "Without a key: LLM-Guided uses template cycling and "
                "No-Evolution returns the initial code unchanged."
            )

    if not GEMINI_AVAILABLE:
        st.warning(
            "📦 `google-generativeai` not installed.\n"
            "Run: `pip install google-generativeai` then restart the app."
        )

    # Explicit mode badge
    _real_api = bool(api_key.strip())
    _mode_label = "🟢 Real Gemini API" if _real_api else "🟡 Demo Mode"
    st.caption("**Current LLM mode:** {}".format(_mode_label))
    # ──────────────────────────────────────────────────────────────────────────

    model_name = st.selectbox(
        "Gemini Model", GEMINI_MODELS, index=0,
        help=(
            "gemini-2.0-flash — fastest, free tier, recommended.\n"
            "gemini-1.5-flash — fast and capable.\n"
            "gemini-1.5-pro — most capable, slower."
        ),
        disabled=not api_key.strip())
    if not api_key.strip():
        st.caption("Model selector active only with API key.")

    # VDB status — inline, no subheader
    if FAISS_AVAILABLE:
        st.caption("VDB: FAISS + sentence-transformers ✓")
    else:
        st.caption("VDB: list fallback (`pip install faiss-cpu sentence-transformers`)")

    st.subheader("Problem Description")
    st.caption(
        "Prototype scope: evaluates Python code only. "
        "Other languages can be described here for LLM inspiration, "
        "but only Python is executed and scored.")
    default_prob = DEFAULT_PACMAN_PROBLEM if problem_type == PROBLEM_PACMAN \
                   else DEFAULT_MATRIX_PROBLEM
    problem = st.text_area("problem_desc", value=default_prob,
                            height=80, label_visibility="collapsed")

    st.subheader("Initial Code")

    # Input-type and language selector — clean, no info box clutter
    input_type = st.selectbox(
        "Input type",
        INPUT_TYPES,
        index=0,
        help=(
            "Python → executed directly. Must define the required function.\n"
            "Other Language → LLM translates to Python (API key needed).\n"
            "Pseudocode / Text → LLM implements in Python (API key needed).\n"
            "Without an API key, the default template is used as a fallback."
        )
    )
    # Show language picker only when needed
    selected_language = ""
    if input_type == INPUT_OTHER_LANG:
        selected_language = st.selectbox("Source language", OTHER_LANGUAGES)

    # Contextual one-liner hint — no box, no clutter
    _fn_hint = (
        "def pacman_agent(state):" if problem_type == PROBLEM_PACMAN
        else "def matmul_3x3(A, B):"
    )
    if input_type == INPUT_PYTHON:
        st.caption("Must define `{}` at the top level.".format(_fn_hint))
    elif input_type == INPUT_OTHER_LANG:
        st.caption("Paste {} code — LLM will translate to Python.".format(selected_language or "source"))
    elif input_type == INPUT_PSEUDOCODE:
        st.caption("Describe your algorithm in pseudocode — LLM will implement it.")
    else:
        st.caption("Plain English description — LLM will write the Python code.")

    default_code = DEFAULT_PACMAN_CODE if problem_type == PROBLEM_PACMAN \
                   else DEFAULT_MATRIX_CODE
    initial_code_raw = st.text_area("initial_code", value=default_code,
                                    height=220, label_visibility="collapsed")

    # Fix 4: Normalize non-Python input to Python for internal pipeline use.
    # Do this eagerly so the user sees the result before running evolution.
    if input_type != INPUT_PYTHON and initial_code_raw.strip():
        with st.spinner("Normalizing {} input to Python…".format(input_type)):
            initial_code, norm_note = normalize_to_python(
                initial_code_raw, input_type, problem_type,
                api_key, model_name, selected_language)
        st.info("**Input normalization:** {}".format(norm_note))
        with st.expander("Preview normalized Python code"):
            st.code(initial_code, language="python")
            ok_norm, reason_norm = validate_candidate(initial_code, INPUT_PYTHON)
            if ok_norm:
                st.success("Normalized code is valid Python.")
            else:
                st.error("Normalization issue: {} — fix the input or provide an API key.".format(reason_norm))
    else:
        initial_code = initial_code_raw
        # Fix 8: Show validation error immediately when user edits code
        if initial_code.strip():
            ok_pre, reason_pre = validate_candidate(initial_code, INPUT_PYTHON)
            if not ok_pre:
                st.error("⚠️ Code validation error: {}".format(reason_pre))

    st.subheader("Generations")
    n_gen = st.slider("Generations", 1, 15, 5, label_visibility="collapsed")

    st.subheader("Population Size")
    pop_size = st.slider("Candidates per strategy per gen", 2, 8, 4,
                         help="Larger = stronger evolution; slower per gen.")

    st.subheader("Top-k Selection")
    top_k = st.slider("k – keep top-k per gen", 1, 5, 1,
                       help="k=1: single best; k>1: diverse gene pool")

    if problem_type == PROBLEM_PACMAN:
        st.subheader("Fitness Weights")
        st.caption("Fitness = w1·score + w2·survival - w3·cost  (sum must = 1.0)")
        w1_raw = st.slider("w1 – Score",    0.0, 1.0, 0.5, 0.05)
        w2_raw = st.slider("w2 – Survival", 0.0, 1.0, 0.3, 0.05)
        w3_raw = st.slider("w3 – Cost",     0.0, 1.0, 0.2, 0.05)
        total_w_raw = round(w1_raw + w2_raw + w3_raw, 4)
        if total_w_raw <= 0:
            st.error("All weights are zero — set at least one > 0.")
            w1, w2, w3, total_w = 0.5, 0.3, 0.2, 1.0
        elif abs(total_w_raw - 1.0) > 0.001:
            w1 = round(w1_raw / total_w_raw, 4)
            w2 = round(w2_raw / total_w_raw, 4)
            w3 = round(1.0 - w1 - w2, 4)
            total_w = 1.0
            st.warning(
                "Weights summed to {} — auto-normalised to "
                "w1={}, w2={}, w3={} (sum=1.0).".format(total_w_raw, w1, w2, w3))
        else:
            w1, w2, w3, total_w = w1_raw, w2_raw, w3_raw, 1.0
            st.success("Weights valid — w1={} w2={} w3={} (sum=1.0)".format(w1, w2, w3))
    else:
        # ── Matrix-specific controls ─────────────────────────────────────────
        st.subheader("Matrix Fitness")
        st.caption("Fitness = correctness (0–60) + efficiency (0–40). Naive baseline: 54 ops.")
        mx_target_ops = st.slider(
            "Target op count (reference)",
            min_value=10, max_value=60, value=40, step=1,
            help="Scoring targets ≤ 40 executed ops for full efficiency score."
        )
        mx_emphasise = st.radio(
            "LLM emphasis",
            ["Minimise multiplications", "Minimise additions", "Minimise total ops"],
            index=2,
        )
        _emphasis_map = {
            "Minimise multiplications": " Prioritise reducing the number of scalar multiplications above all else.",
            "Minimise additions":        " Prioritise reducing the number of scalar additions above all else.",
            "Minimise total ops":        " Minimise the combined total of multiplications and additions.",
        }
        _matrix_emphasis_hint = _emphasis_map.get(
            str(mx_emphasise),
            " Minimise the combined total of multiplications and additions."
        )
        if _matrix_emphasis_hint not in str(problem):
            problem = str(problem).rstrip() + _matrix_emphasis_hint
        w1, w2, w3, total_w = 0.5, 0.3, 0.2, 1.0

    st.subheader("Reproducibility")
    use_seed   = st.checkbox("Fix random seed", value=True)
    fixed_seed = int(st.number_input("Seed value", value=42, step=1)) \
                 if use_seed else None

    # Pacman-only: eval_runs is irrelevant for matrix
    if problem_type == PROBLEM_PACMAN:
        eval_runs = st.slider("Eval runs per candidate", 1, 4, 2,
                               help="More runs = more stable Pacman fitness scores, but slower.")
    else:
        eval_runs = 1  # matrix is deterministic — single eval is sufficient
        st.caption("Matrix evaluation is deterministic — single eval used.")

    st.divider()
    run_btn   = st.button("▶ Start Evolution",  type="primary", use_container_width=True)
    bench_btn = st.button("📊 Run Benchmark",    use_container_width=True,
                          help="Runs each of the 3 required strategies in isolation with the same budget")

    # Human-in-Loop status notice
    _hc = st.session_state.get("human_code", "")
    if _hc and _hc.strip():
        st.markdown("""
<div style="background:#2d1b69;border:1px solid #7c3aed;border-radius:8px;
     padding:8px 12px;font-size:12px;color:#e9d5ff;margin-top:6px;">
  ✋ <strong>Human-in-Loop active</strong> — your saved code will compete
  as the <code>Human-in-Loop</code> strategy in the next run.
  Edit or clear it in the <strong>✋ Human-in-Loop</strong> tab.
</div>
""", unsafe_allow_html=True)
    else:
        st.markdown("""
<div style="background:#1e293b;border:1px solid #334155;border-radius:8px;
     padding:8px 12px;font-size:12px;color:#64748b;margin-top:6px;">
  ✋ <strong>Human-in-Loop:</strong> no code saved yet.
  Go to the <strong>✋ Human-in-Loop</strong> tab to add your own candidate.
</div>
""", unsafe_allow_html=True)

    if bench_btn and not api_key.strip():
        st.warning(
            "No API key detected. Benchmark results use the demo/fallback LLM "
            "(template cycling), NOT a real language model. "
            "The No-Evolution curve will be flat (initial code unchanged). "
            "These results are for demonstration only."
        )

    st.divider()
    with st.expander("📚 Key References & About"):
        st.markdown("""
**[1] AlphaEvolve** — Novikov *et al.*, arXiv:2506.13131, 2025.
[doi.org/10.48550/arXiv.2506.13131](https://doi.org/10.48550/arXiv.2506.13131)

**[2] Genetic Algorithms** — S. Tamilselvi, IntechOpen, 2022.
[doi.org/10.5772/intechopen.104198](https://doi.org/10.5772/intechopen.104198)

**[3] EA Overview** — H. Amit, *We Talk Data*, 2025.

**[6] Google Gemini API** — [ai.google.dev](https://ai.google.dev/gemini-api/docs)

📖 Full bibliography in the **References** tab ↑
""")

    if st.button("Clear all results", use_container_width=True):
        for k in ["history","log_lines","multi_run_data","cache","vdb",
                  "ran_once","best_code","best_fitness","runtime_total",
                  "benchmark_results"]:
            st.session_state[k] = [] if k in ("history","log_lines","multi_run_data") \
                                   else ({} if k in ("cache","benchmark_results") \
                                   else (None if k == "vdb" \
                                   else (False if k == "ran_once" \
                                   else ("" if k == "best_code" \
                                   else 0.0))))
        st.rerun()

# =============================================================================
# MAIN AREA
# =============================================================================

st.title("Evolve Agent")
st.caption("CS5381 Analysis of Algorithms  ·  AlphaEvolve-inspired  ·  v7 (Gemini)")
st.divider()

tab_main, tab_bench, tab_human, tab_data, tab_refs = st.tabs([
    "🧬 Evolution",
    "📊 Benchmark",
    "✋ Human-in-Loop",
    "📥 Export Data",
    "📚 References",
])

# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 – EVOLUTION
# ─────────────────────────────────────────────────────────────────────────────
with tab_main:
    st.caption(
        "Three strategies compete each generation — "
        "No Evolution · Random Mutation · LLM-Guided — "
        "top-k survivors seed the next round. "
        "Configure in the sidebar, then click ▶ Start Evolution."
    )

    left_col, right_col = st.columns([1, 1], gap="large")

    with left_col:
        st.subheader("Initial Code")
        st.code(initial_code, language="python")
        st.subheader("Problem Description")
        st.info(problem)

        # Matrix-mode: show fitness formula as a clean caption, not an expander
        if problem_type == PROBLEM_MATRIX:
            st.caption(
                "Matrix fitness = correctness (0–60 pts, all 3 tests) "
                "+ efficiency (0–40 pts, ≤ 40 ops). "
                "Naive baseline: 27 muls + 27 adds = 54 ops."
            )

        if run_btn:
            if problem_type == PROBLEM_PACMAN and abs(total_w-1.0) > 0.001:
                st.error("Fix fitness weights to sum to 1.0 before running.")
            else:
                hc = st.session_state.get("human_code", "")
                with st.spinner("Running evolution... please wait"):
                    run_evolution(initial_code, problem, api_key, n_gen,
                                  w1, w2, w3, top_k, hc, problem_type,
                                  fixed_seed, eval_runs, model_name, pop_size)
                st.success("Evolution complete!")

        if st.session_state.log_lines:
            st.subheader("Execution Log")
            st.markdown(
                '<div class="log-box">{}</div>'.format(
                    "\n".join(st.session_state.log_lines)),
                unsafe_allow_html=True)

    with right_col:
        if st.session_state.ran_once:
            history    = st.session_state.history
            active_ran = _active_strategies(history)
            is_matrix_run = problem_type == PROBLEM_MATRIX

            first_fit = max(
                (history[0][s].get("cumulative_best", history[0][s]["fitness"])
                 for s in active_ran if s in history[0]),
                default=0.0)
            last_fit = st.session_state.best_fitness
            delta    = round(last_fit - first_fit, 2)

            st.markdown("""
<div class="metric-row">
  <div class="metric-card">
    <div class="label">Generations</div>
    <div class="value blue">{gens}</div>
  </div>
  <div class="metric-card">
    <div class="label">Best Fitness</div>
    <div class="value green">{fit:.2f}</div>
  </div>
  <div class="metric-card">
    <div class="label">Improvement</div>
    <div class="value {imp_color}">{delta:+.2f}</div>
  </div>
  <div class="metric-card">
    <div class="label">Cache Hits</div>
    <div class="value purple">{cache}</div>
  </div>
  <div class="metric-card">
    <div class="label">Runtime</div>
    <div class="value amber">{rt}s</div>
  </div>
</div>
""".format(
                gens=len(history),
                fit=last_fit,
                delta=delta,
                imp_color="green" if delta > 0 else "amber",
                cache=st.session_state.accel_cache_hits,
                rt=st.session_state.runtime_total,
            ), unsafe_allow_html=True)

            # ── Per-strategy final fitness summary (replaces noisy progress bars) ──
            final_fits = {
                s: history[-1][s].get("cumulative_best", history[-1][s]["fitness"])
                for s in active_ran if s in history[-1]
            }
            best_val = max(final_fits.values()) if final_fits else 0.0
            # Collect all strategies within 1% of the best (show co-winners)
            CLOSE_THRESHOLD = max(0.5, best_val * 0.01)
            top_strategies = [
                s for s, v in sorted(final_fits.items(), key=lambda x: x[1], reverse=True)
                if best_val - v <= CLOSE_THRESHOLD
            ]

            badge_map = {
                NO_EVOLUTION: ("badge-no-evo", "#818cf8"),
                RANDOM_MUT:   ("badge-random", "#4ade80"),
                LLM_GUIDED:   ("badge-llm",    "#fb923c"),
                HUMAN:        ("badge-human",  "#f472b6"),
            }

            if len(top_strategies) == 1:
                _winner_label = "Best Strategy: <strong>{}</strong> · Fitness: <strong>{:.2f}</strong>".format(
                    top_strategies[0], best_val)
            else:
                _winner_label = "Top strategies (tied within {:.2f}): ".format(CLOSE_THRESHOLD)
                _winner_label += " &nbsp; ".join(
                    '<strong>{}</strong> ({:.2f})'.format(s, final_fits[s])
                    for s in top_strategies
                )
            st.markdown(
                '<div class="winner-box">{}</div>'.format(_winner_label),
                unsafe_allow_html=True)

            # Generation-by-generation delta table in a collapsed expander (clean)
            with st.expander("Generation-by-generation fitness detail", expanded=False):
                _prev_best = {}
                _delta_rows = []
                for h in history:
                    for s in active_ran:
                        if s not in h: continue
                        cb   = h[s].get("cumulative_best", h[s]["fitness"])
                        prev = _prev_best.get(s)
                        delta_str = "—" if prev is None else "{:+.2f}".format(cb - prev)
                        arrow = "" if prev is None else ("▲" if cb - prev > 0.001 else ("▼" if cb - prev < -0.001 else "—"))
                        _prev_best[s] = cb
                        _delta_rows.append({
                            "Gen":            h["gen"],
                            "Strategy":       s.replace(" (Single-Shot LLM)", " (No-Evo)"),
                            "Cumul. Fitness": "{:.2f}".format(cb),
                            "Δ":              delta_str,
                            "Trend":          arrow,
                        })
                if _delta_rows:
                    st.dataframe(_delta_rows, use_container_width=True,
                                 height=min(300, 40 + 35 * len(_delta_rows)),
                                 hide_index=True)

            # Charts
            st.subheader("Fitness Across Generations")
            fig_l = make_line_chart(history)
            st.pyplot(fig_l, use_container_width=True); plt.close(fig_l)

            col_b, col_r = st.columns(2)
            with col_b:
                st.subheader("Strategy Comparison")
                fig_b = make_bar_chart(history)
                st.pyplot(fig_b, use_container_width=True); plt.close(fig_b)
            with col_r:
                st.subheader("Runtime per Gen")
                fig_rt = make_runtime_chart(history)
                st.pyplot(fig_rt, use_container_width=True); plt.close(fig_rt)

            # Results table — different columns per problem type
            st.subheader("Results Table")
            rows = []
            for h in history:
                for s in active_ran:
                    if s in h:
                        d = h[s]
                        if is_matrix_run:
                            rows.append({
                                "Gen":            h["gen"],
                                "Strategy":       s,
                                "Matrix Fitness": d.get("matrix_fitness", d["fitness"]),
                                "Correctness":    d.get("correctness", ""),
                                "Muls":           d.get("mul_count", ""),
                                "Adds":           d.get("add_count", ""),
                                "Op Count":       d.get("op_count", ""),
                                "Runtime_s":      h.get("runtime_s",""),
                                "Steps":          h.get("steps",""),
                                "Operation":      d["op"],
                            })
                        else:
                            rows.append({
                                "Gen":       h["gen"],
                                "Strategy":  s,
                                "Fitness":   d["fitness"],
                                "Score":     d["score"],
                                "Survival":  d["survival"],
                                "Cost":      d["cost"],
                                "Runtime_s": h.get("runtime_s",""),
                                "Steps":     h.get("steps",""),
                                "Operation": d["op"],
                            })
            st.dataframe(rows, use_container_width=True, height=260, hide_index=True)

            # Final best solution
            st.subheader("Final Best Solution")
            st.markdown(
                '<div class="best-box">'
                'Strategy: <strong>{}</strong> &nbsp;|&nbsp; '
                'Fitness: <strong>{:.2f}</strong>'
                '</div>'.format(
                    st.session_state.best_strategy,
                    st.session_state.best_fitness),
                unsafe_allow_html=True)
            st.code(st.session_state.best_code, language="python")

            # Validation
            st.subheader("Validation Test Results")
            best_c = st.session_state.best_code
            if problem_type == PROBLEM_MATRIX:
                ns_val = {}
                try:
                    exec(compile(best_c, "<val>", "exec"), ns_val)
                    fn_val = ns_val.get("matmul_3x3")
                    if fn_val is None:
                        st.error(
                            "❌ Required function `matmul_3x3` not found in the best solution code. "
                            "Your code MUST define: `def matmul_3x3(A, B):` that accepts two 3×3 "
                            "matrices (list-of-lists) and returns the 3×3 product. "
                            "Check spelling (case-sensitive) and ensure it is a top-level function."
                        )
                    else:
                        test_rows = []; all_pass = True
                        for i, (A, B, expected) in enumerate(_MATRIX_TESTS):
                            try:
                                got = fn_val(A, B)
                                passed = (got == expected)
                            except Exception as te:
                                got = str(te); passed = False
                            if not passed: all_pass = False
                            test_rows.append({"Test":"Case {}".format(i+1),
                                              "Pass":"✅ OK" if passed else "❌ FAIL",
                                              "Expected":str(expected),"Got":str(got)})
                        if all_pass: st.success("All {} tests passed".format(len(_MATRIX_TESTS)))
                        else:        st.error("Some tests FAILED")
                        st.dataframe(test_rows, use_container_width=True, hide_index=True)
                        # Show executed op count for best solution
                        fit_v, m_v, a_v = compute_fitness_matrix(best_c)
                        st.info("Best solution: {} muls + {} adds = {} total ops  |  fitness = {:.2f}".format(
                            m_v, a_v, m_v+a_v, fit_v))
                except Exception as exc:
                    st.error("Validation error: {}".format(exc))
            else:
                with st.spinner("Running benchmark validation games..."):
                    val_rows = []
                    for mi in range(len(PACMAN_MAPS)):
                        res = run_pacman_game(best_c, map_idx=mi, seed=99)
                        val_rows.append({"Map":"Map {}".format(mi),
                                         "Score":res["score"],"Steps":res["steps"],
                                         "Ate All":"Yes" if res["ate_all"] else "No",
                                         "Error":res["error"] or "—"})
                all_ok = all(r["Error"]=="—" for r in val_rows)
                if all_ok: st.success("Agent valid on all {} maps".format(len(PACMAN_MAPS)))
                else:      st.warning("Agent had errors on some maps.")
                st.dataframe(val_rows, use_container_width=True, hide_index=True)

            # Fix 9: Operation Explanations — MANDATORY per project spec.
            # Shown as a clearly labelled section, not buried in a collapsed expander.
            st.subheader("Generation & Selection Operation Explanations")
            st.markdown(
                "This section fulfills the **mandatory** project requirement to explain "
                "the operations performed when generating and selecting candidates each generation."
            )
            for h in history:
                gen_n = h["gen"]
                with st.expander(
                    "**Generation {}** — runtime: {}s | eval steps: {}".format(
                        gen_n, h.get("runtime_s","?"), h.get("steps","?")),
                    expanded=(gen_n == len(history))  # auto-expand last generation
                ):
                    # Candidate generation explanations
                    st.markdown("**🧬 Candidate Generation Operations:**")
                    for s in active_ran:
                        if s not in h:
                            continue
                        d = h[s]
                        badge_cls = {
                            NO_EVOLUTION: "badge-no-evo",
                            RANDOM_MUT:   "badge-random",
                            LLM_GUIDED:   "badge-llm",
                            HUMAN:        "badge-human"
                        }.get(s, "")
                        if is_matrix_run:
                            detail = (
                                "matrix_fitness={:.2f}, correctness={}, "
                                "muls={}, adds={}".format(
                                    float(d.get("matrix_fitness", 0) or 0),
                                    d.get("correctness","?"),
                                    d.get("mul_count","?"),
                                    d.get("add_count","?")))
                        else:
                            detail = (
                                "fitness={:.2f}, score={}, survival={}, cost={}".format(
                                    d["fitness"],
                                    d["score"], d["survival"], d["cost"]))
                        st.markdown(
                            '<span class="strategy-badge {}">{}</span> '
                            '{} *({})*'.format(badge_cls, s, d["op"], detail),
                            unsafe_allow_html=True)

                    # Selection policy explanation for this generation
                    st.markdown("**📋 Selection Operation:**")
                    gen_fits = [
                        (s, h[s].get("cumulative_best", h[s]["fitness"]))
                        for s in active_ran if s in h
                    ]
                    gen_fits.sort(key=lambda x: x[1], reverse=True)
                    if gen_fits:
                        best_s, best_f = gen_fits[0]
                        st.markdown(
                            "Top-{k} selection applied: **{winner}** achieved the highest "
                            "cumulative fitness of **{fit:.2f}** and was promoted to seed "
                            "the next generation's population. "
                            "All {n} strategy candidates were ranked and the top-{k} "
                            "survivors were retained.".format(
                                k=top_k,
                                winner=best_s,
                                fit=best_f,
                                n=len(gen_fits))
                        )

            st.divider()

            # Selection policy reference panel
            with st.expander("Selection Policy Reference — Top-k Survivor Selection"):
                st.markdown("""
**How survivors are chosen each generation:**

1. **Evaluate** — every candidate is scored with the fitness function.
2. **Sort** — candidates are ranked by fitness score (highest first).
3. **Top-k selection** — the best **k** candidates survive *(k set via sidebar slider)*.
4. **Pool formation** — next-generation pool seeded from top-k survivors.

**Acceleration:**
- **Cache** — hash(code) → fitness; identical code is never re-evaluated.
- **VDB** — FAISS (or list fallback) stores high-fitness snippets for LLM context.
""")

            # Acceleration details — compact caption
            vdb_now = st.session_state.vdb
            st.caption(
                "⚡ Cache hits: **{}** · Unique evals: **{}** · "
                "Reused: **{}** · Eval time: **{:.2f}s** · "
                "VDB records: **{}** · Cache entries: **{}** · "
                "Backend: `{}`".format(
                    st.session_state.accel_cache_hits,
                    st.session_state.accel_unique_evals,
                    st.session_state.accel_reused_candidates,
                    st.session_state.accel_total_eval_time,
                    vdb_now.size if vdb_now else 0,
                    len(st.session_state.cache),
                    vdb_now.backend if vdb_now else "N/A",
                )
            )

        else:
            st.info("Configure settings in the sidebar and click **Start Evolution** to begin.")

# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 – BENCHMARK
# ─────────────────────────────────────────────────────────────────────────────
with tab_bench:
    st.subheader("Strategy Benchmark")
    st.caption(
        "Runs No Evolution · Random Mutation · LLM-Guided in strict isolation "
        "with the same budget and seed. Human-in-Loop is excluded from this required comparison."
    )

    if bench_btn:
        with st.spinner("Running benchmark... please wait"):
            bench_results = run_benchmark(
                initial_code, problem, api_key, n_gen, w1, w2, w3,
                problem_type,
                fixed_seed if fixed_seed is not None else 42,
                eval_runs, model_name, pop_size)
        st.session_state.benchmark_results = bench_results
        st.success("Benchmark complete!")

    br = st.session_state.get("benchmark_results", {})
    if br:
        if not api_key.strip():
            st.warning(
                "⚠️ Demo-only results — no API key was present when this benchmark "
                "was run. LLM-Guided used template cycling (not a real LLM). "
                "The No-Evolution baseline returns the initial code unchanged. "
                "For a fully compliant comparison, re-run with a valid Google Gemini API key."
            )
        # Filter to exactly the 3 required strategies
        br_3 = {s: v for s, v in br.items() if s in [NO_EVOLUTION, RANDOM_MUT, LLM_GUIDED]}
        final_scores = {s:(v[-1] if v else 0) for s,v in br_3.items()}
        winner = max(final_scores, key=final_scores.get) if final_scores else "—"

        st.markdown(
            '<div class="winner-box">'
            '🏆 Benchmark Winner: <strong>{}</strong>'
            ' &nbsp;|&nbsp; Final Fitness: <strong>{:.2f}</strong>'
            '</div>'.format(winner, final_scores.get(winner, 0)),
            unsafe_allow_html=True)

        # Dedicated benchmark summary table
        st.subheader("📊 Benchmark Summary Table")
        st.caption("Required comparison: No Evolution vs Random Mutation vs LLM-Guided Mutation")
        table_rows = []
        strategy_labels = {
            NO_EVOLUTION: "No Evolution (Single-Shot LLM)",
            RANDOM_MUT:   "Random Mutation",
            LLM_GUIDED:   "LLM-Guided Mutation",
        }
        for s in [NO_EVOLUTION, RANDOM_MUT, LLM_GUIDED]:
            curve = br_3.get(s, [])
            if curve:
                improvement = curve[-1] - curve[0]
                table_rows.append({
                    "Strategy":          strategy_labels.get(s, s),
                    "Gen 1 Fitness":     "{:.2f}".format(curve[0]),
                    "Final Fitness":     "{:.2f}".format(curve[-1]),
                    "Improvement":       "{:+.2f}".format(improvement),
                    "Max Fitness":       "{:.2f}".format(max(curve)),
                    "Mean Fitness":      "{:.2f}".format(float(np.mean(curve))),
                    "Winner":            "🏆 Winner" if s == winner else "",
                })
        st.dataframe(table_rows, use_container_width=True, hide_index=True)

        st.markdown("""
**How to interpret this table:**
- **Gen 1 Fitness** — fitness of the very first candidate each strategy produced.
- **Final Fitness** — best cumulative fitness after all generations.
- **Improvement** — net gain from Gen 1 → Final (positive = evolved successfully).
- **Max Fitness** — peak fitness achieved at any generation.
- **Mean Fitness** — average across all generations (measures consistency).
""")

        st.subheader("📈 Fitness Curves — All Three Strategies")
        fig_bench = make_benchmark_chart(br_3)
        if fig_bench:
            st.pyplot(fig_bench, use_container_width=True); plt.close(fig_bench)

        buf_b = io.BytesIO()
        fig_b2 = make_benchmark_chart(br_3)
        if fig_b2:
            fig_b2.savefig(buf_b, format="png", dpi=150, bbox_inches="tight")
            plt.close(fig_b2)
            st.download_button("Download Benchmark Chart PNG",
                               data=buf_b.getvalue(),
                               file_name="benchmark_comparison.png",
                               mime="image/png")
    else:
        st.info("Click **Run Benchmark** in the sidebar to generate the comparison.")

# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 – HUMAN-IN-LOOP
# ─────────────────────────────────────────────────────────────────────────────
with tab_human:
    st.subheader("✋ Human-in-the-Loop Mutation")
    st.caption(
        "Write or paste improved Python code below. "
        "Save it to include as a candidate in the next evolution run, "
        "competing alongside the automated strategies. "
        "Must define `{}`.".format(
            "def pacman_agent(state):" if problem_type == PROBLEM_PACMAN
            else "def matmul_3x3(A, B):"
        )
    )
    st.markdown(
        "**Allowed imports:** `math random collections heapq itertools functools copy typing`"
    )

    human_input = st.text_area(
        "Your modified code (Python only)",
        value=st.session_state.human_code or initial_code,
        height=340, key="human_editor")

    col_save, col_clear = st.columns(2)
    with col_save:
        if st.button("Save for next run", use_container_width=True):
            ok, reason = validate_candidate(human_input, INPUT_PYTHON)
            if ok:
                st.session_state.human_code = human_input
                st.success("Saved! Re-run evolution to include this candidate.")
            else:
                st.error("Validation failed: {}".format(reason))
    with col_clear:
        if st.button("Clear human code", use_container_width=True):
            st.session_state.human_code = ""
            st.info("Human code cleared.")

    if st.session_state.human_code:
        ok, reason = validate_candidate(st.session_state.human_code, INPUT_PYTHON)
        st.success("Saved code is valid ({})".format(reason))
        st.code(st.session_state.human_code, language="python")

    st.divider()
    if problem_type == PROBLEM_PACMAN:
        st.markdown("### Predefined Template Library (Pacman)")
        template_names = [
            "Flee radius 3 + nearest food",
            "Power-pellet priority",
            "Cluster-aware food agent",
            "Safety-first move scoring",
            "Ghost-penalised food targeting",
        ]
        for i, (tpl, name) in enumerate(zip(_PACMAN_TEMPLATES, template_names)):
            with st.expander("Template {}: {}".format(i+1, name)):
                st.code(tpl, language="python")
    else:
        st.markdown("### Predefined Template Library (Matrix)")
        matrix_names = [
            "Explicit sum unroll",
            "Transpose B + dot rows",
            "Fully unrolled (27 muls)",
            "Row-cached multiply",
        ]
        for i, (tpl, name) in enumerate(zip(_MATRIX_TEMPLATES, matrix_names)):
            with st.expander("Template {}: {}".format(i+1, name)):
                st.code(tpl, language="python")

# ─────────────────────────────────────────────────────────────────────────────
# TAB 4 – EXPORT DATA
# ─────────────────────────────────────────────────────────────────────────────
with tab_data:
    st.subheader("Export Data — Round 2 Submission")

    if not st.session_state.ran_once:
        st.info("Run evolution first, then return here to export.")
    else:
        history    = st.session_state.history
        active_ran = _active_strategies(history)

        st.markdown("### Round 2 Data Checklist")
        for item in [
            "Runtime performance (s per generation)",
            "Steps per generation (candidate evaluation count)",
            "Generation count with fitness scores",
            "True No-Evolution (single-shot LLM) baseline",
            "Comparative plots: No Evolution vs Random vs LLM-Guided",
            "CSV file (student_name_data.csv)",
            "PNG charts: line chart, bar chart, runtime chart",
            "Strong hash-based memoization cache",
            "Vector DB acceleration layer",
            "References section (tab 5)",
        ]:
            st.markdown("OK  {}".format(item))

        st.divider()
        st.download_button(
            "Download student_name_data.csv",
            data=build_csv(history),
            file_name="student_name_data.csv", mime="text/csv")

        import json as _json
        vdb = st.session_state.vdb
        run_summary = {
            "best_fitness":    st.session_state.best_fitness,
            "best_strategy":   st.session_state.best_strategy,
            "total_runtime_s": st.session_state.runtime_total,
            "generations":     len(history),
            "total_steps":     sum(h.get("steps",0) for h in history),
            "vdb_backend":     vdb.backend if vdb else "N/A",
            "vdb_records":     vdb.size if vdb else 0,
            "cache_entries":   len(st.session_state.cache),
            "multi_run_data":  st.session_state.multi_run_data,
            "best_code":       st.session_state.best_code,
        }
        st.download_button(
            "Download Run Summary JSON",
            data=_json.dumps(run_summary, indent=2),
            file_name="run_summary.json", mime="application/json")

        st.divider()
        st.markdown("### Download Charts")
        ca, cb, cc = st.columns(3)
        for col, fig_fn, label, fname in [
            (ca, lambda: make_line_chart(history),
             "Fitness Line Chart", "fitness_line_chart.png"),
            (cb, lambda: make_bar_chart(history),
             "Strategy Bar Chart", "comparative_bar_chart.png"),
            (cc, lambda: make_runtime_chart(history),
             "Runtime Chart",      "runtime_chart.png"),
        ]:
            fig = fig_fn(); buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
            plt.close(fig)
            col.download_button(label, data=buf.getvalue(),
                                file_name=fname, mime="image/png")

        br_d = st.session_state.get("benchmark_results", {})
        if br_d:
            fb = make_benchmark_chart(br_d)
            if fb:
                bb = io.BytesIO()
                fb.savefig(bb, format="png", dpi=150, bbox_inches="tight")
                plt.close(fb)
                st.download_button("Download Benchmark Comparison Chart",
                                   data=bb.getvalue(),
                                   file_name="benchmark_comparison.png",
                                   mime="image/png")

        st.divider()
        st.markdown("### Summary Statistics")
        st.markdown("""
| Metric | Value |
|---|---|
| Total generations | {} |
| Total eval steps | {} |
| Total runtime (s) | {} |
| Best fitness | {:.2f} |
| Best strategy | {} |
| Cache entries | {} |
| VDB records | {} |
| VDB backend | {} |
""".format(
            len(history),
            sum(h.get("steps",0) for h in history),
            st.session_state.runtime_total,
            st.session_state.best_fitness,
            st.session_state.best_strategy,
            len(st.session_state.cache),
            vdb.size if vdb else 0,
            vdb.backend if vdb else "N/A"))

        mrd = st.session_state.multi_run_data
        if len(mrd) >= 2:
            st.markdown("### Multi-Run Analysis")
            st.markdown(
                "Runs: {}  |  Mean: **{:.2f}**  |  "
                "Std: **{:.2f}**  |  Min: {:.2f}  Max: {:.2f}".format(
                    len(mrd), float(np.mean(mrd)), float(np.std(mrd)),
                    min(mrd), max(mrd)))
            fig_mr = make_multirun_chart(mrd); buf_mr = io.BytesIO()
            fig_mr.savefig(buf_mr, format="png", dpi=150, bbox_inches="tight")
            plt.close(fig_mr)
            st.download_button("Download Multi-Run Chart PNG",
                               data=buf_mr.getvalue(),
                               file_name="multirun_chart.png", mime="image/png")

# ─────────────────────────────────────────────────────────────────────────────
# TAB 5 – REFERENCES
# ─────────────────────────────────────────────────────────────────────────────
with tab_refs:
    st.subheader("About & References")
    st.markdown("""
<div style="background:#1e293b;border:1px solid #334155;border-radius:10px;
     padding:18px 22px;margin-bottom:20px;">
  <h3 style="color:#f1f5f9;margin:0 0 10px 0;">About Evolve Agent</h3>
  <p style="color:#cbd5e1;font-size:13px;line-height:1.7;margin:0;">
    <strong>Evolve Agent</strong> is a simplified evolutionary algorithm assistant inspired by
    <strong>AlphaEvolve</strong> [1], built for CS5381 Analysis of Algorithms.
    It uses a Large Language Model (LLM) to generate, mutate, and refine code candidates
    across generations — evaluating each with a domain-specific fitness function and
    selecting the best survivors via top-k selection.<br><br>
    <strong>Strategies implemented:</strong>
    No Evolution (single-shot LLM baseline) ·
    Random Mutation (AST + template) ·
    LLM-Guided Mutation (Gemini API) ·
    Human-in-the-Loop (manual injection)<br><br>
    <strong>Acceleration:</strong>
    Hash-based memoization cache (zero re-evaluation of identical code) +
    FAISS/list Vector DB (top-fitness snippet retrieval for LLM context).<br><br>
    <strong>Use cases:</strong>
    Pacman agent optimisation (score + survival − cost) ·
    3×3 Matrix Multiplication (correctness + operation efficiency) [Bonus]
  </p>
</div>
""", unsafe_allow_html=True)

    st.subheader("References")
    st.markdown("""
**[1] AlphaEvolve**
A. Novikov *et al.*, "AlphaEvolve: A coding agent for scientific and algorithmic discovery,"
*arXiv*:2506.13131, Jun. 2025.
https://doi.org/10.48550/arXiv.2506.13131

**[2] Evolutionary Algorithms**
S. Tamilselvi, "Introduction to Evolutionary Algorithms," in *Genetic Algorithms*,
IntechOpen, 2022.
https://doi.org/10.5772/intechopen.104198

**[3] EA Overview**
H. Amit, "An Overview of Evolutionary Algorithms," *We Talk Data*, 2025.

**[4] Promptbreeder**
C. Fernando *et al.*, "Promptbreeder: Self-Referential Self-Improvement via Prompt Evolution,"
*arXiv*:2309.16797, 2023.
https://arxiv.org/abs/2309.16797

**[5] Berkeley Pacman AI Projects**
UC Berkeley AI Group.
http://ai.berkeley.edu/project_overview.html

**[6] Google Gemini API**
Google DeepMind (2025). *Gemini API Documentation*.
https://ai.google.dev/gemini-api/docs

**[7] Strassen's Algorithm**
V. Strassen, "Gaussian elimination is not optimal," *Numer. Math.*, 13(4), 1969.
https://doi.org/10.1007/BF02165411

**[8] FAISS**
J. Johnson, M. Douze, H. Jegou, "Billion-scale similarity search with GPUs,"
*IEEE Transactions on Big Data*, 2019.
https://arxiv.org/abs/1702.08734

**[9] Sentence-Transformers**
N. Reimers & I. Gurevych, "Sentence-BERT," *arXiv*:1908.10084, 2019.
https://arxiv.org/abs/1908.10084

**[10] Python AST Module**
Python Software Foundation. *ast — Abstract Syntax Trees*.
https://docs.python.org/3/library/ast.html
""")

    st.divider()
    st.markdown("### System Architecture")
    st.code("""
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
|    No-Evolution  |  Random (problem-aware)  |  LLM-Guided  |
|                                                             |
|  Fitness (Pacman):  w1*score + w2*survival - w3*cost       |
|  Fitness (Matrix):  correctness(60) + efficiency(40)       |
|    efficiency via _CountedNum executed op counting         |
|    Naive=54 ops | Strassen~51 | target <= 40 for max       |
|                                                             |
|  Acceleration:                                              |
|    Cache: hash(code) -> fitness  (zero re-evaluation)      |
|    VDB:   FAISS / list  (top-k snippet retrieval)          |
+-------------------------------------------------------------+
""", language="text")