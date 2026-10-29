# Project Synapse-X: The Self-Improving AI Workshop

![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

**Project Synapse-X** is a demonstration of a new AI paradigm. Instead of using a single, monolithic model, Synapse-X is a dynamic system that **learns and forges new specialist skills on demand** in response to user needs. It is an AI that grows its own capabilities over time.

## The Vision: From a Toolbox to a Workshop

*   **Standard AI:** A toolbox with a fixed set of tools.
*   **Synapse-X:** A complete workshop with a master craftsman (the **Meta-Mind**). You bring it a new problem, and it forges a brand-new, custom tool just for that job, adding it to its toolbox permanently.

### How It Works: The Skill Forging Process
1.  **Problem Definition:** A user describes a new skill and provides a few examples.
2.  **Synthetic Data Generation:** The **Meta-Mind** (a powerful local LLM like Llama 3) uses this information to generate a large, synthetic training dataset.
3.  **Specialist Training:** A new, lightweight `SpecialistOctave` is trained on this data.
4.  **Skill Finalization (HONet Principle):** The trained `Octave` is frozen and saved to the Skill Library, becoming a permanent, forget-free specialist.

---

## How to Run the Demo

### Step 1: Install and Run Ollama (The Meta-Mind)
1.  Download and install **Ollama** from [https://ollama.com/](https://ollama.com/).
2.  Pull a powerful model to act as the Meta-Mind:
    ```bash
    ollama pull llama3
    ```
3.  Keep the Ollama application or server running in the background.

### Step 2: Set Up and Run the Synapse-X Project
1.  Clone this repository and set up the environment:
    ```bash
    git clone https://github.com/your-username/Project-Synapse-X.git
    cd Project-Synapse-X
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```
2.  Install the dependencies: `pip install -r requirements.txt`
3.  Run the main application:
    ```bash
    python run.py
    ```

### Step 3: Live Test - Teach Your AI a New Skill!
Follow these steps in the interactive console.

**1. Teach it "Pirate Speak"**
Copy and paste this entire command into the prompt and press Enter:

## Evidence-Aware Discovery Loop (prototype)

This branch adds a deterministic first step toward research workflows that expose uncertainty instead of presenting unsupported conclusions as facts.

- `discovery_engine/core.py` summarizes supplied support/refutation evidence and surfaces explicit contradictions.
- `suggest_next_test` prioritizes hypotheses with no decisive evidence or balanced evidence.
- `skills/discovery-research/SKILL.md` defines a reusable research workflow compatible with OpenClaw's file-backed skill format.
- `docs/discovery-evidence-loop.md` documents score semantics and acceptance criteria.
- `tests/test_discovery_engine.py` covers core behavior and invalid inputs.

Run the unit tests from the repository root:

```bash
python -m unittest discover -s tests -v
```

**Important:** scores are descriptive evidence summaries, not probabilities or proof. This prototype is not yet an OpenClaw runtime plugin, does not call a language model, and has not established improvement over a baseline. The next milestone is to run the tests, add a structured adapter, and compare the augmented workflow with a baseline on a fixed benchmark.


### Run the discovery prototype

Run the unit tests from the repository root:

```bash
python -m unittest discover -s tests -v
```

Run the deterministic demo:

```bash
python -m examples.discovery_demo
```

The discovery core summarizes explicitly supplied evidence and flags conflicts. Its scores are descriptive counts, not probabilities or proof. The OpenClaw-compatible skill is an instruction pack, not yet a native runtime plugin. No baseline improvement is claimed; runtime integration and controlled comparison remain future work.
