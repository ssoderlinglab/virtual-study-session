# NIH Grant Review Agent

An end-to-end, NIH-style study section built with large-language-model agents. The agent stack reads a grant, selects reviewers, runs the simulated discussion rounds, merges chair notes, and delivers an NIH scoring form and cover letter—giving you a full rehearsal of how a proposal will be received.

## Overview

- **Agent engine (`src/virtual_lab/`)** – Houses the reusable building blocks: agent definitions (`agent.py`), prompt templates (`prompts.py`), meeting runners (`run_meeting.py`, `run_meeting_original_assistantAPI.py`), utilities for summarizing discussions, and constants (temperatures, defaults).
- **Grant workflow (`grant_review_process/`)** – Domain-specific layer that configures the Study Section Chair, reviewer personas, NIH scoring instructions, agendas, and orchestration scripts.
- **Data (`data/`)** – Contains the grant text (`data/toy/grant.txt`) plus reviewer guidelines and scoring templates referenced in `review_constants.py`.

The workflow reproduces NIH mechanics:
1. Assemble reviewers aligned to the proposal’s aims.
2. Run independent and collaborative discussion rounds with retrieval-augmented prompts.
3. Merge the Study Section Chair’s summaries into final language.
4. Produce NIH-style scoring sheets and PI-facing feedback.

## Installation

1. **Create an environment (optional).**
   ```bash
   conda create -y -n grant_agent python=3.12
   conda activate grant_agent
   ```
2. **Install the package and extras.**
   ```bash
   pip install -e .
   ```
3. **Set your OpenAI key.**
   ```bash
   export OPENAI_API_KEY=<your_key>
   ```

The workflow currently targets GPT-5-mini (see `review_constants.py`). Update the model names there if you are experimenting with other endpoints or providers.

## Configuring a Grant Review

Most knobs live in `grant_review_process/review_constants.py`:

- `grant_filepath` – Path to the grant text; defaults to `data/toy/grant.txt`.
- `num_iterations`, `num_rounds`, `CLEAR_DIRS` – Control how many meetings are run and whether prior transcripts are cleared.
- Agent definitions (`study_section_chair`, `primary_reviewer`, etc.) – Adjust goals, expertise blurbs, or add new reviewers.
- Prompts (`background_prompt`, `grant_scoring_*`) – Tune agendas, scoring forms, or NIH anchor text.
- `discussions_phase_to_dir` – Directory layout for each phase of the workflow; useful if you want to add new stages or route outputs differently.

Supporting assets such as reviewer guides and score sheet templates are stored in `data/review_info/`. Replace those files (or point to new paths) to reflect the mechanism or funding agency you need.

## Running the Study Session

From the repository root:

```bash
python grant_review_process/run_study_session.py
```

`run_study_session.py` drives the default workflow:

1. **Team selection** – The Study Section Chair interviews the system to pick primary/secondary/tertiary reviewers and updates their personas with the generated descriptions.
2. **Independent review loops** – Each reviewer runs meetings using agendas defined in `review_constants.py`, pulling in grant excerpts and recent literature via RAG.
3. **Chair merges** – Summaries are combined in `chair_merge/` to create a unified view of strengths, weaknesses, and required fixes.
4. **Final delivery** – The mentor/letter agent fills out the NIH scoring form, writes a cover letter, and saves it under `final_output/`.

If you need a lighter iteration (e.g., for debugging prompt edits), use `grant_review_process/SHORT_run_study_session.py`. For experimental variations, `grant_review_process/tightline_ss.py` contains a more customized sequence that you can adapt or mine for snippets.

Prefer an interactive walkthrough? The companion notebook `grant_review_process/run_study_session.ipynb` mirrors the same flow and is handy for step-by-step prompt tweaking, though the Python script is usually updated first.

## Outputs

All transcripts are stored under `grant_review_process/discussions/<grant_name>_<model>/<phase>/`. The helper `src/virtual_lab/utils.py` handles loading and writing summaries:

- `team_selection/` – Raw meetings and merged reviewer selection notes.
- `independent_review/`, `collaboration_review/`, etc. – Phase-specific discussions (`discussion_*.json`).
- `chair_merge/` – The Study Section Chair’s merged markdown summaries plus final text files via `write_final_summary`.
- `final_output/` – Mentor letter, NIH scoring form, and auxiliary summaries (e.g., `mentor_out_summary.txt`).

Every `.json` file is a conversation log; `.txt` exports capture the distilled summaries for quick sharing. Use these artifacts to audit rationales, cite literature suggestions, or drop reviewer insights back into your grant.

## Extending the Workflow

- **Add new phases** by expanding `workflow_phases` and updating `discussions_phase_to_dir`.
- **Swap models/providers** by editing `model` and `model_mini`, or route certain agents to different temperatures in the scripts.
- **Integrate additional retrieval** by augmenting `background_prompt` or hooking new context loaders inside `run_study_session.py`.
- **Run ablations or human evals** using the notebooks in `grant_review_process/` (`ablations.ipynb`, `human_eval.ipynb`, etc.).

## Citation

This project builds on the Virtual Lab framework:

> Swanson, K., Wu, W., Bulaong, N.L. et al. *The Virtual Lab of AI agents designs new SARS-CoV-2 nanobodies.* Nature (2025). https://doi.org/10.1038/s41586-025-09442-9

Please cite it if this workflow assists your research or grant preparation.
