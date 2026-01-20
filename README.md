# Virtual Lab

[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/virtual-lab)](https://badge.fury.io/py/virtual-lab)
[![PyPI version](https://badge.fury.io/py/virtual-lab.svg)](https://badge.fury.io/py/virtual-lab)
[![Downloads](https://pepy.tech/badge/virtual-lab)](https://pepy.tech/project/virtual-lab)
[![license](https://img.shields.io/github/license/zou-group/virtual-lab.svg)](https://github.com/zou-group/virtual-lab/blob/main/LICENSE.txt)

![Virtual Lab](images/virtual_lab_architecture.png)

The **Virtual Lab** is an AI-human collaboration framework for scientific research. In the Virtual Lab, a human researcher works with a team of large language model (LLM) **agents** to perform scientific tasks. Interaction between the human and LLM agents occurs via **team meetings**, where agents discuss an agenda together, and **individual meetings**, where a single agent tackles a focused task.

Please see our paper [The Virtual Lab of AI agents designs new SARS-CoV-2 nanobodies](https://www.nature.com/articles/s41586-025-09442-9) for more details on the Virtual Lab and an application to nanobody design for SARS-CoV-2.

If you use the Virtual Lab, please cite our work as follows:

Swanson, K., Wu, W., Bulaong, N.L. et al. *The Virtual Lab of AI agents designs new SARS-CoV-2 nanobodies.* Nature (2025). https://doi.org/10.1038/s41586-025-09442-9

---

# NIH Grant Review Agent

This repository repurposes Virtual Lab to simulate an NIH-style study section for grant preparation. A configurable chair and reviewer team reads a proposal, runs independent and collaborative meetings, and outputs NIH scoring forms plus PI-facing feedback. Below is everything you need to configure, run, and extend that workflow.

## Repository Layout

- `src/virtual_lab/` – Core engine: agent classes (`agent.py`), prompt templates (`prompts.py`), meeting runners (`run_meeting.py`, `run_meeting_original_assistantAPI.py`), and shared utilities/constants.
- `grant_review_process/` – Workflow layer for the grant agent. Contains reviewer personas, prompts, orchestration scripts (`run_study_session.py`, `tightline_ss.py`, notebooks), and helper scripts.
- `data/` – Grant text (`data/toy/grant.txt`) plus reviewer guides, NIH score sheet templates, and any additional reference materials used by prompts.

## Installation

1. **Create an environment (optional)**
   ```bash
   conda create -y -n grant_agent python=3.12
   conda activate grant_agent
   ```
2. **Install the package**
   ```bash
   pip install -e .
   ```
3. **Set your OpenAI API key**
   ```bash
   export OPENAI_API_KEY=<your_key>
   ```

The workflow currently targets `gpt-5-mini` (see `grant_review_process/review_constants.py`). Adjust the `model` variables there if you plan to use other endpoints or providers.

## Configuring a Grant Review

Most customization happens in `grant_review_process/review_constants.py`:

- `grant_filepath` – Path to the grant being evaluated (default `data/toy/grant.txt`).
- `num_iterations`, `num_rounds`, `CLEAR_DIRS` – Control how many meetings run per phase and whether prior transcripts are cleared before new runs.
- Agent definitions (`study_section_chair`, `primary_reviewer`, `secondary_reviewer`, `tertiary_reviewer`, `scientific_critic`) – Update expertise blurbs, goals, or add new reviewers to match your domain.
- Prompt strings (`background_prompt`, `grant_scoring_*`, agendas) – Tune instructions, scoring anchors, and context injection.
- `workflow_phases` / `discussions_phase_to_dir` – Define which phases execute and where transcripts/summaries are saved.

Supporting reference material—reviewer guides, NIH forms, etc.—lives in `data/review_info/`. Replace or extend these files to match different mechanisms or agencies.

## Running the Study Session

From the repository root:

```bash
python grant_review_process/run_study_session.py
```

`run_study_session.py` orchestrates the default pipeline:

1. **Team selection** – The Study Section Chair selects primary/secondary/tertiary reviewers and augments their personas with auto-generated descriptions.
2. **Independent review loops** – Each reviewer conducts meetings using agendas defined in `review_constants.py`, referencing grant text and recent literature via RAG.
3. **Collaborative/chair merges** – Summaries are merged in `chair_merge/`, yielding consolidated strengths, weaknesses, and required revisions.
4. **Final delivery** – A mentor agent fills out NIH scoring forms, writes a formal feedback letter, and saves outputs in `final_output/`.

Need a lighter/faster run while iterating on prompts? Use `grant_review_process/SHORT_run_study_session.py`. For a more customized flow, see `grant_review_process/tightline_ss.py`. Prefer a notebook? `grant_review_process/run_study_session.ipynb` mirrors the script with step-by-step cells, though the `.py` version is usually updated first.

## Outputs

Artifacts live under `grant_review_process/discussions/<grant_name>_<model>/<phase>/`:

- `team_selection/` – Team-selection meetings plus merged reviewer summaries.
- `independent_review/`, `collaboration_review/`, etc. – Phase-specific transcripts (`discussion_*.json`).
- `chair_merge/` – Study Section Chair merged markdown plus exported `.txt` summaries via `write_final_summary`.
- `final_output/` – Mentor/letter agent results, NIH scoring form text, and any additional summary files (e.g., `mentor_out_summary.txt`).

JSON files capture the full conversations; TXT exports capture distilled summaries for quick sharing with collaborators or PIs.

## Extending the Workflow

- **Add phases** by editing `workflow_phases` and updating meeting logic in `run_study_session.py` or custom scripts.
- **Swap models/providers** via the `model` / `model_mini` variables in `review_constants.py`, optionally routing specific agents through different endpoints.
- **Inject new retrieval or data sources** by modifying `background_prompt` and the `contexts=` arguments passed to `run_meeting`.
- **Run experiments** using the notebooks (`ablations.ipynb`, `human_eval.ipynb`, etc.) or by scripting new flows in `grant_review_process/scripts/`.

## OpenAI API Key

The Virtual Lab currently uses GPT-4o or GPT-5-mini from OpenAI. Save your API key as `OPENAI_API_KEY` (e.g., add `export OPENAI_API_KEY=<your_key>` to `.bashrc` or `.bash_profile`).
