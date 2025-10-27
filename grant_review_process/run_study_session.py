#!/usr/bin/env python
# coding: utf-8

import concurrent.futures
import json
from pathlib import Path

from virtual_lab.constants import CONSISTENT_TEMPERATURE, CREATIVE_TEMPERATURE
from virtual_lab.prompts import (
    CODING_RULES,
    REWRITE_PROMPT,
    create_merge_prompt,
)
import time
from virtual_lab.utils import (load_summaries, 
                               write_final_summary,
                               get_recent_markdown,
                               clear_dir)

from virtual_lab.agent import Agent

from review_constants import (
    background_prompt,
    num_iterations,
    num_rounds,
    discussions_phase_to_dir,
    model,
    model_mini,
    study_section_chair, # study_section_chair
    team_members,
    primary_reviewer,
    secondary_reviewer,
    tertiary_reviewer,
    scientific_critic, 
    grant_scoring_prompt,
    grant_scoring_criteria,
    grant_scoring_form,
    nih_score_anchors,
    GRANTNAME,
    reviewer_criteria,
    my_grant
)

## my imports
import pandas as pd 
import os 
import sys 
import re
from io import StringIO

conversation_id = None
if '5' in model:
    from virtual_lab.run_meeting import run_meeting
else:
    from virtual_lab.run_meeting_original_assistantAPI import run_meeting

### clear directories
for d in discussions_phase_to_dir:
    clear_dir(discussions_phase_to_dir[d])

os.environ["TQDM_DISABLE"] = "1"

# form_requirements = StringIO(open('/hpc/group/soderlinglab/tools/virtual-study-session/data/toy/form.txt').read()).getvalue()
print('mygrant', my_grant[:100])

ORCID_number = '0000-0002-7599-1430'
study_section_chair.expertise += ORCID_number

## search for aims
research_strategy = re.search(r'^(.*?)RESEARCH STRATEGY', my_grant, re.DOTALL | re.IGNORECASE)
if research_strategy:
    research_result = research_strategy.group(1)
    aims = ','.join(sorted(list(set(re.findall(r'Aim\s*(\d)', research_result)))))
else:
    aims = ''

# ## Team selection
# #### technicalities
# - Chair talks to the reviewer 1,2,3; individual reviews return to chair, then looped into reviewer 4 and 5


# Team selection - prompts
team_selection_agenda = f"""You are simulating an NIH-style study section. The goal is to assemble a team of three reviewers 
(primary, secondary, tertiary) who will help refine and strengthen my grant in accordance with 
the NIH requirements. 

Your task is to select reviewers whose expertise aligns with the scientific scope and policy 
requirements of the grant. Each reviewer should be described in terms of their role and domain 
expertise rather than a personal identity. Their skills should collectively ensure a rigorous, 
policy-compliant, and competitive proposal. 

Use Retrieval Augmented Grounding (RAG) to ensure your selections reflect the most current NIH 
guidelines and scientific literature. Reviewers should provide perspectives that are 
well-informed, scientifically critical, and directly relevant to the proposal’s research aims {aims}.

Do not include the Study Section Chair (you). The Chair’s role is to oversee the process, guide 
discussion, and ensure alignment with policy and criteria, but the selected reviewers should be 
the primary contributors of domain-specific evaluation. 

Please acknowledge which individual the ORCID_number is referencing, and how they contribute to the expertise of the overview
process.
Agent(
    title="Study Section Chair",
    expertise="Proposing Study to support research space of proposed Grant, my expertise is in Cell Biology",
    goal="perform research in your area of expertise that maximizes the scientific impact of the proposed project to ensure project feasibility and success",
    role="oversee the grant development process, ensure alignment with recent scientific literature, guide expert discussions, and maintain the overall coherence and competitiveness of the proposal"
)
"""

# Team selection - discussion
for iteration_num in range(num_iterations):
    _, conversation_id = run_meeting(
    meeting_type="individual",
    team_member=study_section_chair,
    agenda=team_selection_agenda,
    save_dir=discussions_phase_to_dir["team_selection"] ,
    save_name=f"discussion_{iteration_num + 1}",
    temperature=CREATIVE_TEMPERATURE,
    pubmed_search=True,
    contexts=(f'my_grant: {my_grant}',),
    conversation_id = conversation_id
    )
    time.sleep(5)

# Team selection - merge
team_selection_summaries = load_summaries(
    discussion_paths=sorted(list(discussions_phase_to_dir["team_selection"].glob("discussion_*.json"))))

print(f"Number of summaries: {len(team_selection_summaries)}")

team_selection_merge_prompt = create_merge_prompt(agenda=team_selection_agenda)

_, conversation_id = run_meeting(
    meeting_type="individual",
    team_member=study_section_chair,
    summaries=team_selection_summaries,
    agenda=team_selection_merge_prompt,
    save_dir=discussions_phase_to_dir["team_selection"],
    save_name="merged",
    temperature=CONSISTENT_TEMPERATURE,
    contexts=(f'Full text: {my_grant}',),
    conversation_id = conversation_id
)


## update team members
team_members_selection_file = Path(discussions_phase_to_dir['team_selection']) / 'merged.json'
team_content = load_summaries([team_members_selection_file])[-1]

# -------- helpers --------
# Capture the "Reviewer:", "Domain expertise:", "Focus across aims:", "Policy/rigor checks:", "Immediate requests to PI:" sections
HEADER_PATTERN = (
    r"Reviewer:(.*?)"
    r"Domain expertise:(.*?)"
    r"Focus across aims:(.*?)"
    r"Policy/rigor checks:(.*?)"
    r"Immediate requests to PI:(.*?)"
)

def extract_sections(block: str):
    m = re.search(HEADER_PATTERN, block, flags=re.DOTALL | re.IGNORECASE)
    if not m:
        return None
    reviewer_txt = m.group(1).strip()
    domain_txt   = m.group(2).strip()
    focus_txt    = m.group(3).strip()
    policy_txt   = m.group(4).strip()
    request_txt  = m.group(5).strip()
    return reviewer_txt, domain_txt, focus_txt, policy_txt, request_txt

def clean_bullets_to_paragraph(txt: str) -> str:
    # remove leading bullets/dashes per line, then collapse all whitespace
    txt = re.sub(r'^\s*[-•]\s*', '', txt, flags=re.MULTILINE)
    txt = re.sub(r'\s+', ' ', txt).strip()
    return txt

# -------- slice blocks for each reviewer --------
# Primary: up to Secondary
primary_block = re.search(r'(Primary Reviewer.*?)(?=Secondary Reviewer)', team_content, flags=re.DOTALL | re.IGNORECASE)
# Secondary: between Secondary and Tertiary
secondary_block = re.search(r'(Secondary Reviewer.*?)(?=Tertiary Reviewer)', team_content, flags=re.DOTALL | re.IGNORECASE)
# Tertiary: to end of file
tertiary_block = re.search(r'(Tertiary Reviewer.*)$', team_content, flags=re.DOTALL | re.IGNORECASE)

# -------- PRIMARY --------
if primary_block:
    s = extract_sections(primary_block.group(1))
    if s:
        reviewer_txt, domain_txt, focus_txt, policy_txt, request_txt = s
        primary_reviewer.expertise += ' ' + clean_bullets_to_paragraph(reviewer_txt)
        primary_reviewer.role      += ' ' + clean_bullets_to_paragraph(domain_txt)
        primary_reviewer.goal      += ' ' + clean_bullets_to_paragraph(focus_txt + ' ' + policy_txt + ' ' + request_txt)

# -------- SECONDARY --------
if secondary_block:
    s = extract_sections(secondary_block.group(1))
    if s:
        reviewer_txt, domain_txt, focus_txt, policy_txt, request_txt = s
        secondary_reviewer.expertise += ' ' + clean_bullets_to_paragraph(reviewer_txt)
        secondary_reviewer.role      += ' ' + clean_bullets_to_paragraph(domain_txt)
        secondary_reviewer.goal      += ' ' + clean_bullets_to_paragraph(focus_txt + ' ' + policy_txt + ' ' + request_txt)
else:
    # optional: keep your previous static fallback
    # secondary_reviewer.expertise += ' Molecular and Cellular Biochemistry Specialist'
    pass

# -------- TERTIARY --------
if tertiary_block:
    s = extract_sections(tertiary_block.group(1))
    if s:
        reviewer_txt, domain_txt, focus_txt, policy_txt, request_txt = s
        tertiary_reviewer.expertise += ' ' + clean_bullets_to_paragraph(reviewer_txt)
        tertiary_reviewer.role      += ' ' + clean_bullets_to_paragraph(domain_txt)
        # If end is unknown, it's fine: this concatenates whatever was present (can be empty)
        tertiary_reviewer.goal      += ' ' + clean_bullets_to_paragraph(focus_txt + ' ' + policy_txt + ' ' + request_txt)
else:
    # optional: keep your previous static fallback
    # tertiary_reviewer.expertise += ' Computational Biology and AI Integration Specialist'
    pass

# ---- debug prints ----
print('Primary Reviewer expertise:', primary_reviewer.expertise)
print('Secondary Reviewer expertise:', secondary_reviewer.expertise)
print('Tertiary Reviewer expertise:', tertiary_reviewer.expertise)
print('######## finished selecting reviewers!!!! ####### ')

## each reviewer independently evaluates.
## Scientific critic should ensure statements are grounded in literature
reviewers = (primary_reviewer, secondary_reviewer, tertiary_reviewer)
fill_out_form= (f'1. Please fill out the bracketed [] areas in the following template for each aim: {grant_scoring_form}',
               f'2. Please provide a score for each factor in each aim!')


for i, r in enumerate(reviewers):
    for iteration_num in range(num_iterations):
        run_meeting(
            meeting_type="team",
            team_lead=r,  # PI resolves/merges
            team_members = (scientific_critic, reviewers[i+1 if i != len(reviewers)-1 else 0]),
            agenda = reviewer_criteria,
            agenda_questions = fill_out_form,
            save_dir=discussions_phase_to_dir["independent_review"],
            save_name=f"reviewer{i+1}_iter{iteration_num+1}", #            save_name=f"discussion_{iteration_num + 1}",
        #     pubmed_search = True,
            temperature=CONSISTENT_TEMPERATURE,
            num_rounds=num_rounds,
            contexts=(f'full text/propposal: : {my_grant}',),
            conversation_id = conversation_id
    )
        time.sleep(5)


print('######## finished independent review selection!!!! ####### ')

## converge all team members to debate 
converge_summaries_agenda = '''Goal: Surface disagreements in the individual reviews, stress-test the reasoning, and converge to a clear, evidence-based consensus.
Study Section Chair (neutral): runs the process, enforces ground rules.
4) Ground rules (state at the start)
Focus on evidence in the strengths and weaknesses. Cite text/data for each strength and weakness.
One mic, equal airtime. Use round-robins; no interruptions in report-outs.
Steelman first. Summarize the opposing view to their satisfaction before rebutting.
Separate people from ideas. Critique arguments, not authors or reviewers.
Meeting details:
Begin with the Study Section Chair (SSC) restating the goal: to reach an evidence-based, written consensus for each Aim on NIH Factor 1 (Importance of the Research—significance and innovation) and Factor 2 (Rigor and Feasibility—soundness and realism of the approach, analyses, milestones, and risk-mitigation). The SSC shares a variance snapshot compiled from the individual reviews (initial F1/F2 scores by each reviewer and their top 2–3 strengths/weaknesses per Aim). Ground rules: cite the application (page/figure/section) for every claim; focus on the text as written (no external prestige or anecdotes); separate fixable concerns from “serious” or “fatal” flaws; keep turns brief and non-interruptive; disclose any lingering uncertainties explicitly.
For each Aim, the SSC designates a speaking order. Reviewer 1 gives a neutral summary of the Aim’s objective and outcomes, then states their top strengths and weaknesses mapped to Factor 1 vs Factor 2 with citations. Reviewers 2 and 3 each surface true deltas—only what they see differently, not a full re-review. The Scientific Critic (SC) then poses ≤3 targeted probes aimed at the largest divergences (e.g., “Is this weakness conceptual novelty (F1) or execution risk (F2)?”, “Where are the quantitative benchmarks and fallback paths?”, “What evidence shows the proposed design is adequately powered/controlled?”). The SSC lists the points of disagreement on a shared screen/notepad in F1 vs F2 buckets.
Resolve facts before opinions. When a claim is challenged, the group opens the cited text and agrees on what the application actually asserts; if information is missing or ambiguous, label that gap explicitly rather than inferring. The SC stress-tests feasibility and rigor: verify controls and comparators, statistics/power, data/analysis reproducibility, and cross-Aim resource dependencies. The SSC prevents double-counting by ensuring each issue lives in exactly one factor unless it has distinct, non-overlapping implications.
Calibrate and converge. After clarifications, the SSC asks each reviewer for a quick revised range for F1 and F2 (e.g., “R1: F1 2–3; F2 4,” etc.) to de-anchor extremes. The SC runs a short counterfactual if needed (“If Aim 2 slips by one quarter, does Aim 1 still meet its benchmark?”) to test robustness. The SSC proposes a straw-man consensus for each factor—a one-sentence rationale plus up to three succinct evidence-backed bullets mixing strengths and weaknesses—and checks for agreement. Language must be neutral and actionable (e.g., “Major Strength: clearly articulated biological rationale with compelling preliminary data in Fig. 2 supporting mechanism X,” “Serious Concern: underpowered primary endpoint; no variance estimates or multiplicity plan specified”).
Decide and document. Once consensus on text and score is reached, the SSC records the final F1 and F2 numeric scores (1–9) for the Aim and the agreed-upon rationale paragraphs. If not, the SSC records a majority score and a one-sentence dissent capturing the minority perspective with a single citation.
Close the Aim with a quick quality check: confirm no issue is counted twice across factors, confirm that “fatal” vs “fixable” labeling is consistent with the rationale, and ensure cross-Aim dependencies are coherent (no double-commit of the same resource). Repeat for all Aims. At the end, the SSC fills out a new grant_scoring_form the per-Aim consensus statements and scores for F1 and F2 so reviewers 1–3 can align their individual forms accordingly (or leave a documented dissent). The final deliverable to be written by the SSC is a clean and detailed, per-Aim consensus narrative for Factor 1 and Factor 2—with page/figure citations and the agreed numeric scores—that accurately reflects the debate and can stand alone in the study section summary.'''

converge_summaries_questions = (
'What exactly is the Aim’s objective, success benchmarks, and deliverables as written? (cite)',
'What are the top 1–2 strengths and 1–2 concerns, mapped to Factor 1 (importance/innovation) vs Factor 2 (rigor/feasibility), and are concerns fixable or serious/fatal? (cite)?',
'Where do Reviewers 2–3 disagree with Reviewer 1 (deltas only), and can each steelman the opposing view before rebutting? (cite)',
'For each contested issue, is it fundamentally Factor 1 or Factor 2, and what concrete evidence (power/variance, controls/comparators, analyses, milestones, fallback paths) supports or undermines it? (cite)',
'After resolving facts, what are your revised score ranges for Factor 1 and Factor 2, and what neutral one-sentence rationale + up to three evidence-backed bullets support the final numeric scores (or a one-line dissent with a single citation)?',
'Debate the scores output from each reviewer and discuss which score to agree upon for each factor/aim correspondence.')

independent_summaries  = load_summaries(
    discussion_paths=sorted(list(discussions_phase_to_dir["independent_review"].glob("reviewer*.json"))))

print(f"Number of independent summaries: {len(independent_summaries)}")


print('######## finished independent summary selection!!!! ####### ')

for n_iter in range(num_iterations):
        run_meeting(
        meeting_type="team",
        team_lead=study_section_chair,  # PI resolves/merges
        team_members = team_members,
        agenda = converge_summaries_agenda,
        agenda_questions = converge_summaries_questions,
        save_dir=discussions_phase_to_dir["collaboration_review"],
        save_name=f"converge_{n_iter+1}",
    #     pubmed_search = True,
        summaries = independent_summaries,
        temperature=CONSISTENT_TEMPERATURE,
        num_rounds=num_rounds,
        conversation_id = conversation_id)
        time.sleep(5)
        
print('######## finished converging summaries!!!! ####### ')


## study section chair merges  
collaboration_summaries  = load_summaries(
    discussion_paths=sorted(list(discussions_phase_to_dir["collaboration_review"].glob("converge*.json"))))

print(f"Number of collaboration summaries: {len(collaboration_summaries)}")

final_agenda = f'''Provide a final summary of the collaboration meetings and fill out {grant_scoring_form} as a consensus
of summaries, and return it. Please elongate/define all abbreviations.'''
final_output_questions = ('Provide an executive summary of the discussion',
                         'What are the consensus strengths for each aim?',
                         'What are the weaknesses that were retained?',
                         'Provide detailed advice on how each aim could be improved based on the discussion points.')

for n_iter in range(num_iterations):
    run_meeting(
    meeting_type="individual",
    team_member=study_section_chair,  # PI resolves/merges
    summaries=collaboration_summaries,
    agenda=final_agenda,
    agenda_questions= final_output_questions,
    save_dir=discussions_phase_to_dir["chair_merge"],
    save_name=f"final_{n_iter+1}",
#     pubmed_search = True,
    temperature=CONSISTENT_TEMPERATURE,
    num_rounds=num_rounds,
    conversation_id = conversation_id
    )
    time.sleep(5)

print('######## study section chair has selected the final output ####### ')

# --- Grant specification – merge ---
final_output_summary = load_summaries(
    discussion_paths=sorted(list(discussions_phase_to_dir["chair_merge"].glob("final*.json"))
))


## write out final summary
final_summary = final_output_summary[-1]
write_final_summary(discussions_phase_to_dir['final_output'], final_summary, GRANTNAME, 'chair_summary')


## get last markdown file
final_markdown = get_recent_markdown(discussions_phase_to_dir['chair_merge'])

mentor_agent = Agent(
        title="mentor",
        expertise="Author and Literary Expert",
        goal=f"Provide targeted feedback and fill out form:\n{grant_scoring_form}",
        role=f"Act as a literary expert and return feedback on proposal based off {final_markdown} in letter format",
        model=model,)

out_agenda = f'Please fill out the grant scoring form {grant_scoring_form} and concisely yet targettedly, provide feedback in formal letter format to PI who wrote proposal on strengths, weaknesses, likely NIH decision, significance, and scores in all depts. Please elongate/define all abbreviations.'

## run meeting with final markdown 
run_meeting(
    meeting_type="individual",
    team_member=mentor_agent,  # PI resolves/merges
    summaries=(final_markdown,),
    agenda=out_agenda,
    agenda_questions= final_output_questions,
    save_dir=discussions_phase_to_dir["final_output"],
    save_name=f"final_delivery",
    #     pubmed_search = True,
    temperature=CONSISTENT_TEMPERATURE,
    num_rounds=num_rounds,
    conversation_id = conversation_id
    )


mentor_out_summary = load_summaries(
    discussion_paths=sorted(list(discussions_phase_to_dir["final_output"].glob("final_delivery.json"))
))[-1]

write_final_summary(discussions_phase_to_dir['final_output'], mentor_out_summary, GRANTNAME, 'mentor_out_summary')
