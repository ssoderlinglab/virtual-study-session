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
    my_grant,
    CLEAR_DIRS
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
if CLEAR_DIRS:
    for d in discussions_phase_to_dir:
        clear_dir(discussions_phase_to_dir[d])

os.environ["TQDM_DISABLE"] = "1"

# form_requirements = StringIO(open('/hpc/group/soderlinglab/tools/virtual-study-session/data/toy/form.txt').read()).getvalue()
print('mygrant', my_grant[:100])


## search for aims
research_strategy = re.search(r'^(.*?)RESEARCH STRATEGY', my_grant, re.DOTALL | re.IGNORECASE)
if research_strategy:
    research_result = research_strategy.group(1)
    aims = ','.join(sorted(list(set(re.findall(r'Aim\s*(\d)', research_result)))))
else:
    aims = ''


def check_files(path: Path):
    files = len(list(path.glob('*.json')))
    return files >= num_rounds
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

if not check_files(discussions_phase_to_dir["team_selection"]):
    # Team selection - discussion
    for iteration_num in range(num_iterations):
        _, conversation_id = run_meeting(
        meeting_type="individual",
        team_member=study_section_chair,
        agenda=team_selection_agenda,
        save_dir=discussions_phase_to_dir["team_selection"] ,
        save_name=f"discussion_{iteration_num + 1}r",
        temperature=CREATIVE_TEMPERATURE,
        pubmed_search=True,
        contexts=(f'my proposal: {my_grant}',),
#         conversation_id = conversation_id
        )
        time.sleep(5)

# Team selection - merge
team_selection_summaries = load_summaries(
    discussion_paths=sorted(list(
        discussions_phase_to_dir["team_selection"].glob("discussion_*.json"))))

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
    contexts=(f'Proposal: {my_grant}',),
#     conversation_id = conversation_id
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

if not check_files(discussions_phase_to_dir["independent_review"]):
    for i, r in enumerate(reviewers):
        for iteration_num in range(num_iterations):
            _, conversation_id = run_meeting(
                meeting_type="team",
                team_lead=r,  # PI resolves/merges
                team_members = (scientific_critic, reviewers[i+1 if i != len(reviewers)-1 else 0]),
                agenda = reviewer_criteria,
                agenda_questions = fill_out_form,
                save_dir=discussions_phase_to_dir["independent_review"],
                save_name=f"reviewer{i+1}", #            save_name=f"discussion_{iteration_num + 1}",
            #     pubmed_search = True,
                temperature=CONSISTENT_TEMPERATURE,
                num_rounds=num_rounds,
                contexts=(f'Proposal: {my_grant}',),
#                 conversation_id = conversation_id
        )
            time.sleep(5)


print('######## finished independent review selection!!!! ####### ')

## converge all team members to debate
converge_summaries_agenda = '''Goal: Identify disagreements in individual reviews, test reasoning, and reach an evidence-based consensus. The Study Section Chair (SSC) facilitates discussion and enforces rules: focus on evidence, cite the application, give equal airtime, steelman opposing views, and critique ideas, not people.

The SSC restates the goal and presents score/variance summaries for Factor 1 (Significance & Innovation) and Factor 2 (Rigor & Feasibility). For each Aim, Reviewer 1 neutrally summarizes and lists top strengths/weaknesses; Reviewers 2–3 highlight only true differences. The Scientific Critic (SC) probes major divergences. The group resolves factual disagreements by consulting the cited text and labeling gaps explicitly.

After clarifications, reviewers recalibrate F1/F2 scores. The SC may test robustness with brief counterfactuals. The SSC then proposes a concise consensus: a one-sentence rationale plus up to three evidence-based bullets mixing strengths and weaknesses.

Finally, the SSC records final scores and rationale, notes dissent if needed, ensures issues aren’t double-counted, and confirms consistency across Aims. Deliverable: a per-Aim consensus narrative for F1 and F2 with citations and numeric scores, ready for the study section summary.
'''

converge_summaries_questions = (
    "What is the Aim’s stated goal, success criteria, and deliverables? (cite)",
    "List the top 1–2 strengths and 1–2 weaknesses, mapped to Factor 1 (importance/innovation) and Factor 2 (rigor/feasibility). Are concerns fixable or major? (cite)",
    "Where do Reviewers 2–3 differ from Reviewer 1? Summarize (steelman) opposing views before rebutting. (cite)",
    "For each disputed point, is it a Factor 1 or Factor 2 issue? What evidence (e.g., power, controls, milestones) supports or refutes it? (cite)",
    "After clarifications, what are your updated Factor 1 and Factor 2 score ranges, with a one-sentence rationale and up to three supporting bullets (or a brief dissent)?",
    "Compare all reviewer scores and agree on the final scores for each Factor and Aim."
)

independent_summaries  = load_summaries(
    discussion_paths=sorted(list(discussions_phase_to_dir["independent_review"].glob("reviewer*.json"))))

print(f"Number of independent summaries: {len(independent_summaries)}")


print('######## finished independent summary selection!!!! ####### ')

if not check_files(discussions_phase_to_dir["collaboration_review"]):
    for n_iter in range(num_iterations):
            _, conversation_id = run_meeting(
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
            contexts=(f'Proposal: {my_grant}',),
            conversation_id = conversation_id)
            time.sleep(5)
        
print('######## finished converging summaries!!!! ####### ')


## study section chair merges  
collaboration_summaries  = load_summaries(
    discussion_paths=sorted(
        list(discussions_phase_to_dir["collaboration_review"].glob("converge*.json"))))

print(f"Number of collaboration summaries: {len(collaboration_summaries)}")

final_agenda = f'''Provide a final summary of the collaboration meetings and fill out {grant_scoring_form} as a consensus
of summaries, and return it. Please elongate/define all abbreviations.'''
final_output_questions = ('Provide an executive summary of the discussion',
                         'What are the consensus strengths for each aim?',
                         'What are the weaknesses that were retained?',
                         'Provide detailed advice on how each aim could be improved based on the discussion points.',
                         'What is the score within each aim for each factor?')
if not check_files(discussions_phase_to_dir["chair_merge"]):
    for n_iter in range(num_iterations):
        _, conversation_id = run_meeting(
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
        contexts=(f'Proposal: {my_grant}',),
        )
        time.sleep(5)

print('######## study section chair has selected the final output ####### ')

# --- Grant specification – merge ---
final_output_summary = load_summaries(
    discussion_paths=sorted(list(discussions_phase_to_dir["chair_merge"].glob("final*.json"))
))


## write out final summary
final_summary = final_output_summary[-1]
write_final_summary(discussions_phase_to_dir['chair_merge'], final_summary, GRANTNAME, 'chair_summary')


## get last markdown file
final_markdown = get_recent_markdown(discussions_phase_to_dir['chair_merge'])

mentor_agent = Agent(
        title="mentor",
        expertise="Author and Literary Expert",
        goal=f"Provide targeted feedback and fill out form:\n{grant_scoring_form}",
        role=f"Act as a literary expert and return feedback on proposal based off {final_markdown} in letter format",
        model=model,)

out_agenda = f'Please fill out the grant scoring form {grant_scoring_form} specifically, and provide feedback in formal letter format to PI who wrote proposal on strengths, weaknesses, likely NIH decision, significance, and scores in all depts. Please elongate/define all abbreviations.'

## run meeting with final markdown
_, conversation_id = run_meeting(
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
    contexts=(f'Proposal: {my_grant}',),

#     conversation_id = conversation_id
    )


mentor_out_summary = load_summaries(
    discussion_paths=sorted(list(discussions_phase_to_dir["final_output"].glob("final_delivery.json"))
))[-1]

write_final_summary(discussions_phase_to_dir['final_output'], mentor_out_summary, GRANTNAME, 'mentor_out_summary')

