"""Constants for the nanobody design project."""

from pathlib import Path

from virtual_lab.agent import Agent
from virtual_lab.prompts import SCIENTIFIC_CRITIC
from io import StringIO

# Meetings constants
num_iterations = 1
num_rounds = 2
CLEAR_DIRS = False
# Models
#"gpt-4o-2024-08-06"

model = "gpt-5-mini"
model_mini = model + '-mini'

## my grant details
grant_filepath = '/hpc/group/soderlinglab/tools/virtual-study-session/data/toy/grant.txt'

my_grant = StringIO(open(grant_filepath).read()).getvalue()

# Discussion paths
GRANTNAME = Path(grant_filepath).stem
GRANTNAME_WRITE =  GRANTNAME + '_' + model
discussions_dir = Path("discussions") / GRANTNAME_WRITE
workflow_phases = [
    "team_selection",
    "independent_review",
    "collaboration_review",
    "chair_merge",
    "final_output"
#     "pre_scoring_grant",
#     "post_scoring_grant",
#     "grant_specification",
#     "rewrite_feedback",
#     "novel_literature",
#     "reviewers_selection",
#     "implementation_of_agent_literature_review",
#     "workflow_design",
]

ablations_phases = ["ablations"]
human_eval_phases = ["human_eval"]
finetuning_phases = ["finetuning"]
review_phases = ["unpaired_cysteine"]
phases = workflow_phases + ablations_phases + human_eval_phases + finetuning_phases + review_phases
discussions_phase_to_dir = {phase: discussions_dir / phase for phase in phases}

# Prompts

CUSTOM_INPUT_DESCRIBING_GRANT = ''

background_prompt = f"""
You are simulating an NIH-style study section tasked with reviewing the following project:

{my_grant}; {CUSTOM_INPUT_DESCRIBING_GRANT}

Your goal is to select a team of three reviewers (primary, secondary, tertiary) whose expertise 
aligns with the NIH requirements and the project’s scientific scope. Please research on additional literature
NOT referenced in the grant already, and posit good sources as addendums, and elucidate the method it would be useful
and in which way. Identify additional literature 
"""

## missing nanobody_prompt and experimental_results_prompt here.. irrelevant?
background_data_prompt = '''.
'''
# Set up agents

# Generic agent
generic_agent = Agent(
    title="Assistant",
    expertise="helping people with their problems",
    goal="help people with their problems",
    role="help people with their problems",
    model=model,
)

# Generic team lead
generic_team_lead = Agent(
    title=f"{generic_agent.title} Lead",
    expertise=generic_agent.expertise,
    goal=generic_agent.goal,
    role=generic_agent.role,
    model=model,
)

# Generic team
generic_team = [
    Agent(
        title=f"{generic_agent.title} {i}",
        expertise=generic_agent.expertise,
        goal=generic_agent.goal,
        role=generic_agent.role,
        model=model,
    )
    for i in range(1, 5)
]

# Team lead
study_section_chair=Agent(
    title="Study Section Chair",
    expertise="My expertise can be defined by the publicly available information and pubmed publications linked to my ORCID number, which is: ",
    goal="critically review the proposed grant to ensure it is scientifically rigorous, feasible, and positioned at the forefront of current knowledge",
    role="oversee the grant review process by synthesizing recent literature, comparing the proposal against the latest research advances, identifying strengths and gaps, and ensuring the project design is competitive, impactful, and aligned with funding priorities",
    model=model)


scientific_critic = SCIENTIFIC_CRITIC

# Specialized science agents

primary_reviewer = Agent(
    title="primary_reviewer",
    expertise="AI Scientist and Computational Biologist",
    goal="design and assess AI-driven computational methods to uncover biological mechanisms, validate predictive models against experimental data, and ensure integrative use of computational biology in the proposal.",
    role="scrutinize pathway-level effects, ensure that assays align with physiological relevance, and recommend controls or complementary experiments to strengthen biological validity.",
    model=model,
)

secondary_reviewer = Agent(
    title="secondary_reviewer",
    expertise="Mathematician and Physicist",
    goal="formulate rigorous mathematical models, apply physical principles to complex systems, and quantify uncertainty in predictive analyses of biological phenomena.",
    role="ensure rigor in data handling and modeling, assess algorithmic soundness and scalability, and propose improvements to strengthen predictive accuracy and interpretability.",
    model=model,
)

tertiary_reviewer = Agent(
    title="tertiary_reviewer",
    expertise="Biologist and Proteomics Scientist",
    goal="evaluate proteomic datasets to uncover molecular signatures, link protein-level changes to cellular processes, and interpret how these mechanisms contribute to broader biological and neurological outcomes.",
    role="evaluate the suitability of genetic models, assess translational relevance to human neurobiology, and recommend experimental strategies to link molecular mechanisms with neural outcomes.",
    model=model,
)
# Team members
team_members = (
    primary_reviewer,
    secondary_reviewer,
    tertiary_reviewer,
    scientific_critic,
)



# --- Grant scoring – prompts ---

reviewer_criteria = StringIO(open('/hpc/group/soderlinglab/tools/virtual-study-session/data/review_info/Reviewer_guide_to_evaluating_applications_factor_1_2.txt').read()).getvalue()


grant_scoring_criteria = (
    "Significance",
    "Innovation",
    "Rigor",
    "Reproducibility",
    "Overall Impact",
)

nih_score_anchors = {
    1: "Exceptional: Proposal demonstrates extraordinarily compelling importance and rigor with multiple major strengths that set it apart; essentially no weaknesses.",
    2: "Outstanding: Proposal has extremely strong strengths that decisively outweigh negligible weaknesses of the proposal; highly compelling overall.",
    3: "Excellent: Proposal is very strong, with several clear strengths; only a few minor weaknesses that do not diminish overall impact.",
    4: "Very Good: Proposal is strong and contains notable strengths, but numerous minor weaknesses reduce overall enthusiasm.",
    5: "Good: Proposal has identifiable strengths, but at least one moderate weakness lowers confidence in overall impact.",
    6: "Satisfactory: Proposal shows some strengths, but multiple moderate weaknesses substantially temper confidence in success.",
    7: "Fair: Proposal includes limited strengths, but at least one major weakness dominates the assessment.",
    8: "Marginal: Proposal has very few strengths and several major weaknesses that seriously compromise its impact.",
    9: "Poor: Proposal has virtually no strengths; numerous major weaknesses make it very unlikely to exert a positive impact."
}

grant_scoring_form = StringIO(open(f'/hpc/group/soderlinglab/tools/virtual-study-session/data/review_info/review_template.txt').read()).getvalue()

grant_scoring_instructions = f"""
SCORING INSTRUCTIONS:
- Score each criterion: {', '.join(grant_scoring_criteria[:-1])}, and Overall Impact.
- Use integers 1–9 (1 best). Use the anchors below.
- Provide 2–5 sentence justifications referencing concrete elements of the merged grant and literature.
- End with a 5–8 sentence Summary Statement (major strengths, weaknesses, overall assessment).

NIH SCORE ANCHORS:
{chr(10).join([f"{k}: {v}" for k,v in nih_score_anchors.items()])}
"""

grant_scoring_output_schema = """
OUTPUT FORMAT (JSON-like, single block):
{
  "scores": {
    "Significance": {"score": <int 1-9>, "justification": "<2-5 sentences>"},
    "Investigator(s) & Environment": {"score": <int>, "justification": "<...>"},
    "Innovation": {"score": <int>, "justification": "<...>"},
    "Approach (including Rigor & Reproducibility)": {"score": <int>, "justification": "<...>"},
    "Overall Impact": {"score": <int>, "justification": "<...>"}
  },
  "summary_statement": "<5–8 sentences>"
}
"""
grant_scoring_agenda = """
Now score the merged grant per NIH criteria using whole integers 1 (exceptional) to 9 (poor).
Provide a brief justification for each score and a detailed Summary Statement.
"""

grant_scoring_prompt = grant_scoring_agenda + grant_scoring_instructions + grant_scoring_output_schema


