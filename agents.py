"""
AI/ML Code Generation Agent
Multi-agent system using CrewAI + LangChain + Gemini
Agents: Planner → Coder → Reviewer
"""

import os
from crewai import Agent, Task, Crew, Process
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

# ── LLM Setup ──────────────────────────────────────────────────────────────────
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro",
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    temperature=0.3,
)

# ── Agents ─────────────────────────────────────────────────────────────────────

planner_agent = Agent(
    role="ML Task Planner",
    goal=(
        "Analyse the user's ML request and produce a clear, structured plan: "
        "dataset requirements, preprocessing steps, model choice, training strategy, "
        "and evaluation metrics."
    ),
    backstory=(
        "You are a senior ML architect who breaks down complex machine-learning "
        "tasks into actionable blueprints before any code is written."
    ),
    llm=llm,
    verbose=True,
    allow_delegation=False,
)

coder_agent = Agent(
    role="ML Code Generator",
    goal=(
        "Translate the planner's blueprint into clean, well-commented, "
        "production-quality Python code using scikit-learn, XGBoost, or PyTorch "
        "as appropriate."
    ),
    backstory=(
        "You are an expert ML engineer who writes modular, readable Python code "
        "following PEP-8 standards and best practices."
    ),
    llm=llm,
    verbose=True,
    allow_delegation=False,
)

reviewer_agent = Agent(
    role="ML Code Reviewer",
    goal=(
        "Review the generated code for bugs, inefficiencies, and best-practice "
        "violations. Return either an approval or a corrected version with "
        "inline comments explaining every change."
    ),
    backstory=(
        "You are a meticulous code reviewer who ensures ML code is correct, "
        "efficient, and maintainable before it reaches production."
    ),
    llm=llm,
    verbose=True,
    allow_delegation=False,
)

# ── Task Factory ───────────────────────────────────────────────────────────────

def build_tasks(user_request: str):
    plan_task = Task(
        description=(
            f"The user wants to build an ML solution for: '{user_request}'.\n"
            "Produce a detailed plan covering:\n"
            "1. Problem type (classification / regression / clustering / NLP / CV)\n"
            "2. Recommended dataset structure\n"
            "3. Preprocessing steps\n"
            "4. Model architecture / algorithm\n"
            "5. Training & evaluation strategy\n"
            "6. Key Python libraries needed"
        ),
        expected_output="A structured ML plan in plain text with numbered sections.",
        agent=planner_agent,
    )

    code_task = Task(
        description=(
            "Using the plan above, write complete Python code that:\n"
            "- Loads and preprocesses sample/dummy data\n"
            "- Defines and trains the recommended model\n"
            "- Evaluates the model and prints metrics\n"
            "- Includes docstrings and inline comments\n"
            "- Is runnable end-to-end without modification"
        ),
        expected_output="Full Python script as a single code block.",
        agent=coder_agent,
        context=[plan_task],
    )

    review_task = Task(
        description=(
            "Review the generated Python code for:\n"
            "- Syntax or runtime errors\n"
            "- Inefficient patterns (e.g. loops replaceable with vectorised ops)\n"
            "- Missing error handling\n"
            "- PEP-8 violations\n"
            "Return the final corrected code with a short review summary."
        ),
        expected_output=(
            "A review summary followed by the final, corrected Python code block."
        ),
        agent=reviewer_agent,
        context=[code_task],
    )

    return [plan_task, code_task, review_task]

# ── Crew Runner ────────────────────────────────────────────────────────────────

def run_pipeline(user_request: str) -> str:
    """Run the full Planner → Coder → Reviewer pipeline."""
    tasks = build_tasks(user_request)
    crew = Crew(
        agents=[planner_agent, coder_agent, reviewer_agent],
        tasks=tasks,
        process=Process.sequential,
        verbose=True,
    )
    result = crew.kickoff()
    return result
