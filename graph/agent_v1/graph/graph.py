import os
from typing import Dict, Any

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph
from langgraph.constants import END
from langchain.agents import create_agent

from agent_v1.graph.states import File, Plan, TaskPlan, CoderState
from agent_v1.prompts.prompts import planner_prompt, architect_prompt, coder_system_prompt
from agent_v1.tools.filesystem import read_file, write_file, list_files, get_current_directory, set_project_root

# Environment & LLM Setup
def get_llm() -> ChatOpenAI:
    """
    Centralized LLM factory.
    Makes switching models or providers easy.
    """
    return ChatOpenAI(
        model="gpt-4o-mini-2024-07-18",
        temperature=0.6
    )

def init_environment() -> None:
    """
    Initialize environment variables once.
    Safe to call multiple times.
    """
    os.environ.setdefault("LANGSMITH_TRACING", "true")
    load_dotenv()

# Agent Nodes
def planner_agent(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Converts user prompt into a structured Plan.
    """
    llm = get_llm()
    user_prompt = state["user_prompt"]

    plan = llm.with_structured_output(Plan).invoke(
        planner_prompt(user_prompt)
    )

    if not plan:
        raise ValueError("Planner agent returned empty output")

    return {"plan": plan}


def architect_agent(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Converts Plan into TaskPlan.
    """
    llm = get_llm()
    plan: Plan = state["plan"]

    task_plan = llm.with_structured_output(TaskPlan).invoke(
        architect_prompt(plan)
    )

    if not task_plan:
        raise ValueError("Architect agent returned empty output")

    return { "plan": plan, "task_plan": task_plan}


def coder_agent(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Iterative tool-using coding agent.
    """
    llm = get_llm()

    coder_state: CoderState | None = state.get("coder_state")

    if coder_state is None:
        # project_dir = create_project_root(state["plan"].name)
        project_dir = ""
        coder_state = CoderState(
            task_plan=state["task_plan"],
            project_root=str(project_dir),
            current_step_idx=0
        )

    set_project_root(coder_state.project_root)

    steps = coder_state.task_plan.implementation_steps

    if coder_state.current_step_idx >= len(steps):
        return {
            "coder_state": coder_state,
            "status": "DONE"
        }

    current_task = steps[coder_state.current_step_idx]

    existing_content = read_file.run(current_task.filepath)

    system_prompt = coder_system_prompt()
    user_prompt = (
        f"Task: {current_task.task_description}\n"
        f"File: {current_task.filepath}\n\n"
        f"Existing Content:\n{existing_content}\n\n"
        "Use write_file(path, content) to save your changes."
    )

    tools = [
        read_file,
        write_file,
        list_files,
        get_current_directory,
    ]

    agent = create_agent(
        model=llm,
        tools=tools,
        system_prompt=system_prompt
    )

    agent.invoke(
        {"messages": [{"role": "user", "content": user_prompt}]}
    )

    coder_state.current_step_idx += 1

    return {"coder_state": coder_state}

# Graph Factory
def build_graph():
    """
    Builds and compiles the LangGraph.
    Safe to reuse across API calls.
    """
    graph = StateGraph(dict)

    graph.add_node("planner", planner_agent)
    graph.add_node("architect", architect_agent)
    graph.add_node("coder", coder_agent)

    graph.add_edge("planner", "architect")
    graph.add_edge("architect", "coder")

    graph.add_conditional_edges(
        "coder",
        lambda state: "END" if state.get("status") == "DONE" else "coder",
        {
            "END": END,
            "coder": "coder"
        }
    )

    graph.set_entry_point("planner")

    return graph.compile()

# Public API (FastAPI-friendly)
def run_agent(user_prompt: str) -> Dict[str, Any]:
    """
    Public callable entry point.
    This is what FastAPI should call.
    """
    init_environment()
    agent = build_graph()

    return agent.invoke(
        {"user_prompt": user_prompt}
    )


# Local CLI Test
if __name__ == "__main__":
    result = run_agent(
        "build an 404 error page using internal css and html"
    )

    print("Final State:")
    print(result)
