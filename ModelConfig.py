MANAGER_MODEL = "gemini-3-pro-preview"
WORKER_MODEL = "gemini-3-pro-preview"
COORDINATOR_MODEL = "gemini-3-pro-preview"


def get_model_config():
    return {
        "manager": MANAGER_MODEL,
        "worker": WORKER_MODEL,
        "coordinator": COORDINATOR_MODEL,
    }


def set_model(agent_type: str, model_name: str):
    global MANAGER_MODEL, WORKER_MODEL, COORDINATOR_MODEL
    
    agent_type = agent_type.lower()
    if agent_type == "manager":
        MANAGER_MODEL = model_name
    elif agent_type == "worker":
        WORKER_MODEL = model_name
    elif agent_type == "coordinator":
        COORDINATOR_MODEL = model_name
    else:
        raise ValueError(f"未知的 Agent 类型: {agent_type}，可选: manager, worker, coordinator")

