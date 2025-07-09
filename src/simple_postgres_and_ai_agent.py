# %%
import os
from azure.ai.projects import AIProjectClient
from azure.identity import DefaultAzureCredential
from azure.ai.agents.models import FunctionTool, ToolSet
from datetime import datetime
from legal_agent_tools import user_functions # user functions which can be found in a legal_agent_tools.py file.
from dotenv import load_dotenv
# Load environment variables
load_dotenv("../.env")


# %%

# Create an Azure AI Client from a project endpoint, copied from your Azure AI Foundry project.
# Customers need to set the environment variables
project_client = AIProjectClient(
    credential=DefaultAzureCredential(),
    endpoint=os.getenv("PROJECT_ENDPOINT"),
)

functions = FunctionTool(user_functions)
toolset = ToolSet()
toolset.add(functions)
agents_client = project_client.agents
agents_client.enable_auto_function_calls(toolset)
print("Initialized agents client and toolset with user functions.")


# %%

agent = None

try:
    agent = agents_client.get_agent("asst_mMiZtMQ1euvQtQ7hh56dMT96")
    print(f"Reuse agent from AGENT_ID: {agent.id}")
except Exception:
    print("There is no agent with ID asst_mMiZtMQ1euvQtQ7hh56dMT96.")

if agent is None:
    agent = agents_client.create_agent(
        model= os.environ["MODEL_DEPLOYMENT_NAME"], 
        name=f"legal-cases-agent-{datetime.now().strftime('%Y%m%d%H%M')}",
        description="Legal Cases Agent",
        instructions=f"""
        You are a helpful legal assistant that can retrieve information about legal cases. 
        The current date is {datetime.now().strftime('%Y-%m-%d')}.
        """,
        toolset=toolset
    )
    print(f"Created agent, ID: {agent.id}")


# %%
# Create thread for communication
thread = agents_client.threads.create()
print(f"Created thread, ID: {thread.id}")

# Create message to thread
message = agents_client.messages.create(
    thread_id=thread.id,
    role="user",
    content="Water leaking into the apartment from the floor above, What are the prominent legal precedents in Washington on this problem in the last 10 years?"
)
print(f"Created message, ID: {message.id}")


# %%

# Create and process agent run in thread with tools
run = agents_client.runs.create_and_process(thread_id=thread.id, agent_id=agent.id)
print(f"Run finished with status: {run.status}")

if run.status == "failed":
    print(f"Run failed: {run.last_error}")

# Fetch and log all messages
messages = agents_client.messages.list(thread_id=thread.id)
for message in messages:
    print(f"Role: {message.role}, Content: {message.content}")


# %%
#Delete the agent when done
# project_client.agents.delete_agent(agent.id)
# print("Deleted agent")
