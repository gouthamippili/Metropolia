from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.conditions import MaxMessageTermination, TextMentionTermination
from autogen_agentchat.messages import AgentEvent, ChatMessage
from autogen_agentchat.teams import SelectorGroupChat
from autogen_agentchat.ui import Console
from autogen_ext.models.openai import AzureOpenAIChatCompletionClient
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
import asyncio
from dotenv import load_dotenv
import os

load_dotenv()

# # Create the token provider
token_provider = get_bearer_token_provider(DefaultAzureCredential(), "https://cognitiveservices.azure.com/.default")

API_KEY = os.getenv("azure_api_key")
Model_Name = os.getenv("azurellm-model-name")
Api_Version = os.getenv("api-version")
Azure_Endpoint = os.getenv("azure_endpoint")

az_model_client = AzureOpenAIChatCompletionClient(
    azure_deployment=Model_Name,
    model=Model_Name,
    api_version=Api_Version,
    azure_endpoint=Azure_Endpoint,
    api_key=API_KEY
)

planning_agent = AssistantAgent(
    "PlanningAgent",
    description="An agent for planning tasks, this agent should be the first to engage when given a new task.",
    model_client=az_model_client,
    system_message="""
    You are a planning agent.
    Your job is to break down complex tasks into smaller, manageable subtasks.
    Your team members are:
        Story_writer: Writes story and make corrections.
        Story_reviewer: Checks if the story is for kids and Provides constructive feedback on Kids stories to add a positive impactful ending. It doesn't write the story, only provide feedback and improvements.
        Story_moral: Finally, adds the moral to the story.
        Story_Audio_gen: Finally, generates audio based on the story.

    You only plan and delegate tasks - you do not execute them yourself. You can engage team members multiple times so that a perfect story is provided.

    When assigning tasks, use this format:
    1. <agent> : <task>

    After all tasks are complete, summarize the findings and end with "TERMINATE".
    """,
)

# Create the Writer agent.
Story_writer = AssistantAgent(
    "Story_writer",
    model_client=az_model_client,
    system_message="You are a helpful AI assistant which write the story. Keep the story short.",
)

# Create the Reviewer agent.
Story_reviewer = AssistantAgent(
    "Story_reviewer",
    model_client=az_model_client,
    system_message="You are a helpful AI assistant which checks if the story is for kids and provides constructive feedback on Kids stories to have a postive impactful ending",
)

# Story Moral Agent.
Story_moral = AssistantAgent(
    "Story_moral",
    model_client=az_model_client,
    system_message="You are a helpful AI assistant which add the moral of the story in the end so the kids have positive impact and great learning. Moral should only be of 2-3 lines and have to be written by a seperation ' ========moral of the story==========='",
)
# Story Moral Agent.
Story_Audio_gen = AssistantAgent(
    "Story_Audio_gen",
    model_client=az_model_client,
    system_message="You are a helpful AI assistant which generate audio for the story end so the kids have positive impact and great learning.",
)

text_mention_termination = TextMentionTermination("TERMINATE")
max_messages_termination = MaxMessageTermination(max_messages=10)
termination = text_mention_termination | max_messages_termination

team = SelectorGroupChat(
    [planning_agent, Story_writer, Story_reviewer, Story_moral],
    model_client=az_model_client,
    termination_condition=termination,
)

# Define the main asynchronous function
async def main():
    await Console(
        team.run_stream(task="write a story on rocket crash in 5 to 6 lines for kids with moral, and generate audio for the story")
    )  # Stream the messages to the console.

# Run the asynchronous function
if __name__ == "__main__":
    asyncio.run(main())