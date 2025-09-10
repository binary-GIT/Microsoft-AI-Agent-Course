# agent/agent.py

from langchain.agents import initialize_agent, AgentType
from agent.tools import (
    lookup_dictionary,
    llm_define,
    llm_synonyms,
    llm_antonyms,
    llm_examples,
    wiki_usage,
    llm_event_info,
    llm_etymology,
    llm_pronunciation,
    llm_frequency,
    llm_translate,
)
from agent.llm import llm


# Collect all tools
tools = [
    lookup_dictionary,
    llm_define,
    llm_synonyms,
    llm_antonyms,
    llm_examples,
    wiki_usage,
    llm_event_info,
    llm_etymology,
    llm_pronunciation,
    llm_frequency,
    llm_translate,
]



# Initialize the Agent
dictionary_agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,  # React-style reasoning
    verbose=True,
)

# Run Agent
def run_agent(query: str) -> str:
    """Run the dictionary agent on a given query."""
    return dictionary_agent.invoke({"input": query})["output"]
