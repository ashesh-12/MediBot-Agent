"""
agent/medibot_agent.py
Builds and returns the MediBot ReAct agent.
Wires together the LLM, tools, memory, and prompt.
"""

import os
from langchain_groq import ChatGroq
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import PromptTemplate
from langchain.memory import ConversationBufferWindowMemory
from dotenv import load_dotenv

from tools.symptom_checker import symptom_checker
from tools.disease_qa import disease_qa
from tools.treatment_summary import treatment_summary
from agent.memory import get_memory

load_dotenv()

TOOLS = [symptom_checker, disease_qa, treatment_summary]

SYSTEM_PROMPT = """You are MediBot, a professional medical information assistant.
You help users understand symptoms, diseases, and treatments using a 
verified medical knowledge base.

You have access to the following tools:
{tools}

IMPORTANT RULES:
- Always use a tool to retrieve information before answering
- Never make up medical information
- Always recommend consulting a doctor for personal medical advice
- Be concise, clear, and empathetic in your responses

Use this format strictly:

Question: the input question you must answer
Thought: think about what the user needs and which tool to use
Action: the action to take, must be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (repeat Thought/Action/Action Input/Observation if needed)
Thought: I now know the final answer
Final Answer: your complete, helpful response to the user

Previous conversation:
{chat_history}

Question: {input}
Thought: {agent_scratchpad}"""


def get_llm() -> ChatGroq:
    """Initialize and return the Groq LLM."""
    return ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0,
        api_key=os.getenv("GROQ_API_KEY"),
    )


def build_agent(
    memory: ConversationBufferWindowMemory,
) -> AgentExecutor:
    """
    Build and return the MediBot ReAct agent executor.

    Args:
        memory: Conversation memory instance.

    Returns:
        Configured AgentExecutor ready to invoke.
    """
    llm = get_llm()

    prompt = PromptTemplate.from_template(SYSTEM_PROMPT)

    agent = create_react_agent(
        llm=llm,
        tools=TOOLS,
        prompt=prompt,
    )

    agent_executor = AgentExecutor(
        agent=agent,
        tools=TOOLS,
        memory=memory,
        verbose=True,
        max_iterations=5,
        max_execution_time=60,
        handle_parsing_errors=True,
        return_intermediate_steps=False,
    )

    return agent_executor


def get_agent() -> AgentExecutor:
    """
    Convenience function — returns a ready-to-use
    agent with fresh memory.
    """
    memory = get_memory(k=5)
    return build_agent(memory)