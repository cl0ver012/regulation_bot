import asyncio
import json
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage
from langchain_openai import ChatOpenAI

from src.models.graph_state import GraphState
from src.models.graph_state import RouteResult, Route
from src.tools.pg_utils import query_postgres_with_pgvector

from config import ModelType, PromptTemplate, get_prompt_template


def entry_node(state:GraphState) -> GraphState:
    return state


def determine_route(state: GraphState) -> Route:
    chat_history = state.chat_history
    messages = []
    for message in chat_history:
        if message.role == "system":
            messages.append(SystemMessage(message.content))
        elif message.role == "assistant":
            messages.append(AIMessage(message.content))
        elif message.role == "user":
            messages.append(HumanMessage(message.content))
    model = ChatOpenAI(
        model=ModelType.gpt4o,
        response_format=RouteResult
    )
    
    route = json.loads(model.invoke(messages).content)['route']

    return route


def data_retrieval_node(state:GraphState) -> GraphState:
    query = state.query
    state.retrieved_result = query_postgres_with_pgvector(query)
    return state


def answering_node(state: GraphState) -> GraphState:
    query = state.query
    state.retrieved_result = query_postgres_with_pgvector(query)
    
    prompt = get_prompt_template(PromptTemplate.GENERATE_ANSWER).format(
        question=query,
        context=str(state.retrieved_result)
    )
    
    chat_history = state.chat_history
    messages = []
    for message in chat_history:
        if message.role == "system":
            messages.append(SystemMessage(message.content))
        elif message.role == "assistant":
            messages.append(AIMessage(message.content))
        elif message.role == "user":
            messages.append(HumanMessage(message.content))

    messages = messages[:-1]
    messages.append(SystemMessage(prompt))

    model = ChatOpenAI(
        model=ModelType.gpt4o
    )

    state.response = model.invoke(messages).content

    return state


def generate_simple_answer(state:GraphState)->GraphState:
    query=state.chat_history[-1]
    prompt = get_prompt_template(PromptTemplate.GENERATE_SIMPLE_ANSWER).format(
        question=str(query),
        route=state.route
    )
    
    chat_history = state.chat_history
    messages = [SystemMessage(prompt)]

    for message in chat_history:
        if message.role == "system":
            messages.append(SystemMessage(message.content))
        elif message.role == "assistant":
            messages.append(AIMessage(message.content))
        elif message.role == "user":
            messages.append(HumanMessage(message.content))

    model = ChatOpenAI(
        model=ModelType.gpt4o
    )

    state.response = model.invoke(messages).content

    return state