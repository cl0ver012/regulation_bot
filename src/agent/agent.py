from typing import List, Optional, Tuple
from pydantic import BaseModel, HttpUrl
from langgraph.graph import StateGraph

from src.models.graph_state import GraphState, Route
from src.agent.nodes import determine_route, data_retrieval_node, answering_node, entry_node, generate_simple_answer

class Agent:

    def __init__(self):
        self.langgraph_agent = self._build_graph()

            
    def _build_graph(self):
        """Builds and compiles the StateGraph for choose_action."""
        workflow = StateGraph(state_schema=GraphState)
        workflow.add_node("entry_node", entry_node)
        workflow.add_node("answering_node", answering_node)
        workflow.add_node("generate_simple_answer", generate_simple_answer)
        workflow.add_conditional_edges("entry_node", determine_route, {
            Route.ASK_AGAIN: "generate_simple_answer",
            Route.SIMPLE: "generate_simple_answer",
            Route.INTERPRET: "answering_node",
            Route.SUMMARY: "answering_node"
        })
        
        workflow.set_entry_point("entry_node")
        workflow.set_finish_point("generate_simple_answer")
        workflow.set_finish_point("answering_node")
        
        workflow = workflow.compile()
        
        return workflow

    
    def answer_question(self, messages):
        initial_state = GraphState(
            query=messages[-1].content,
            chat_history=messages
        )
        
        result = self.langgraph_agent.invoke(initial_state)
        
        return result