from typing import Optional, List
from pydantic import HttpUrl, BaseModel
from datetime import datetime
from enum import Enum
from typing import List
from pydantic import BaseModel, Field


class Route(str, Enum):
    SUMMARY = "Summary"
    INTERPRET = "Interpret"
    SIMPLE = "Simple"
    ASK_AGAIN = "Ask_again"
    

class RouteResult(BaseModel):
    """Represents the structured output of the router
    There are only four Routes: ['summary', 'interpret', 'simple','ask_again']
    Options Guide:
    SUMMARY: will be a summary of a legislation chapter or article. Uses when user ask about a specific part of Legislation, including respective paragraphs/articles.
    INTERPRET: Will be a Interpretation of legistlation. Uses when user need to understand how the legislation work about a specific topic.
    SIMPLE: Will be a simple Q&A about a question. Uses when user have a open and simple question about something.
    ASK_AGAIN: If the user question DO NOT relation with legislation, articles, tax, etc. Ask him to a new question.
    """
    route: Route = Field(..., description="This is the route value")


class ChatMessage(BaseModel):
    """Represents a single message in the conversation."""
    role: str = Field(..., description="The role of the message sender (e.g., 'user', 'assistant')")
    content: str = Field(..., description="The content of the message")


class GraphState(BaseModel):
    query: Optional[str] = None
    chat_history: List[ChatMessage] = Field(default_factory=list, description="A list of messages in the conversation")
    retrieved_result: Optional[List[str]] = None
    route: Optional[Route] = None
    response: Optional[str] = None    
