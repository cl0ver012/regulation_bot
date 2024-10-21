from enum import Enum
from os.path import join
import json
from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = ""
PROMPTS = join(PROJECT_ROOT, "prompt_template")


def load_env():
    load_dotenv(join(PROJECT_ROOT, ".env"))


class ModelType(str, Enum):
    gpt4o = "gpt-4o"
    gpt4o_mini = "gpt-4o-mini"
    embedding = "text-embedding-3-large"


class PromptTemplate(Enum):
    GENERATE_ANSWER = "generate_answer.txt"
    GENERATE_SIMPLE_ANSWER = "generate_simple_answer.txt"

def get_prompt_template(prompt_template: PromptTemplate):
    with open(join(PROMPTS, prompt_template.value), "rt") as f:
        return f.read()
