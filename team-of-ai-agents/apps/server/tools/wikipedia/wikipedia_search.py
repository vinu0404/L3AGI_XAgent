from typing import Optional, Type

from langchain.callbacks.manager import CallbackManagerForToolRun
from langchain_community.utilities import WikipediaAPIWrapper
from pydantic import BaseModel, Field

from tools.base import BaseTool


class WikipediaSearchSchema(BaseModel):
    query: str = Field(
        ...,
        description="The search query for Wikipedia search.",
    )


class WikipediaSearchTool(BaseTool):
    """Tool for the Wikipedia API."""

    name = "Wikipedia Search"

    slug = "wikipediaSearch"

    description = (
        "A wrapper around Wikipedia. "
        "Useful for when you need to answer general questions about "
        "people, places, companies, facts, historical events, or other subjects. "
        "Input should be a search query."
    )

    args_schema: Type[WikipediaSearchSchema] = WikipediaSearchSchema

    tool_id = "eb161647-b858-4863-801f-ba7d2e380601"

    def _run(
        self, query: str, run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Search Wikipedia and return the results."""
        wikipedia = WikipediaAPIWrapper()
        return wikipedia.run(query)
