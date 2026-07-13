from crewai import Agent, Crew, LLM, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent
from typing import List
from rag_chatbot.retrieval import DomainSearchTool
from rag_chatbot.settings import get_settings

settings = get_settings()
knowledge_search = DomainSearchTool(domain="hr")

@CrewBase
class HRCrew():
    """HR crew."""

    agents: List[BaseAgent]
    tasks: List[Task]

    @agent
    def hr_advisor(self) -> Agent:
        return Agent(
            config=self.agents_config['hr_advisor'],
            llm=LLM(
                model=settings.ollama_model(settings.answer_model),
                base_url=settings.ollama_url,
                stream=settings.streaming_enabled,
            ),
            verbose=True,
            tools=[knowledge_search]
        )

    @task
    def answering_task(self) -> Task:
        return Task(
            config=self.tasks_config['answering_task'],
        )

    @crew
    def crew(self) -> Crew:
        """Create the HR crew."""

        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
        )
