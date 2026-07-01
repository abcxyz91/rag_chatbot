from crewai import Agent, Crew, LLM, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent
from typing import List
from rag_chatbot.settings import get_settings

settings = get_settings()

@CrewBase
class GeneralCrew():
    """GeneralCrew crew"""

    agents: List[BaseAgent]
    tasks: List[Task]

    @agent
    def general_assistant(self) -> Agent:
        return Agent(
            config=self.agents_config['general_assistant'],
            llm=LLM(
                model=settings.ollama_model(settings.answer_model),
                base_url=settings.ollama_url,
                stream=settings.streaming_enabled,
            ),
            verbose=True
        )

    @task
    def answering_task(self) -> Task:
        return Task(
            config=self.tasks_config['answering_task'],
        )

    @crew
    def crew(self) -> Crew:
        """Creates the GeneralCrew crew"""

        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
        )
