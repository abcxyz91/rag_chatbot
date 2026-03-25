from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent
from typing import List

@CrewBase
class GeneralCrew():
    """GeneralCrew crew"""

    agents: List[BaseAgent]
    tasks: List[Task]

    @agent
    def general_assistant(self) -> Agent:
        return Agent(
            config=self.agents_config['general_assistant'],
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
