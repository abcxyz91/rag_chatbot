from crewai import Agent, Crew, LLM, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent
from typing import List
from crewai_tools import DirectoryReadTool, PDFSearchTool, DOCXSearchTool, CSVSearchTool
from rag_chatbot.settings import domain_rag_config, get_settings

settings = get_settings()
directory_search = DirectoryReadTool(directory=str(settings.domain_knowledge_path('hr')))

hr_db_config = domain_rag_config('hr')

pdf_search = PDFSearchTool(config=hr_db_config)

docx_search = DOCXSearchTool(config=hr_db_config)

csv_search = CSVSearchTool(config=hr_db_config)

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
            tools=[directory_search, pdf_search, docx_search, csv_search]
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
