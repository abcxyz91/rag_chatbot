from crewai import Agent, Crew, LLM, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent
from typing import List
from crewai_tools import DirectoryReadTool, PDFSearchTool, DOCXSearchTool, CSVSearchTool
from rag_chatbot.settings import domain_rag_config, get_settings

settings = get_settings()
directory_search = DirectoryReadTool(directory=str(settings.domain_knowledge_path('accounting')))

accounting_db_config = domain_rag_config('accounting')

pdf_search = PDFSearchTool(config=accounting_db_config)

docx_search = DOCXSearchTool(config=accounting_db_config)

csv_search = CSVSearchTool(config=accounting_db_config)

@CrewBase
class AccountingCrew():
    """AccountingCrew crew"""

    agents: List[BaseAgent]
    tasks: List[Task]

    @agent
    def accounter(self) -> Agent:
        return Agent(
            config=self.agents_config['accounter'],
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
        """Creates the AccountingCrew crew"""

        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
        )
