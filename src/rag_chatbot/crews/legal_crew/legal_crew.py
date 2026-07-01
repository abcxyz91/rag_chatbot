from crewai import Agent, Crew, LLM, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent
from typing import List
from crewai_tools import DirectoryReadTool, PDFSearchTool, DOCXSearchTool, CSVSearchTool
from rag_chatbot.settings import domain_rag_config, get_settings

settings = get_settings()
directory_search = DirectoryReadTool(directory=str(settings.domain_knowledge_path('legal')))

legal_db_config = domain_rag_config('legal')

pdf_search = PDFSearchTool(config=legal_db_config)

docx_search = DOCXSearchTool(config=legal_db_config)

csv_search = CSVSearchTool(config=legal_db_config)

@CrewBase
class LegalCrew():
    """LegalCrew crew"""

    agents: List[BaseAgent]
    tasks: List[Task]

    @agent
    def lawyer(self) -> Agent:
        return Agent(
            config=self.agents_config['lawyer'],
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
        """Creates the LegalCrew crew"""

        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
        )
