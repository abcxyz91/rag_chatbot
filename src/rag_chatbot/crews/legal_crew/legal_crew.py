from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent
from typing import List
from crewai_tools import DirectoryReadTool, PDFSearchTool, DOCXSearchTool, CSVSearchTool
from chromadb.config import Settings

directory_search = DirectoryReadTool(directory='./knowledge_base/legal')

legal_db_config={
    'embedding_model': {
        'provider': 'ollama',
        'config': {
            'model': 'embeddinggemma:300m'
        }
    },
    'vectordb': {
        'provider': 'chromadb',
        'config': {
            'settings': Settings(
                persist_directory='./chroma_db/legal',
                allow_reset=True,
                is_persistent=True
            )
        }
    }
}

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
