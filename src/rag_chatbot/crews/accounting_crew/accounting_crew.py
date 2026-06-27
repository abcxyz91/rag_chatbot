from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent
from typing import List
from crewai_tools import DirectoryReadTool, PDFSearchTool, DOCXSearchTool, CSVSearchTool

directory_search = DirectoryReadTool(directory='./knowledge_base/accounting')

accounting_db_config={
    'embedding_model': {
        'provider': 'ollama',
        'config': {
            'model': 'embeddinggemma:300m'
        }
    },
    'vectordb': {
        'provider': 'chroma',
        'config': {
            'dir': './chroma_db/accounting',
            'allow_reset': True
        }
    }
}

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
