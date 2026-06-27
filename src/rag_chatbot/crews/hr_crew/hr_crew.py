from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent
from typing import List
from crewai_tools import DirectoryReadTool, PDFSearchTool, DOCXSearchTool, CSVSearchTool

directory_search = DirectoryReadTool(directory='./knowledge_base/hr')

hr_db_config={
    'embedding_model': {
        'provider': 'ollama',
        'config': {
            'model': 'embeddinggemma:300m'
        }
    },
    'vectordb': {
        'provider': 'chroma',
        'config': {
            'dir': './chroma_db/hr',
            'allow_reset': True
        }
    }
}

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
