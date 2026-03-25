from crewai import Agent, Task, Crew, LLM
from crewai_tools import OCRTool

ocr_llm = LLM(model='ollama/gemma3:27b')

ocr_tool = OCRTool(llm=ocr_llm, image_path_url='./knowledge_base/accounting/receipt.png')

agent = Agent(
    role="OCR Specialist",
    goal="Extract text from images",
    backstory="Vision‑enabled analyst",
    tools=[ocr_tool(result_as_answer=True)],  # Set the OCRTool to return its output as the final answer of the agent
    verbose=True,
)

task = Task(
    description="Extract text from image and return in plain text format",
    expected_output="All detected text in plain text",
    agent=agent,
    markdown=True,
    output_file='./knowledge_base/accounting/receipt_text.md'
)

crew = Crew(agents=[agent], tasks=[task])
result = crew.kickoff()