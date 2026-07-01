#!/usr/bin/env python
from pydantic import BaseModel
from typing import List, Dict
import json
import asyncio

from crewai.flow import Flow, listen, start, router
from crewai import LLM
from rag_chatbot.crews.accounting_crew.accounting_crew import AccountingCrew
from rag_chatbot.crews.general_crew.general_crew import GeneralCrew
from rag_chatbot.crews.hr_crew.hr_crew import HRCrew
from rag_chatbot.crews.legal_crew.legal_crew import LegalCrew

from rag_chatbot.settings import get_settings
from rag_chatbot.routing import QueryClassifier, ROUTER_SYSTEM_PROMPT

settings = get_settings()

'''
Uncomment this if you want to persist the state of the flow (including conversation history) across sessions using CrewAI's built-in persistence
'''
#@persist  

'''
Define the state model - Used Pydantic `BaseModel` for structured state
CrewAI can even maintain conversation history between sessions by @persist decorator
In this example, to maintain coversation history within session, need to do it manually
'''
class UserMessageState(BaseModel):
    user_query: str = ''
    query_type: str = ''  # For routing purposes, e.g., 'hr', 'legal'...
    response: str = ''
    conversation_history: List[Dict[str, str]] = []  # Save conversation history

'''
Define query classifier model - Used for classifying user queries into types for routing to appropriate crew
Field is used to provide extra metadata (like description) and to declare that query_type is a required field
'''
# Define LLM model
router_llm = LLM(
    model=settings.ollama_model(settings.router_model),
    base_url=settings.ollama_url,
    temperature=0,  # Fully deterministic — just classification
    response_format=QueryClassifier,
    timeout=60
)

answer_llm = LLM(
    model=settings.ollama_model(settings.answer_model),
    base_url=settings.ollama_url,
    temperature=0.3,  # Some creativity allowed in answers
    stream=settings.streaming_enabled,
    timeout=300
)


class RagChatbotFlow(Flow[UserMessageState]):

    @start()
    async def classify_query(self):
        # If there is no user query, exit flow
        if not self.state.user_query:
            print('No user query provided')
            return 'exit'
        else:
            print('Classifying user query...')
        
        # Classify the user query using LLM direct call
        try: 
            result = await router_llm.acall([
                {
                    'role': 'system',
                    'content': ROUTER_SYSTEM_PROMPT
                },
                {
                    'role': 'user',
                    'content': self.state.user_query
                }
            ])

            # Parse the LLM response
            self.state.query_type = result.query_type  # Access the query_type field from the LLM response (Pydantic object)
            print(f'Query classified as: {self.state.query_type}')
        except Exception as e:
            print(f"Error during LLM call for query classification: {e}")
            self.state.query_type = 'general'  # Default to general if LLM call fails
        return self.state.query_type

    @router(classify_query)  # ← listens to classify_query's output ('accounting', 'hr', 'legal' or 'general')
    def route_query(self, query_type: str):
        print(f"Routing query to {query_type} crew...")
        if query_type == 'accounting':
            return 'accounting'
        elif query_type == 'hr':
            return 'hr'
        elif query_type == 'legal':
            return 'legal'
        else:
            return 'general'
        
    async def _run(self, crew, query_type):
        print(f'Generating {query_type} response...')

        try:
            response = await(crew.crew().kickoff_async(inputs={
                'user_query': self.state.user_query,
                'conversation_history': json.dumps(self.state.conversation_history, indent=2)  # Convert to JSON string
            }))
            print(f'{query_type.capitalize()} crew response: \n\n{response.raw}')
            self.state.response = response.raw
        except Exception as e:
            print(f"Error during crew kickoff: {e}")
            self.state.response = 'Sorry, I encountered an error while generating the response.'

        # Update conversation history (use extend to add a list of dict into conversation history list)
        self.state.conversation_history.extend([
            {'role': 'user', 'content': self.state.user_query},
            {'role': 'assistant', 'content': self.state.response}
        ])

        # Trim conversation history to last 10 exchanges (20 messages) to prevent it from growing indefinitely
        if len(self.state.conversation_history) > 20:
            self.state.conversation_history = self.state.conversation_history[-20:]

    @listen('accounting')
    async def accounting_response(self):
        return await self._run(AccountingCrew(), 'accounting')
  
    @listen('hr')
    async def hr_response(self):
        return await self._run(HRCrew(), 'hr')
    
    @listen('legal')
    async def legal_response(self):
        return await self._run(LegalCrew(), 'legal')
    
    @listen('general')
    async def general_response(self):
        return await self._run(GeneralCrew(), 'general')


def kickoff():
    settings.validate_startup()
    rag_chatbot_flow = RagChatbotFlow()
    while True:
        user_input = input('\nEnter your question (or type "exit" to quit): ')
        if user_input.lower() == 'exit':
            break
        if len(user_input) > 1000:
            print("Query too long. Please shorten your question.")
            continue  # skip the rest of the code in the current iteration and jump directly to the next iteration

        rag_chatbot_flow.state.user_query = user_input
        asyncio.run(rag_chatbot_flow.kickoff_async())  # ← must use async kickoff

def plot():
    rag_chatbot_flow = RagChatbotFlow()
    rag_chatbot_flow.plot()


if __name__ == "__main__":
    kickoff()
