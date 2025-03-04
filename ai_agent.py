from dotenv import load_dotenv
load_dotenv()
import os

GROQ_API_KEY = os.environ.get('GROQ_API_KEY')
TAVILY_API_KEY = os.environ.get('TAVILY_API_KEY')
GOOGLE_API_KEY = os.environ.get('GOOGLE_API_KEY')

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_community.tools import TavilySearchResults
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from typing import List
from langchain_core.tools import tool

system_prompt = "You are a helpful chatbot who is smart and friendly."

# Define a custom tool for Tavily search
@tool
def tavily_search(query: str) -> str:
    """Search the web using Tavily and return the results."""
    search_tool = TavilySearchResults(api_key=TAVILY_API_KEY, max_results=5)
    results = search_tool.invoke(query)
    return str(results)

def get_response_from_ai_agent(llm_id, query, model_provider, allow_search):
    # Initialize the model
    if model_provider == 'Groq':
        model = ChatGroq(model=llm_id, api_key=GROQ_API_KEY)
    elif model_provider == 'Gemini':
        model = ChatGoogleGenerativeAI(model=llm_id, api_key=GOOGLE_API_KEY)
    else:
        raise ValueError(f"Unsupported model provider: {model_provider}")

    # Initialize tools
    if allow_search:
        tools = [tavily_search]  # Use the custom tool
    else:
        tools = []

    # Create the agent
    agent = create_react_agent(model, tools=tools)

    # Initialize the state with system and user messages
    state = {
        "messages": [
            SystemMessage(content=system_prompt),
            HumanMessage(content=query)
        ]
    }

    # Invoke the agent
    response = agent.invoke(state)

    # Extract the AI response
    messages = response.get("messages", [])
    ai_messages = [message.content for message in messages if isinstance(message, AIMessage)]

    return ai_messages[-1] if ai_messages else "No response generated."





# from dotenv import load_dotenv
# load_dotenv()
# import os

# GROQ_API_KEY = os.environ.get('GROQ_API_KEY')
# TAVILY_API_KEY = os.environ.get('TAVILY_API_KEY')
# GOOGLE_API_KEY = os.environ.get('GOOGLE_API_KEY')

# from langchain_google_genai import ChatGoogleGenerativeAI
# # google_llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", api_key=GOOGLE_API_KEY)

# from langchain_groq import ChatGroq
# # groq_llm = ChatGroq(model="mixtral-8x7b-32768", api_key=GROQ_API_KEY)

# from langchain_community.tools import TavilySearchResults
# # search_tool = TavilySearchResults(max_results=5)

# system_prompt = "You are a helpful chatbot who is smart and friendly."

# from langgraph.prebuilt import create_react_agent
# from langchain_core.messages import SystemMessage

# # Wrap the search_tool in a list
# # tools = [search_tool]

# # Create the agent with the system prompt
# from typing import List 

# def get_response_from_ai_agent(llm_id, query, model_provider, allow_search):

#     if model_provider == 'Groq':
#         model = ChatGroq(model="mixtral-8x7b-32768", api_key=GROQ_API_KEY)
#     elif model_provider == 'Gemini':
#         model = ChatGoogleGenerativeAI(model="gemini-1.5-pro", api_key=GOOGLE_API_KEY)    


#     if allow_search is True:
#         search_tool = TavilySearchResults(max_results=5)
#     else:
#         []    

#     agent = create_react_agent(model, tools=search_tool)

    
#     state = {"messages": [("user", query)]}
#     response = agent.invoke(state)

#     from langchain_core.messages.ai import AIMessage
#     messages= response.get("messages")
#     ai_messages= [message.content for message in messages if isinstance(message, AIMessage)]

#     return ai_messages[-1]

    