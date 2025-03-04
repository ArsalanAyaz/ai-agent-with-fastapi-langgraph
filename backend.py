from pydantic import BaseModel
from typing import List
from ai_agent import get_response_from_ai_agent
from fastapi import FastAPI

app = FastAPI(title="LangGraph AI Agent")

ALLOWED_MODEL_NAMES = ["mixtral-8x7b-32768", "gemini-1.5-pro","llama-3.3-70b-versatile"]

class RequestState(BaseModel):
    model_name: str
    model_provider: str
    messages: List[str]
    allow_search: bool

@app.post("/chat")
def chat_endpoint(request: RequestState):
    if request.model_name not in ALLOWED_MODEL_NAMES:
        return {"error": f"Model {request.model_name} is not allowed"}

    llm_id = request.model_name
    model_provider = request.model_provider
    allow_search = request.allow_search
    query = request.messages[0]  # Assuming the first message is the query

    # Get response from the AI agent
    response = get_response_from_ai_agent(llm_id, query, model_provider, allow_search)
    return {"response": response}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)











# from pydantic import BaseModel
# from typing import List
# from ai_agent import get_response_from_ai_agent

# class RequestState(BaseModel):
#     model_name : str 
#     model_provider: str 
#     # system_prompt: str 
#     messages : List[str]
#     allow_search: bool


# from fastapi import FastAPI 

# app = FastAPI(title="LangGraph AI Agent")

# ALLOWED_MODEL_NAMES=["mixtral-8x7b-32768","gemini-1.5-pro"]


# @app.post("/chat")
# def chat_endpoint(request: RequestState):
#     """
#     API endpoint to interact with chatbot using LangGraph and search tools.
#     It dynamically selects the model specified in the request

#     """

#     if request.model_name not in ALLOWED_MODEL_NAMES:
#         return {"error": f"Model {request.model_name} is not allowed"}

#     llm_id = request.model_name
#     model_provider = request.model_provider
#     allow_search = request.allow_search
#     query = request.messages

    
#     # create ai agent and get response from it    
#     response = get_response_from_ai_agent(llm_id, query, model_provider, allow_search)
#     return {"response": response}

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="127.0.0.1", port=8000)    