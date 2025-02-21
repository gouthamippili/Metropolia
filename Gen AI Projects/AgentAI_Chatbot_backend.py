#Setup Pydentic model
from pydantic import BaseModel
from AgenAI_ChatBot_Groq_Travily import get_response_from_ai_agent


class RequestState(BaseModel):
    model_name: str
    model_provider:str
    system_prompt: str
    messages: list[str]
    allow_search:bool

# setup Agent for FASTAPI to accept frontend request, install fastapi
from fastapi import FastAPI
ALLOWED_MODEL_NAMES = ["gemma2-9b-it", "qwen-2.5-32b","deepseek-r1-distill-qwen-32b","llama-3.1-8b-instant"]
app = FastAPI(title="Agent AI Chatbot")
@app.post("/agent")
def agent_endpoint(request: RequestState):
    '''
    API Endpoint to interact with the chatbot using Langgraph and serachtools.
    It dynamically selects the model specified in the request
    '''
    if request.model_name not in ALLOWED_MODEL_NAMES:
        raise ValueError(f"Invalid model name: {request.model_name}")
    llm_id=request.model_name
    query=request.messages
    allow_search=request.allow_search
    system_prompt=request.system_prompt
    provider=request.model_provider


# create AI agent and get response from it

    response=get_response_from_ai_agent(llm_id,query,allow_search,system_prompt, provider)
    return response
# Run app through uvicorn
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=9999)



