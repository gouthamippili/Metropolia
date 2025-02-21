import os
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
OPEN_AI_API_KEY = os.getenv("OPENAI_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")


from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults

openai_llm = ChatOpenAI(model="gpt-3.5-turbo", openai_api_key=OPEN_AI_API_KEY)
groq_llm = ChatGroq(model="gemma2-9b-it", api_key=GROQ_API_KEY)

#2 Setup Tools

#search_tool=TavilySearchResults(tavily_api_key=TAVILY_API_KEY,max_results=5)

#3 Setup agent
system_prompt="Act as an super intilligent AI chatbot who is smart and friendly"
from langgraph.prebuilt import create_react_agent
def get_response_from_ai_agent(llm_id,query,allow_search,system_prompt, provider):
    if provider == "Groq":
        llm=ChatGroq(model=llm_id)
    elif provider == "OpenAI":
        llm=ChatOpenAI(model=llm_id)
    tools=TavilySearchResults(max_results=5) if allow_search else []
    agent=create_react_agent(
    model=llm,
    tools=tools,
    state_modifier=system_prompt
    )
    state={"messages":query}
    resposnse=agent.invoke(state)
    from langchain_core.messages.ai import AIMessage
    messages=resposnse.get("messages")
    ai_messages = [message.content for message in messages if isinstance(message, AIMessage)]
    return ai_messages[-1]