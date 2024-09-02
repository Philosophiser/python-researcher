import streamlit as st
import os
from dotenv import load_dotenv
from duckduckgo_search import DDGS
from bs4 import BeautifulSoup
import requests
import re
import json
import functools
import operator
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate, MessagesPlaceholder
from langgraph.graph import StateGraph, END
from langchain.tools import tool
from langchain_openai import ChatOpenAI
from typing import Annotated, Sequence, TypedDict

# Load environment variables
load_dotenv()

# Ensure OpenAI API key is set
if 'OPENAI_API_KEY' not in os.environ:
    st.error('OpenAI API key not found. Please set it in your .env file.')
    st.stop()

# Initialize model
llm = ChatOpenAI(model="gpt-4-0613")  # Make sure to use an appropriate model

# Define custom tools
@tool("internet_search", return_direct=False)
def internet_search(query: str) -> str:
    """Searches the internet using DuckDuckGo."""
    with DDGS() as ddgs:
        results = [r for r in ddgs.text(query, max_results=5)]
        return json.dumps(results) if results else "No results found."

@tool("process_content", return_direct=False)
def process_content(url: str) -> str:
    """Processes content from a webpage."""
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    return soup.get_text()

@tool("image_search", return_direct=False)
def image_search(query: str) -> str:
    """Searches for images using DuckDuckGo."""
    with DDGS() as ddgs:
        images = [img for img in ddgs.images(query, max_results=5)]
        return json.dumps(images) if images else "No images found."

@tool("calculate_price_range", return_direct=False)
def calculate_price_range(prices: str) -> str:
    """Calculates the minimum and maximum price from a list of price strings."""
    price_list = [float(re.sub(r'[^\d.]', '', price)) for price in prices.split(',') if re.search(r'\d', price)]
    if not price_list:
        return "Unable to calculate price range due to insufficient data."
    min_price = min(price_list)
    max_price = max(price_list)
    return json.dumps({"price_min": f"${min_price:.2f}", "price_max": f"${max_price:.2f}"})

tools = [internet_search, process_content, image_search, calculate_price_range]

# Helper function for creating agents
def create_agent(llm: ChatOpenAI, tools: list, system_prompt: str):
    template_str = (
        "{system}\n"
        "{messages}\n"
        "{agent_scratchpad}\n"
    )
    
    prompt = PromptTemplate(
        template=template_str,
        input_variables=["system", "messages", "agent_scratchpad"]
    ).partial(system=system_prompt)
    
    agent = create_openai_tools_agent(llm, tools, prompt)
    executor = AgentExecutor(agent=agent, tools=tools)
    return executor

# Define agent nodes
def agent_node(state, agent, name):
    result = agent.invoke(state)
    return {"messages": [HumanMessage(content=result["output"], name=name)]}

# Create Agent Supervisor
members = ["Web_Searcher", "Image_Searcher", "Wine_Researcher"]
system_prompt = (
    "As a supervisor, your role is to oversee a dialogue between these"
    " workers: {members}. Based on the user's request for wine information,"
    " determine which worker should take the next action. Each worker is responsible for"
    " executing a specific task and reporting back their findings and progress. Once all tasks are complete,"
    " indicate with 'FINISH'."
)

options = ["FINISH"] + members
function_def = {
    "name": "route",
    "description": "Select the next role.",
    "parameters": {
        "title": "routeSchema",
        "type": "object",
        "properties": {"next": {"title": "Next", "anyOf": [{"enum": options}] }},
        "required": ["next"],
    },
}

prompt = PromptTemplate(
    template="{system}\n{messages}\nGiven the conversation above, who should act next? Or should we FINISH? Select one of: {options}",
    input_variables=["system", "messages", "options"],
).partial(system=system_prompt, options=str(options))

supervisor_chain = (prompt | llm.bind_functions(functions=[function_def], function_call="route") | JsonOutputParser())

search_agent = create_agent(llm, tools, "You are a web searcher. Search the internet for information about the specified wine.")
search_node = functools.partial(agent_node, agent=search_agent, name="Web_Searcher")

image_search_agent = create_agent(llm, tools, "You are an image searcher. Search for a clear image of the wine bottle.")
image_search_node = functools.partial(agent_node, agent=image_search_agent, name="Image_Searcher")

wine_research_agent = create_agent(llm, tools, 
        """You are a Wine Researcher. Your task is to compile structured information about a specific wine.
        Use the provided search results to extract the following details:
        * image_url (link to a clear picture of the wine bottle)
        * price_min (minimum price found)
        * price_max (maximum price found)
        * varietal
        * production_notes (concise, objective explanation of how the wine is produced)
        * tasting_notes (concise, objective tasting profile of the wine without fluffy language)
        * region_notes (concise, objective description of the country/region/sub-region specific to the wine and how it influences the wine's characteristics)
        * winery_notes (concise, objective description of the winery and how it influences the wine's characteristics)
        
        Ensure the output adheres strictly to the following JSON Schema:""")
wine_research_node = functools.partial(agent_node, agent=wine_research_agent, name="Wine_Researcher")

# Define the Agent State, Edges and Graph
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    next: str

workflow = StateGraph(AgentState)
workflow.add_node("Web_Searcher", search_node)
workflow.add_node("Image_Searcher", image_search_node)
workflow.add_node("Wine_Researcher", wine_research_node)
workflow.add_node("supervisor", supervisor_chain)

# Define edges
for member in members:
    workflow.add_edge(member, "supervisor")

conditional_map = {k: k for k in members}
conditional_map["FINISH"] = END
workflow.add_conditional_edges("supervisor", lambda x: x["next"], conditional_map)
workflow.set_entry_point("supervisor")

graph = workflow.compile()

# Streamlit UI
st.title("Wine Research App")

# Input for wine name
wine_name = st.text_input("Enter the name of the wine you want to research:")

if st.button("Research Wine"):
    if wine_name:
        with st.spinner("Researching wine..."):
            try:
                result = None
                for s in graph.stream({
                    "messages": [HumanMessage(content=f"Research the wine: {wine_name}. Focus on finding accurate price range data.")]
                }):
                    if "__end__" not in s:
                        result = s
                
                if isinstance(result, dict) and all(key in result for key in ["image_url", "price_min", "price_max", "varietal", "production_notes", "tasting_notes", "region_notes", "winery_notes"]):
                    st.success("Research complete!")
                    
                    # Display image
                    st.image(result["image_url"], caption="Wine Bottle Image", use_column_width=True)
                    
                    # Display wine information
                    st.subheader("Wine Information")
                    st.write(f"**Varietal:** {result['varietal']}")
                    st.write(f"**Price Range:** {result['price_min']} - {result['price_max']}")
                    
                    st.subheader("Production Notes")
                    st.write(result["production_notes"])
                    
                    st.subheader("Tasting Notes")
                    st.write(result["tasting_notes"])
                    
                    st.subheader("Region Notes")
                    st.write(result["region_notes"])
                    
                    st.subheader("Winery Notes")
                    st.write(result["winery_notes"])
                else:
                    st.error("Unable to retrieve complete information about the wine.")
            except Exception as e:
                st.error(f"An error occurred while researching the wine: {str(e)}")
    else:
        st.warning("Please enter a wine name.")

# Run the Streamlit app
if __name__ == "__main__":
    st.run()
