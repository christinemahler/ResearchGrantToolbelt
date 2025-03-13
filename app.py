# Import necessary libraries
from dotenv import load_dotenv
import chainlit as cl
from bs4 import BeautifulSoup
import fnmatch
import os

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain_qdrant import QdrantVectorStore
from langchain_text_splitters import HTMLSectionSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import START, StateGraph

from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams

from typing import TypedDict, Annotated, List
from typing_extensions import List, TypedDict

# Load environment variables
load_dotenv()

# Create html splitter
headers_to_split_on = [("h1", "Header 1"), ("h2", "Header 2"), ("h3", "Header 3")]

html_splitter = HTMLSectionSplitter(
    headers_to_split_on=headers_to_split_on
)

# Path to the directory containing HTML files
directory_path = "data/Opportunities"

# Clean html files and load into opportunities list
def clean_html(html_raw):
    soup = BeautifulSoup(html_raw.replace("\n", "").replace("\t", ""), 'html.parser')
    # Remove div tags (could use some more work)
    for div_tag in soup.find_all('div'):
        div_tag.replaceWithChildren()
    # Remove script tags 
    for script in soup.find_all('script'):
        script.decompose()
    # Remove title tags
    for title in soup.find_all('title'):
        opportunity_title = title.string
        title.decompose()
    # Remove meta tags 
    for meta in soup.find_all('meta'):
        meta.decompose()
    # Add metadata to header tags 
    for header in soup.find_all(['h1', 'h2', 'h3']):
        header.string.replace_with(f"{opportunity_title} {header.string}")
    # Return clean file
    return soup.prettify() 

# Initialize opportunities document list
opportunities = []

# Iterate over each file in the directory
for filename in os.listdir(directory_path):
    # Check if the file is an HTML file
    if fnmatch.fnmatch(filename, '*.html'):
        file_path = os.path.join(directory_path, filename)
        #Read html file
        with open(file_path, 'r') as f:
            html_raw = f.read()
        #Add cleaned and chunked html file to opportunities list
        opportunities.extend(html_splitter.split_text(clean_html(html_raw)))
        
# Get fine-tuned embeddings        
#embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
embeddings = HuggingFaceEmbeddings(model_name="christinemahler/aie5-midterm")

opportunities_qdrant = QdrantVectorStore.from_documents(
    opportunities,
    embeddings,
    location=":memory:",  # Local mode with in-memory storage only
    collection_name="opportunities",
)

# Create retriever
opportunities_retriever = opportunities_qdrant.as_retriever(search_kwargs={"k": 5})

# Create retrieve helper function
def retrieve(state):
  retrieved_docs = opportunities_retriever.invoke(state["question"])
  return {"context" : retrieved_docs}

# Configure prompt template
opportunities_rag_prompt_template = """\
Use the provided context to answer the user's question. Only use the provided context to answer the question.

If the user asks you to evaluate the study complexity, provide a complexity score of high, medium, or low and a reason for all of the following categories:
1) Regulatory and Compliance (rank higher if study requires compliance with multiple regulations and provide examples)
2) Data Collection and Management (identify data elements needed, data complexity, data sensitivity, and data collection frequency)
3) Statistical Analysis and Manuscript Development (provide examples)
4) Information Technology (identify data collection services and equipment needed along with software licenses and subscriptions)
5) Operational (includes project administration and site onboarding, coordination and training)
6) Financial (includes budget management and effort allocation over entire duration of project; rank higher if more resources are required)

If you do not know the answer, or it's not contained in the provided context response with "I don't know"

Context:
{context}

Question:
{question}
"""

opportunities_rag_prompt = ChatPromptTemplate.from_template(opportunities_rag_prompt_template)

# Initialize your language model
opportunities_llm = ChatOpenAI(model="gpt-4o-mini", tags=["opportunities_llm"])

# Create generate helper function
def generate(state):
  docs_content = "\n\n".join(doc.page_content for doc in state["context"])
  messages = opportunities_rag_prompt.format_messages(question=state["question"], context=docs_content)
  response = opportunities_llm.invoke(messages)
  return {"response" : response.content}

# Define State class
class State(TypedDict):
  question: str
  context: List[Document]
  response: str
  
# Build graph
graph_builder = StateGraph(State).add_sequence([retrieve, generate])
graph_builder.add_edge(START, "retrieve")
graph = graph_builder.compile()

@cl.on_chat_start
async def start():
  cl.user_session.set("graph", graph)
  
@cl.on_message
async def handle_message(message: cl.Message):
    graph = cl.user_session.get("graph")
    response = await graph.ainvoke({"question" : message.content})
    await cl.Message(response['response']).send()