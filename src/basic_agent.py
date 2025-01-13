from typing import Literal
import openai
from langchain.chat_models import ChatOpenAI
from langchain_core.messages import SystemMessage, RemoveMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import MessagesState, StateGraph, START, END

def print_update(update):
    for k, v in update.items():
        for m in v["messages"]:
            m.pretty_print()
        if "summary" in v:
            print(v["summary"])

from langchain_core.messages import HumanMessage
# We will add a `summary` attribute (in addition to `messages` key,
# which MessagesState already has)
class State(MessagesState):
    summary: str
from chroma_db_manager import ChromaDBManager  # Import the ChromaDBManager class

import openai

class OpenAIEmbedding:
    """
    A wrapper for the OpenAI embedding model.
    """

    def __init__(self, model="text-embedding-ada-002"):
        """
        Initialize the OpenAI embedding model.
        :param model: The name of the OpenAI embedding model to use.
        """
        self.model = model

from openai import OpenAI

class OpenAIEmbedding:
    """
    A wrapper for the OpenAI embedding API.
    """

    def __init__(self, model="text-embedding-3-small"):
        """
        Initialize the OpenAI embedding client.
        :param model: The name of the OpenAI embedding model to use.
        """
        self.client = OpenAI()  # Initialize the OpenAI client
        self.model = model

    def embed(self, text: str):
        """
        Generate an embedding for the given text using OpenAI's embeddings API.
        :param text: The text to embed.
        :return: A list of floats representing the embedding vector.
        """
        try:
            # Use the OpenAI embeddings API
            response = self.client.embeddings.create(
                input=text,
                model=self.model
            )
            return response.data[0].embedding  # Extract the embedding
        except Exception as e:
            print(f"Error generating embedding: {e}")
            raise

class AgentManager:
    def __init__(self, model_name="gpt-4o", temperature=0, messages_before_summary=6, chroma_base_path="data/chroma_dbs"):
        """
        Initialize the AgentManager with model configuration, memory saver, and ChromaDBManager.
        :param model_name: Name of the language model to use.
        :param temperature: Temperature for the language model.
        :param messages_before_summary: Number of messages before summarization is triggered.
        :param chroma_base_path: Base path for Chroma databases.
        """
        # Existing attributes
        self.memory = MemorySaver()
        self.model = ChatOpenAI(model=model_name, temperature=temperature)
        self.messages_before_summary = messages_before_summary
        self.app = self._create_workflow()

        # New attribute: ChromaDBManager instance
        self.chroma_manager = ChromaDBManager(base_path=chroma_base_path)
        # Initialize Embedder (OpenAI embedding model)
        self.embedder = OpenAIEmbedding(model="text-embedding-ada-002")


    def _call_model(self, state: State):
        summary = state.get("summary", "")
        if summary:
            system_message = f"Summary of conversation earlier: {summary}"
            messages = [SystemMessage(content=system_message)] + state["messages"]
        else:
            messages = state["messages"]
        response = self.model.invoke(messages)
        return {"messages": [response]}

    def _should_continue(self, state: State) -> Literal["summarize_conversation", END]:
        """Return the next node to execute."""
        messages = state["messages"]
        if len(messages) > self.messages_before_summary:
            return "summarize_conversation"
        return END

    def _summarize_conversation(self, state: State):
        summary = state.get("summary", "")
        if summary:
            summary_message = (
                f"This is summary of the conversation to date: {summary}\n\n"
                "Extend the summary by taking into account the new messages above:"
            )
        else:
            summary_message = "Create a summary of the conversation above:"

        messages = state["messages"] + [HumanMessage(content=summary_message)]
        response = self.model.invoke(messages)
        delete_messages = [RemoveMessage(id=m.id) for m in state["messages"][:-2]]
        return {"summary": response.content, "messages": delete_messages}

    def _create_workflow(self):
        # Define a new graph
        workflow = StateGraph(State)

        # Define the conversation node and the summarize node
        workflow.add_node("conversation", self._call_model)
        workflow.add_node("summarize_conversation", self._summarize_conversation)

        # Set the entrypoint as conversation
        workflow.add_edge(START, "conversation")

        # Add conditional edges
        workflow.add_conditional_edges(
            "conversation",
            self._should_continue,
        )

        # Add edge from summarize_conversation to END
        workflow.add_edge("summarize_conversation", END)

        return workflow.compile(checkpointer=self.memory)

    def chat(self, message: str, config: dict = None):
        """
        Handle a chat message and manage mandatory memory storage.
        :param message: The user input message.
        :param config: Configuration dictionary, including the thread_id.
        :return: The response generated by the model.
        """

        if config is None:
            config = {"configurable": {"thread_id": "default"}}
            
        thread_id = config["configurable"]["thread_id"]

        # Step 1: Initialize the mandatory memory database
        mandatory_db = self.chroma_manager.get_chroma_instance(thread_id, "mandatory")

        # Step 2: Embed the user input
        query_embedding = self.embedder.embed(message)

        # Step 3: Retrieve relevant memory
        relevant_memory = self.chroma_manager.query_memory(mandatory_db, query_embedding, k=5)

        # (Optional) Process the retrieved memory
        print(f"Retrieved memory: {relevant_memory}")

        
        # Step 4: Combine user input with retrieved memory (if available)
        context = "\n".join([doc for doc in relevant_memory]) if relevant_memory else ""
        full_input = f"{context}\nUser: {message}" if context else message
        
        # Step 5: Generate the response
        input_message = HumanMessage(content=full_input)
        response = self.app.stream({"messages": [input_message]}, config, stream_mode="updates")
        
        # Step 6: Store interaction in mandatory memory
        response_text = response.get("response_text")  # Replace with actual response retrieval logic
        mandatory_db.store_interaction(thread_id, "mandatory", message, query_embedding)
        mandatory_db.store_interaction(thread_id, "mandatory", response_text, self.embedder.embed(response_text))
        
        return response
    
    def conversation(self, message: str, config: dict = None):
        for event in self.chat(message, config):
            print_update(event)

agent = AgentManager()

# Chat with the agent

conversation_items = [
    "hi! I'm bob",
    "what's my name?",
    "i like the celtics!",
    "i like how much they win",
    "what's my name?",
    "which NFL team do you think I like?",
    "i like the patriots!"
]

config = {"configurable": {"thread_id": "123456"}}

embedder = OpenAIEmbedding(model="text-embedding-ada-002")
text = "Testing OpenAI embedding generation."
embedding = embedder.embed(text)
print(embedding)

for item in conversation_items:
    agent.conversation(item, config)

