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
                f"This is a summary of the conversation so far: {summary}\n\n"
                "Update this summary by incorporating the key facts from the latest messages, ensuring that important details such as names, preferences, and user opinions are preserved. Do not lose established facts."
            )
        else:
            summary_message = (
                "Create a concise but factually accurate summary of the conversation above, making sure to retain any key facts such as names, preferences, and opinions."
            )

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
        if config is None:
            config = {"configurable": {"thread_id": "default"}}
        
        thread_id = config["configurable"]["thread_id"]

        # Step 1: Initialize the mandatory memory database
        mandatory_db = self.chroma_manager.get_chroma_instance(thread_id, "mandatory")

        # Step 2: Embed the user input
        query_embedding = self.embedder.embed(message)

        # Step 3: Retrieve relevant memory from the mandatory database
        relevant_memory = self.chroma_manager.query_memory(mandatory_db, query_embedding, k=5)

        # Flatten the nested list of documents and handle empty lists
        relevant_context = ""
        if relevant_memory and "documents" in relevant_memory:
            flattened_documents = [doc for sublist in relevant_memory["documents"] for doc in sublist]
            relevant_context = "\n".join(flattened_documents)

        # Step 4: Prepare the full input by appending the context to the user's message
        full_input = f"{relevant_context}\n{message}" if relevant_context else message

        # Step 5: Generate the response
        input_message = HumanMessage(content=full_input)
        response_generator = self.app.stream({"messages": [input_message]}, config, stream_mode="updates")
        
        response_text = ""
        for event in response_generator:
            # Check if the event contains a conversation with messages
            if "conversation" in event and "messages" in event["conversation"]:
                for msg in event["conversation"]["messages"]:
                    # Ensure the message object has 'content' and extract it
                    if hasattr(msg, "content"):
                        response_text += msg.content  # Append AIMessage content
        # Step 6: Store interaction in mandatory memory
        # Store the user message
        mandatory_db.add(
            documents=[message],
            embeddings=[query_embedding],
            ids=[f"{thread_id}_user_message_{len(relevant_memory)}"]
        )

        # Store the AI response
        response_embedding = self.embedder.embed(response_text)
        mandatory_db.add(
            documents=[response_text],
            embeddings=[response_embedding],
            ids=[f"{thread_id}_ai_response_{len(relevant_memory)}"]
        )

        # Step 7: Return the AI's response
        return response_text    
    def conversation(self, message: str, config: dict = None):
        response = self.chat(message, config)
        print("\nMessage: {}\nResponse: {}\n".format(message, response))

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

