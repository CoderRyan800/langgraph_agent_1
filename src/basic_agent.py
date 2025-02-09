import traceback
import sys
import pdb  # Optional: for interactive post-mortem debugging
from typing import Literal
import openai
from pathlib import Path
from openai import OpenAI
from datetime import datetime
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, ToolMessage, RemoveMessage, AIMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import MessagesState, StateGraph, START, END

from langchain_core.messages import AIMessage
from langchain_core.tools import tool

from langgraph.prebuilt import ToolNode

@tool
def get_weather(location: str):
    """Call to get the current weather."""
    if location.lower() in ["sf", "san francisco"]:
        return "It's 60 degrees and foggy."
    else:
        return "It's 90 degrees and sunny."


@tool
def get_coolest_cities():
    """Get a list of coolest cities"""
    return "nyc, sf"

tool_list = [get_weather, get_coolest_cities]
tool_node = ToolNode(tool_list)

message_with_single_tool_call = AIMessage(
    content="",
    tool_calls=[
        {
            "name": "get_weather",
            "args": {"location": "sf"},
            "id": "tool_call_id",
            "type": "tool_call",
        }
    ],
)

print(tool_node.invoke({"messages": [message_with_single_tool_call]}))

def print_update(update):
    for k, v in update.items():
        for m in v["messages"]:
            m.pretty_print()    
        if "summary" in v:
            print(v["summary"])

# We will add a `summary` attribute (in addition to `messages` key,
# which MessagesState already has)
class State(MessagesState):
    summary: str
from chroma_db_manager import ChromaDBManager  # Import the ChromaDBManager class

def get_sliding_window_chunk(messages, turns=5):
    """
    Extract the last `turns` pairs of messages (assumes one human message
    followed by one AI response per turn). If there are fewer messages than
    required, return the entire list.
    """
    num_messages_per_chunk = turns * 2  # each turn = human + AI message
    return messages[-num_messages_per_chunk:] if len(messages) >= num_messages_per_chunk else messages

def aggregate_chunk(chunk):
    """
    Combine the content of the messages in the chunk into a single string.
    """
    return "\n".join([msg.content for msg in chunk])


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
        self.model = ChatOpenAI(model=model_name, temperature=temperature).bind_tools(tool_list)
        #self.model = ChatOpenAI(model=model_name, temperature=temperature)

        self.messages_before_summary = messages_before_summary
        self.app = self._create_workflow()

        # New attribute: ChromaDBManager instance
        self.chroma_manager = ChromaDBManager(base_path=chroma_base_path)
        # Initialize Embedder (OpenAI embedding model)
        self.embedder = OpenAIEmbedding(model="text-embedding-ada-002")


    def _call_model(self, state: State):
        summary = state.get("summary", "")
        # Create a working copy of the messages list
        messages = state["messages"].copy()

        # Load the custom system message based on the current thread_id (if available)
        if hasattr(self, "current_thread_id"):
            custom_system_message = self.load_system_message(self.current_thread_id)
            if custom_system_message:
                # Prepend the custom system message to the conversation
                messages.insert(0, SystemMessage(content=custom_system_message))
        
        # If there's a summary, add it as an additional system message.
        if summary:
            summary_message = f"Summary of conversation earlier: {summary}"
            # Insert it after the custom system message, or at the beginning if no custom message exists.
            insert_index = 1 if hasattr(self, "current_thread_id") and custom_system_message else 0
            messages.insert(insert_index, SystemMessage(content=summary_message))
        print("DEBUG: MESSAGES AT THE CALL MODEL NODE:\n")
        for index in range(len(messages)):
            print(f"Message {index}: ")
            print(messages[index].pretty_print())
        print("DEBUG: End messages at call model node\n")

        response = self.model.invoke(messages)
        return {"messages": [response]}

    def _should_continue(self, state: State) -> Literal["tools", "summarize_conversation", END]:
        messages = state["messages"]
        last_message = messages[-1]
        if last_message.tool_calls:
            return "tools"
        elif len(messages) > self.messages_before_summary:
            return "summarize_conversation"
        return END
    
        # if len(messages) > self.messages_before_summary:
        #     return "summarize_conversation"
        # return END

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

        # Append the summarization prompt
        messages = state["messages"] + [HumanMessage(content=summary_message)]

        # Debugging: Print the messages before summarization
        print("DEBUG: MESSAGES AT THE SUMMARIZE NODE BEFORE SUMMARIZATION:\n")
        for index, msg in enumerate(messages):
            print(f"Message {index}: {msg.pretty_print()}")
        print("DEBUG: End messages to be summarized\n")

        # Invoke the model with the cleaned message list.
        response = self.model.invoke(messages)

        # Store the updated summary
        new_summary = response.content

        # Set how many messages we want to keep for context (e.g., last 2 messages)
        NUM_MESSAGES_TO_KEEP = 2

        # Step 1: Create `RemoveMessage` objects for all **older** messages
        old_messages_to_remove = [
            RemoveMessage(id=m.id) for m in state["messages"][:-NUM_MESSAGES_TO_KEEP]
        ]

        # Step 2: Create `RemoveMessage` objects for tool-related messages in **recent history**
        recent_tool_messages_to_remove = [
            RemoveMessage(id=m.id)
            for m in state["messages"][-NUM_MESSAGES_TO_KEEP:]
            if isinstance(m, ToolMessage) or (isinstance(m, AIMessage) and bool(m.tool_calls))
        ]

        # Combine both lists
        delete_messages = old_messages_to_remove + recent_tool_messages_to_remove

        return {"summary": new_summary, "messages": delete_messages}
    
    def _create_workflow(self):
        # Define a new graph
        workflow = StateGraph(State)

        # Define the conversation node and the summarize node
        workflow.add_node("conversation", self._call_model)
        workflow.add_node("summarize_conversation", self._summarize_conversation)
        workflow.add_node("tools", tool_node)

        # Set the entrypoint as conversation
        workflow.add_edge(START, "conversation")

        # Add conditional edges
        workflow.add_conditional_edges("conversation", 
                                       self._should_continue, 
                                       ["tools", "summarize_conversation", END])

        # Add edge from summarize_conversation to END
        workflow.add_edge("summarize_conversation", END)
        # Add edge from tool note back to conversation.
        workflow.add_edge("tools","conversation")

        return workflow.compile(checkpointer=self.memory)

    def load_system_message(self, thread_id: str) -> str:
        """
        Load the system message text from a file based on the thread_id.
        The filename is assumed to be in the format: system_messages/system_{thread_id}.txt
        """
        # Resolve the current file's directory (assuming it's in <repo_root>/src)
        current_dir = Path(__file__).resolve().parent
        # Get the repository root by navigating up one directory
        repo_root = current_dir.parent
        # Construct the path to the system message file
        filename = repo_root / "system_messages" / f"system_{thread_id}.txt"

        try:
            with open(filename, "r", encoding="utf-8") as f:
                return f.read().strip()
        except Exception as e:
            print(f"Could not load system message file for thread {thread_id}: {e}")
            return ""

    def update_vector_memory(self, thread_id, messages, turns=5):
        """
        Update the vector memory store by aggregating the last `turns` conversation
        pairs into one chunk, embedding it, and saving it to the vector DB.
        """
        # Get the mandatory DB instance
        mandatory_db = self.chroma_manager.get_chroma_instance(thread_id, "mandatory")
        
        # Extract the sliding window chunk from the conversation history
        chunk = get_sliding_window_chunk(messages, turns)
        # Aggregate the chunk into a single text string
        aggregated_text = aggregate_chunk(chunk)
        # Prepend the current UTC timestamp in ISO format followed by a colon
        chunk_text = f"{datetime.utcnow().isoformat()}: {aggregated_text}"
        # Generate the embedding for the aggregated text
        chunk_embedding = self.embedder.embed(chunk_text)
        
        # Create a unique ID for this memory chunk; here we use the length of the message list
        unique_id = f"{thread_id}_chunk_{datetime.utcnow().isoformat()}"
        
        # Save the chunk to the vector DB
        mandatory_db.add(
            documents=[chunk_text],
            embeddings=[chunk_embedding],
            ids=[unique_id]
        )

    def chat(self, message: str, config: dict = None):
        if config is None:
            config = {"configurable": {"thread_id": "default"}}
        
        thread_id = config["configurable"]["thread_id"]
        self.current_thread_id = thread_id  # Save the current thread_id for later use

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

        # Step 6: Update vector memory with the latest conversation chunk
        # (Instead of storing the user message and AI response separately)

        # Initialize or retrieve the conversation history list on the AgentManager
        if not hasattr(self, "conversation_history"):
            self.conversation_history = []

        # Append the new messages to the conversation history.
        # (Assuming HumanMessage and SystemMessage are the types you use.)
        self.conversation_history.append(HumanMessage(content=message))
        self.conversation_history.append(AIMessage(content=response_text))

        # Now update the vector memory using a sliding window over the last K turns
        # (For example, turns=5 means the last 10 messages will be aggregated.)
        self.update_vector_memory(thread_id, self.conversation_history, turns=5)

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
    "i like the patriots!",
    "what's the weather in san francisco?",
    "which are the coolest cities?"
]

config = {"configurable": {"thread_id": "123456"}}

embedder = OpenAIEmbedding(model="text-embedding-ada-002")
text = "Testing OpenAI embedding generation."
embedding = embedder.embed(text)
print(embedding)

print(agent.model.invoke("what's the weather in sf?").tool_calls)
try:
    for item in conversation_items:
        agent.conversation(item, config)
except Exception as e:
    print("An exception occurred:")
    traceback.print_exc(file=sys.stdout)  # This prints the full traceback to stdout
    # Optionally, drop into the debugger for interactive debugging:
    pdb.post_mortem()


