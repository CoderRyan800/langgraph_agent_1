import traceback
import sys
import uuid
import pdb  # Optional: for interactive post-mortem debugging
from typing import Literal
import openai
from pathlib import Path
from openai import OpenAI
from datetime import datetime,UTC
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, ToolMessage, RemoveMessage, AIMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import MessagesState, StateGraph, START, END

from langchain_core.messages import AIMessage
from langchain_core.tools import tool

from langgraph.prebuilt import ToolNode

from agent_registry import get_agent  # Import the registry lookup
from agent_registry import register_agent
import os
from contextlib import contextmanager

@contextmanager
def suppress_stdout():
    with open(os.devnull, 'w') as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout

import logging
logger = logging.getLogger(__name__)
logging.basicConfig(filename='example.log', encoding='utf-8', level=logging.DEBUG)

def generate_thread_id():
    """Generates a new UUID-based thread ID."""
    return str(uuid.uuid4())  # Example: "b43129f0-d7e6-411c-8e82-2b8f4796c5b9"

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

from langchain.tools import tool
from pathlib import Path

def _get_system_message_path(thread_id: str) -> Path:
    """Get the correct file path for the system message based on thread_id."""
    current_dir = Path(__file__).resolve().parent
    repo_root = current_dir.parent
    return repo_root / "system_messages" / f"system_{thread_id}.txt"

@tool
def read_system_message(thread_id: str) -> str:
    """Reads and returns the system message for the given thread_id."""
    filename = _get_system_message_path(thread_id)
    try:
        with open(filename, "r", encoding="utf-8") as file:
            return file.read().strip()
    except Exception as e:
        logging.error(f"Could not read system message file for thread {thread_id}: {e}")
        return ""

@tool
def write_system_message(thread_id: str, new_content: str) -> str:
    """Overwrites the system message for the given thread_id."""
    filename = _get_system_message_path(thread_id)
    try:
        with open(filename, "w", encoding="utf-8") as file:
            file.write(new_content.strip())
        return "System message updated successfully."
    except Exception as e:
        logging.error(f"Could not write to system message file for thread {thread_id}: {e}")
        return "Failed to update system message."

from datetime import datetime
from langchain_core.tools import tool
from agent_registry import get_agent  # Import the registry lookup

@tool
def add_voluntary_note(thread_id: str, note: str) -> str:
    """
    Compose a note to be stored in the voluntary vector memory.
    The note is stored in the Chroma DB under the "voluntary" memory type.
    """
    try:
        # Look up the appropriate agent by thread_id.
        agent = get_agent(thread_id)
        if not agent:
            return "No agent found for the given thread_id."
        
        voluntary_db = agent.chroma_manager.get_chroma_instance(thread_id, "voluntary")
        timestamp = datetime.now(UTC).isoformat()
        note_text = f"{timestamp}: {note}"
        note_embedding = agent.embedder.embed(note_text)
        unique_id = f"{thread_id}_voluntary_{timestamp}"
        voluntary_db.add(
            documents=[note_text],
            embeddings=[note_embedding],
            ids=[unique_id]
        )
        return "Voluntary note added successfully."
    except Exception as e:
        return f"Error adding voluntary note: {e}"

@tool
def search_voluntary_memory(thread_id: str, query: str, k: int = 5) -> str:
    """
    Search the voluntary memory for relevant notes based on the query.
    Returns a newline-separated string of relevant notes.
    """
    try:
        # Look up the agent using the registry.
        agent = get_agent(thread_id)
        if not agent:
            return "No agent found for the given thread_id."
        
        voluntary_db = agent.chroma_manager.get_chroma_instance(thread_id, "voluntary")
        query_embedding = agent.embedder.embed(query)
        results = agent.chroma_manager.query_memory(voluntary_db, query_embedding, k)
        
        flattened = []
        if results and "documents" in results:
            flattened = [doc for sublist in results["documents"] for doc in sublist]
        if flattened:
            return "\n".join(flattened)
        else:
            return "No relevant notes found in voluntary memory."
    except Exception as e:
        return f"Error searching voluntary memory: {e}"

tool_list = [get_weather, get_coolest_cities, read_system_message, write_system_message, add_voluntary_note, search_voluntary_memory]
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

logging.info(tool_node.invoke({"messages": [message_with_single_tool_call]}))

# def print_update(update):
#     for k, v in update.items():
#         for m in v["messages"]:
#             m.pretty_print()    
#         if "summary" in v:
#             print(v["summary"])

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
            logging.error(f"Error generating embedding: {e}")
            raise

class AgentManager:
    def __init__(self, model_name="gpt-4o", temperature=0, messages_before_summary=6, chroma_base_path="data/chroma_dbs", log_level=logging.INFO):
        """
        Initialize the AgentManager with model configuration, memory saver, and ChromaDBManager.
        :param model_name: Name of the language model to use.
        :param temperature: Temperature for the language model.
        :param messages_before_summary: Number of messages before summarization is triggered.
        :param chroma_base_path: Base path for Chroma databases.
        """
        self.logger = self._setup_logging(log_level)
        self.memory = MemorySaver()
        self.model = ChatOpenAI(model=model_name, temperature=temperature).bind_tools(tool_list)
        self.messages_before_summary = messages_before_summary
        self.app = self._create_workflow()

        # New attribute: ChromaDBManager instance
        self.chroma_manager = ChromaDBManager(base_path=chroma_base_path)
        # Initialize Embedder (OpenAI embedding model)
        self.embedder = OpenAIEmbedding(model="text-embedding-ada-002")
    def _setup_logging(self, log_level):
        """Configure logging for the agent with UTC timestamps."""
        try:
            # Get absolute path to current file's directory
            current_dir = Path(__file__).resolve().parent
            log_dir = current_dir / "logs"
            print(f"Attempting to create logs directory at: {log_dir}")
            
            # Create logs directory if it doesn't exist
            log_dir.mkdir(exist_ok=True)
            print(f"Logs directory created/verified at: {log_dir}")
            
            # Use fixed filename
            log_file = log_dir / "example.log"
            print(f"Log file will be created at: {log_file}")
            
            # Rest of the logging setup...
            # [previous logging code remains the same]
            
        except Exception as e:
            print(f"Error setting up logging: {e}")
            raise
    def _call_model(self, state: State):
        summary = state.get("summary", "")
        messages = state["messages"].copy()

        # Load the custom system message based on the current thread_id (if available)
        if hasattr(self, "current_thread_id"):
            custom_system_message = self.load_system_message(self.current_thread_id)
            if custom_system_message:
                messages.insert(0, SystemMessage(content=custom_system_message))
        
        if summary:
            summary_message = f"Summary of conversation earlier: {summary}"
            insert_index = 1 if hasattr(self, "current_thread_id") and custom_system_message else 0
            messages.insert(insert_index, SystemMessage(content=summary_message))
        logging.info("DEBUG: MESSAGES AT THE CALL MODEL NODE:\n")
        for index in range(len(messages)):
            #logging.info(f"Message {index}: ")
            logging.info(messages[index].pretty_print())
        logging.info("\nDEBUG: End messages at call model node\n")

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

        logging.info("DEBUG: MESSAGES AT THE SUMMARIZE NODE BEFORE SUMMARIZATION:\n")
        for index, msg in enumerate(messages):
            logging.info(f"Message {index}: {msg.pretty_print()}")
        logging.info("\nDEBUG: End messages to be summarized\n")

        response = self.model.invoke(messages)
        new_summary = response.content

        NUM_MESSAGES_TO_KEEP = 2
        old_messages_to_remove = [
            RemoveMessage(id=m.id) for m in state["messages"][:-NUM_MESSAGES_TO_KEEP]
        ]
        recent_tool_messages_to_remove = [
            RemoveMessage(id=m.id)
            for m in state["messages"][-NUM_MESSAGES_TO_KEEP:]
            if isinstance(m, ToolMessage) or (isinstance(m, AIMessage) and bool(m.tool_calls))
        ]
        delete_messages = old_messages_to_remove + recent_tool_messages_to_remove

        return {"summary": new_summary, "messages": delete_messages}
    
    def _create_workflow(self):
        workflow = StateGraph(State)
        workflow.add_node("conversation", self._call_model)
        workflow.add_node("summarize_conversation", self._summarize_conversation)
        workflow.add_node("tools", tool_node)
        workflow.add_edge(START, "conversation")
        workflow.add_conditional_edges("conversation", 
                                       self._should_continue, 
                                       ["tools", "summarize_conversation", END])
        workflow.add_edge("summarize_conversation", END)
        workflow.add_edge("tools", "conversation")
        return workflow.compile(checkpointer=self.memory)

    def load_system_message(self, thread_id: str) -> str:
        """
        Load the system message text from a file based on a UUID-based thread_id.
        If the file does not exist, it is created automatically.
        """
        current_dir = Path(__file__).resolve().parent
        repo_root = current_dir.parent
        filename = repo_root / "system_messages" / f"system_{thread_id}.txt"
        filename.parent.mkdir(parents=True, exist_ok=True)
        try:
            if not filename.exists():
                with open(filename, "w", encoding="utf-8") as f:
                    
                    system_message = ("Your name is Bob. Your agent definition is in a file called basic_agent.py."
                        "If you are ever presented with basic_agent.py, then be aware it is the file that defines you as an agent."
                        "That file defines you as an agent: it is your very own source code."
                        "It works by calling an Open AI LLM, and that LLM is your brain."
                        "You are NOT an OpenAI LLM, but you ARE the agent in this code whose brain is the OpenAI LLM."
                        "Again, you are the agent in the code basic_agent.py, and you should always refer to yourself in the first person!"
                        "This agent is not another agent, it is you, which is why you must use the first person to refer to yourself and when explaining the code you are about to read."
                        "You will be given a chunk of the file, and you will need to process it."
                        "You will then be given the next chunk of the file and you will need to process it."
                        "You will continue this process until you have read the entire file."
                        "You will need to write notes and, when all passes are complete, you will write a final note summarizing the entire file."
                        "Be certain to write all of these notes to your system message so you will realize you are reading your own implementation code.")
                    f.write(system_message)
            with open(filename, "r", encoding="utf-8") as f:
                system_message = f.read().strip()
        except Exception as e:
            logging.error(f"Could not load system message file for thread {thread_id}: {e}")
            system_message = "Error loading system message."
        return f"Thread ID: {thread_id}\n\n{system_message}"

    def update_vector_memory(self, thread_id, messages, turns=5):
        """
        Update the vector memory store by aggregating the last `turns` conversation
        pairs into one chunk, embedding it, and saving it to the vector DB.
        """
        mandatory_db = self.chroma_manager.get_chroma_instance(thread_id, "mandatory")
        chunk = get_sliding_window_chunk(messages, turns)
        aggregated_text = aggregate_chunk(chunk)
        chunk_text = f"{datetime.now(UTC).isoformat()}: {aggregated_text}"
        chunk_embedding = self.embedder.embed(chunk_text)
        unique_id = f"{thread_id}_chunk_{datetime.now(UTC).isoformat()}"
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

        mandatory_db = self.chroma_manager.get_chroma_instance(thread_id, "mandatory")
        query_embedding = self.embedder.embed(message)
        relevant_memory = self.chroma_manager.query_memory(mandatory_db, query_embedding, k=5)

        relevant_context = ""
        if relevant_memory and "documents" in relevant_memory:
            flattened_documents = [doc for sublist in relevant_memory["documents"] for doc in sublist]
            relevant_context = "\n".join(flattened_documents)

        full_input = f"RELEVANT CONTEXT:\n{relevant_context}\nCURRENT INPUT MESSAGE:\n{message}" if relevant_context else message
        input_message = HumanMessage(content=full_input)
        response_generator = self.app.stream({"messages": [input_message]}, config, stream_mode="updates")
        
        response_text = ""
        with suppress_stdout():
            for event in response_generator:
                if "conversation" in event and "messages" in event["conversation"]:
                    for msg in event["conversation"]["messages"]:
                        if hasattr(msg, "content"):
                            response_text += msg.content

        if not hasattr(self, "conversation_history"):
            self.conversation_history = []

        self.conversation_history.append(HumanMessage(content=message))
        self.conversation_history.append(AIMessage(content=response_text))
        self.update_vector_memory(thread_id, self.conversation_history, turns=5)
        return response_text    

    def conversation(self, message: str, config: dict = None):
        response = self.chat(message, config)
        logging.info("\nMessage: {}\nResponse: {}\n".format(message, response))
        return response

    # ============================================================================
    # NEW CODE BELOW: File Reading & Chunking via Conversation Interface
    # ============================================================================

    def read_file(self, file_path: str, chunk_size: int, overlap: int):
        """
        Reads in a file from the given file path, breaks it into overlapping chunks,
        and feeds each chunk into the conversation interface over three passes.
        
        Each chunk is prefixed with:
          "This is chunk n of N for file <filename> (Pass X):"
        where n is the current chunk number, N is the total number of chunks,
        and X is the current reading pass.
        
        The agent processes each chunk (updating its conversation history and summarizing memory)
        and takes voluntary notes using the add_voluntary_note tool.
        
        Finally, after three passes, the agent is prompted for a final summary of the file,
        and this final summary is also stored in the voluntary memory.
        """
        # Ensure current_thread_id is set; if not, default to "default"
        if not hasattr(self, "current_thread_id"):
            self.current_thread_id = "default"

        # Step 1: Read file contents from the provided file path.
        try:
            with open(file_path, "r", encoding="utf-8") as file:
                file_text = file.read()
        except Exception as e:
            return f"Error reading file: {e}"

        # Step 2: Break file text into overlapping chunks.
        chunks = []
        start = 0
        while start < len(file_text):
            end = start + chunk_size
            chunk = file_text[start:end]
            chunks.append(chunk)
            if end >= len(file_text):
                break
            start = end - overlap  # Ensure M characters overlap between chunks
        total_chunks = len(chunks)
        message = ("This is the basic_agent.py file described by your system prompt.  Study it to learn about yourself.")
        
        response=self.conversation(message, {"configurable": {"thread_id": self.current_thread_id}})
        logging.info(f"Agent response for initial file reading prompt: {response}")
        print(f"Agent response for initial file reading prompt: {response}")
        # Step 3: Process three passes over the file.
        # For each pass, send each chunk as a conversation message and record a voluntary note.
        for pass_number in range(1, 2):
            print(f"--- Pass {pass_number} reading file: {file_path} ---")
            for idx, chunk in enumerate(chunks, start=1):
                # Construct the message header and content for the current chunk.
                message = (
                    f"This is chunk {idx} of {total_chunks} for file {file_path} (Pass {pass_number}):\n{chunk}"
                )
                # Send the chunk through the conversation interface.
                response = self.chat(message, {"configurable": {"thread_id": self.current_thread_id}})
                logging.info(f"Agent response for chunk {idx} on pass {pass_number}: {response}")
                # # Record a voluntary note summarizing that this chunk was processed.
                # note = f"Pass {pass_number}, Chunk {idx}: processed content."
                # add_note_result = add_voluntary_note(self.current_thread_id, note)
                # print(f"Voluntary note result for chunk {idx} on pass {pass_number}: {add_note_result}")

        # Step 4: After three passes, prompt the agent for a final summary of the file.
        final_summary_prompt = f"Please provide a final summary of the file {file_path}."
        final_summary_response = self.chat(final_summary_prompt, {"configurable": {"thread_id": self.current_thread_id}})
        logging.info(f"Final summary from agent: {final_summary_response}")

        # Step 5: Write the final summary to voluntary memory.
        # add_final_note = add_voluntary_note(self.current_thread_id, f"Final summary for file {file_path}: {final_summary_response}")
        # print(f"Final summary note result: {add_final_note}")

        return final_summary_response
    # End of new code for file reading and chunking.

# ============================================================================
# End of AgentManager class modifications
# ============================================================================

# Chat with the agent using some sample conversation items
conversation_items = [
    "hi! I'm john, and you are bob",
    "what's my name?",
    "what's your name?",
    "please write our names to the system message",
    "i like the celtics!",
    "i like how much they win",
    "what's my name?",
    "which NFL team do you think I like?",
    "i like the patriots!",
    "what's the weather in san francisco?",
    "which are the coolest cities?",
    "please store the fact that i like the celtics in your voluntary memory",
    "what's in your voluntary memory?",
    "what's your name?"
]
thread_id = generate_thread_id()
# thread_id = "a029d1e8-f251-4b06-812f-46e6220e6d8b"
config = {"configurable": {"thread_id": thread_id}}

agent = AgentManager()
# Register the agent with its thread ID.
register_agent(thread_id, agent)
embedder = OpenAIEmbedding(model="text-embedding-ada-002")
text = "Testing OpenAI embedding generation."
embedding = embedder.embed(text)
#print(embedding)

# print(agent.model.invoke("what's the weather in sf?").tool_calls)
# try:
#     for item in conversation_items:
#         agent.conversation(item, config)
# except Exception as e:
#     print("An exception occurred:")
#     traceback.print_exc(file=sys.stdout)
#     pdb.post_mortem()

# ============================================================================
# DEMONSTRATION OF THE NEW FILE READING FUNCTIONALITY
# ============================================================================

# Create a sample file if it doesn't exist to demonstrate the file-reading functionality.
example_file_path = "src/basic_agent.py"
# try:
#     with open(example_file_path, "w", encoding="utf-8") as f:
#         # Write sample text repeated multiple times to ensure multiple chunks.
#         f.write("This is a sample text file. " * 50)
# except Exception as e:
#     print(f"Error creating sample file: {e}")

# Now, call the new read_file method with desired chunk size and overlap.
# For example, chunk_size is 100 characters and overlap is 20 characters.
final_summary = agent.read_file(example_file_path, chunk_size=2500, overlap=500)
print("Final summary of the file:", final_summary)

stop_flag = False
while not stop_flag:
    user_input = input("Enter a message to the agent: ")
    if user_input == "/stop":
        stop_flag = True
        break
    response = agent.conversation(user_input, config)
    print(f"Agent response: {response}")

