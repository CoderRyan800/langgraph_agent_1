from langchain.schema import BaseMessage
from langchain.chains import SummarizationChain
from langchain.chat_models import ChatOpenAI
from langgraph.graph import AgentGraph, MemoryNode, LLMNode, SummarizationNode

from langchain.chains.summarize import load_summarize_chain
from langchain.llms import OpenAI  # or your chosen LLM

class SummarizingMemoryAgent:
    def __init__(self, model_name="gpt-4o", memory_limit=5, summarization_chunk=3):
        """
        Initializes the agent with memory management and summarization.
        Args:
            model_name (str): The LLM model to use (default: "gpt-4o").
            memory_limit (int): The maximum number of messages in memory before summarization.
            summarization_chunk (int): Number of oldest messages to summarize at a time.
        """
        self.memory_limit = memory_limit
        self.summarization_chunk = summarization_chunk
        
        # Initialize the LLM
        self.llm = ChatOpenAI(model_name=model_name)
        
        # Create memory node
        self.memory_node = MemoryNode(max_size=memory_limit)
        
        # Create summarization node
        self.summarization_node = SummarizationNode(self.llm, self.summarize_messages)
        
        # Create LangGraph agent graph
        self.graph = AgentGraph()
        self.graph.add_node("memory", self.memory_node)
        self.graph.add_node("summarization", self.summarization_node)
        
        # Connect memory to summarization
        self.graph.add_edge("memory", "summarization", self.check_memory_and_summarize)

    def summarize_messages(self, messages: list[BaseMessage]) -> str:
        """
        Summarizes a list of messages into a single summary.
        Args:
            messages (list[BaseMessage]): List of messages to summarize.
        Returns:
            str: Summary of the messages.
        """
        summarization_chain = SummarizationChain(llm=self.llm)
        return summarization_chain.run(messages)

    def check_memory_and_summarize(self):
        """
        Checks if the memory exceeds the limit and triggers summarization if needed.
        """
        if len(self.memory_node.messages) > self.memory_limit:
            # Get the N oldest messages
            old_messages = self.memory_node.messages[:self.summarization_chunk]
            
            # Summarize the oldest messages
            summary = self.summarization_node.process(old_messages)
            
            # Remove N oldest messages and add the summary back
            self.memory_node.messages = self.memory_node.messages[self.summarization_chunk:]
            self.memory_node.add_message(BaseMessage(text=summary))

    def add_message(self, message: str):
        """
        Adds a new message to memory and checks for summarization.
        Args:
            message (str): The message text to add.
        """
        self.memory_node.add_message(BaseMessage(text=message))
        self.check_memory_and_summarize()

    def get_memory(self):
        """
        Retrieves the current memory buffer.
        Returns:
            list[BaseMessage]: The memory buffer.
        """
        return self.memory_node.messages

# Example usage
if __name__ == "__main__":
    agent = SummarizingMemoryAgent(model_name="gpt-4o", memory_limit=50, summarization_chunk=10)
    
    # Add some messages to the agent
    for i in range(60):  # Simulate 60 incoming messages
        agent.add_message(f"This is message {i + 1}")

    # Print the current memory buffer
    memory = agent.get_memory()
    for msg in memory:
        print(msg.text)