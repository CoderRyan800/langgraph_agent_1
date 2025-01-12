from typing import Literal

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

class BasicAgent:
    def __init__(self, model_name="gpt-4o", temperature=0, messages_before_summary=6):
        self.memory = MemorySaver()
        self.model = ChatOpenAI(model=model_name, temperature=temperature)
        self.messages_before_summary = messages_before_summary
        self.app = self._create_workflow()

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
        if config is None:
            config = {"configurable": {"thread_id": "default"}}
        
        input_message = HumanMessage(content=message)
        return self.app.stream({"messages": [input_message]}, config, stream_mode="updates")

    def conversation(self, message: str, config: dict = None):
        for event in self.chat(message, config):
            print_update(event)

agent = BasicAgent()

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

for item in conversation_items:
    agent.conversation(item)

