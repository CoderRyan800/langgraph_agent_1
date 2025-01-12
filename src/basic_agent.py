from langchain.chains.summarize import load_summarize_chain
from langchain.chat_models import ChatOpenAI
from langchain.schema import Document

# Initialize the language model
llm = ChatOpenAI(model="gpt-4", temperature=0)

# Load the summarization chain
chain = load_summarize_chain(llm, chain_type="stuff")

# Prepare your documents
documents = [
    Document(page_content="Apples are red."),
    Document(page_content="Blueberries are blue."),
    Document(page_content="Bananas are yellow."),
]

# Generate the summary
summary = chain({"input_documents": documents})
print(summary["output_text"])