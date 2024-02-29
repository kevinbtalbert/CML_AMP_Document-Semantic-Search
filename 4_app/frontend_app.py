import os
import gradio
from typing import Any, Union, Optional
from pydantic import BaseModel
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
import chromadb

COLLECTION_NAME = os.getenv("COLLECTION_NAME")
persistent_client = chromadb.PersistentClient()
collection = persistent_client.get_or_create_collection(COLLECTION_NAME)

 # create the open-source embedding function
embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

langchain_chroma = Chroma(
    client=persistent_client,
    collection_name=COLLECTION_NAME,
    embedding_function=embedding_function,
)

app_css = f"""
        .gradio-header {{
            color: white;
        }}
        .gradio-description {{
            color: white;
        }}

        #custom-logo {{
            text-align: center;
        }}
        .gr-interface {{
            background-color: rgba(255, 255, 255, 0.8);
        }}
        .gradio-header {{
            background-color: rgba(0, 0, 0, 0.5);
        }}
        .gradio-input-box, .gradio-output-box {{
            background-color: rgba(255, 255, 255, 0.8);
        }}
        h1 {{
            color: white; 
            font-family: -apple-system, BlinkMacSystemFont, sans-serif;
            font-size: large; !important;
        }}
"""

def main():
    # Configure gradio QA app 
    print("Configuring gradio app")
    
    demo = gradio.Interface(fn=get_responses,
                            title="Semantic Search with CML and Chroma DB",
                            description="This services leverages Chroma's vector database to search semantically similar documents to the user's input.",
                            inputs=[gradio.Slider(minimum=1, maximum=10, step=1, value=3, label="Select number of similar documents to return"), gradio.Radio(["Yes", "No"], label="Show full document extract", value="Yes"), gradio.Textbox(label="Question", placeholder="Enter your search here")],
                            outputs=[gradio.Textbox(label="Document Response"), gradio.Textbox(label="Data Source(s) and Page Reference")],
                            allow_flagging="never",
                            css=app_css)

    # Launch gradio app
    print("Launching gradio app")
    demo.launch(share=True,
                enable_queue=True,
                show_error=True,
                server_name='127.0.0.1',
                server_port=int(os.getenv('CDSW_APP_PORT')))
    print("Gradio app ready")

# Helper function for generating responses for the QA app
def get_responses(num_docs, full_doc_display, question):
    if num_docs is "" or question is "" or num_docs is None or question is None:
        return "One or more fields have not been specified."
    
    if full_doc_display is "" or full_doc_display is None:
      full_doc_display = "No"
           
    if full_doc_display == "Yes":
        doc_snippet, page, source = query_chroma_vectordb(question, num_docs)
    if full_doc_display == "No":
        page, source = query_chroma_vectordb(question, num_docs)
    
    return response, sources

def query_chroma_vectordb(query, full_doc_display, num_docs):
    docs = langchain_chroma.similarity_search(query)
    doc_snippet = []
    page = []
    source = []
    for i in range(1, num_docs + 1):
        doc_snippet = doc_snippet.append(docs[i].page_content)
        page = page.append(docs[i].metadata.page)
        source = source.append(docs[i].metadata.source)
        
    if full_doc_display == "Yes":
        return doc_snippet, page, source
    if full_doc_display == "No":
        return page, source
        
if __name__ == "__main__":
    main()
