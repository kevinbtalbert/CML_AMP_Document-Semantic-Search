import os
import gradio
from typing import Any, Union, Optional
from pydantic import BaseModel
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
import chromadb

chroma_client = chromadb.PersistentClient(path="/home/cdsw/chroma_db")

 # create the open-source embedding function
embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

langchain_chroma = Chroma(
    client=chroma_client,
    collection_name=os.getenv("COLLECTION_NAME"),
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
                            outputs=[gradio.Textbox(label="Data Source(s) and Page Reference"), gradio.Textbox(label="Document Response")],
                            allow_flagging="never",
                            css=app_css)

    # Launch gradio app
    print("Launching gradio app")
    demo.launch(share=True,
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
           
    source, doc_snippet = query_chroma_vectordb(question, full_doc_display, num_docs)
    return source, doc_snippet


def query_chroma_vectordb(query, full_doc_display, num_docs):
    docs = langchain_chroma.similarity_search(query)
    doc_snippet = []
    source_info = []
    
    # Gather data into lists
    for i in range(num_docs):
        if full_doc_display == "Yes":
            doc_snippet.append("Doc {}: Relevant content: {}".format(i+1, docs[i].page_content))
        source_info.append("Doc {}: Source link: {}, Page: {}".format(i+1, docs[i].metadata['source'], docs[i].metadata['page']))

    # Format the output as strings with newlines
    doc_snippet_str = "\n".join(doc_snippet) if full_doc_display == "Yes" else "Show document response turned off."
    source_info_str = "\n".join(source_info)
    
    return source_info_str, doc_snippet_str
        
if __name__ == "__main__":
    main()
