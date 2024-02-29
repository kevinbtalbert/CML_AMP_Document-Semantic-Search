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
    collection_name="collection_name",
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
                            title="Enterprise Custom Knowledge Base Chatbot with Llama2",
                            description="This AI-powered assistant uses Cloudera DataFlow (NiFi) to scrape a website's sitemap and create a knowledge base. The information it provides as a response is context driven by what is available at the scraped websites. It uses Meta's open-source Llama2 model and the sentence transformer model all-mpnet-base-v2 to evaluate context and form an accurate response from the semantic search. It is fine tuned for questions stemming from topics in its knowledge base, and as such may have limited knowledge outside of this domain. As is always the case with prompt engineering, the better your prompt, the more accurate and specific the response.",
                            inputs=[gradio.Radio(['llama-2-13b-chat'], label="Select Model", value="llama-2-13b-chat"), gradio.Slider(minimum=0.01, maximum=1.0, step=0.01, value=0.5, label="Select Temperature (Randomness of Response)"), gradio.Radio(["50", "100", "250", "500", "1000"], label="Select Number of Tokens (Length of Response)", value="250"), gradio.Textbox(label="Topic Weight", placeholder="This field can be used to prioritize a topic weight."), gradio.Textbox(label="Question", placeholder="Enter your question here.")],
                            outputs=[gradio.Textbox(label="Llama2 Model Response"), gradio.Textbox(label="Context Data Source(s)")],
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
def get_responses(engine, temperature, token_count, topic_weight, question):
    if engine is "" or question is "" or engine is None or question is None:
        return "One or more fields have not been specified."
    if temperature is "" or temperature is None:
      temperature = 1
      
    if topic_weight is "" or topic_weight is None:
      topic_weight = None
      
    if token_count is "" or token_count is None:
      token_count = 100
    
    if os.getenv('VECTOR_DB').upper() == "MILVUS":
        # Load Milvus Vector DB collection
        vector_db_collection = Collection('cloudera_ml_docs')
        vector_db_collection.load()
    
    # Phase 1: Get nearest knowledge base chunk for a user question from a vector db
    if topic_weight: 
        vdb_question = "Topic: " + topic_weight + " Question: " + question
    else:
        vdb_question = question
        
    if os.getenv('VECTOR_DB').upper() == "MILVUS":
        context_chunk, sources = get_nearest_chunk_from_milvus_vectordb(vector_db_collection, vdb_question)
        vector_db_collection.release()
        
    if os.getenv('VECTOR_DB').upper() == "PINECONE":
        context_chunk, sources, score = get_nearest_chunk_from_pinecone_vectordb(index, vdb_question)

    if engine == "llama-2-13b-chat":
        # Phase 2a: Perform text generation with LLM model using found kb context chunk
        response = get_llama2_response_with_context(question, context_chunk, temperature, token_count, topic_weight)
    
    if os.getenv('VECTOR_DB').upper() == "PINECONE":
        return response, sources, score
    else:
        return response, sources


if __name__ == "__main__":
    main()
