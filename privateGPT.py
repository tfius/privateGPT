from dotenv import load_dotenv
import os
import langchain

from langchain.chains import RetrievalQA
from langchain.embeddings import LlamaCppEmbeddings
from langchain.cache import GPTCache
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.vectorstores import Chroma
from langchain.llms import GPT4All, LlamaCpp, OpenAI
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.retrievers.self_query.base import SelfQueryRetriever

from gptcache.adapter.api import init_similar_cache


load_dotenv()

llama_embeddings_model = os.environ.get("LLAMA_EMBEDDINGS_MODEL")
persist_directory = os.environ.get('PERSIST_DIRECTORY')

model_type = os.environ.get('MODEL_TYPE')
model_path = os.environ.get('MODEL_PATH')

model_n_ctx = os.environ.get('MODEL_N_CTX')
model_temp = os.environ.get('MODEL_TEMP')
model_stop = os.environ.get('MODEL_STOP').split(",")
model_repeat_penalty = os.environ.get('MODEL_REPEAT_PENALTY')

from constants import CHROMA_SETTINGS

# Helper function for printing docs
def pretty_print_docs(docs):
    print(f"\n{'-' * 100}\n".join([f"Document {i+1}:\n\n" + d.page_content for i, d in enumerate(docs)]))

def main():
    try:
        llama = LlamaCppEmbeddings(model_path=llama_embeddings_model, n_ctx=model_n_ctx)
        db = Chroma(persist_directory=persist_directory, embedding_function=llama, client_settings=CHROMA_SETTINGS)
        #retriever = db.as_retriever()
        retriever = db.as_retriever(search_type="mmr") # other type of similarity search
        
        # check  https://python.langchain.com/en/latest/modules/indexes/retrievers/examples/chroma_self_query_retriever.html
        # self_query_retriever = SelfQueryRetriever.from_llm(llm, db, document_content_description, metadata_field_info, verbose=True)
        
        # Prepare the LLM
        callbacks = [StreamingStdOutCallbackHandler()]
        match model_type:
            case "LlamaCpp":
                llm = LlamaCpp(model_path=model_path, n_ctx=model_n_ctx, repeat_penalty = model_repeat_penalty, temperature=model_temp, stop=model_stop, callbacks=callbacks, verbose=False)
            case "GPT4All":
                llm = GPT4All(model=model_path, n_ctx=model_n_ctx, backend='gptj', callbacks=callbacks, verbose=False)
            case _default:
                print(f"Model {model_type} not supported!")
                exit;

        compressor = LLMChainExtractor.from_llm(llm)
        compression_retriever = ContextualCompressionRetriever(base_compressor=compressor, base_retriever=retriever)
        
        # langchain.llm_cache = GPTCache(init_func=lambda cache: init_similar_cache(cache_obj=cache))
        #qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True)
        qa = RetrievalQA.from_chain_type(llm=llm, chain_type="map_reduce", retriever=retriever, return_source_documents=True)
        # Interactive questions and answers
        while True:
            query = input("\nEnter a query: ")
            if query == "exit":
                break
            
            if query.strip() == "":
                print("error: query empty!")
                continue
            
            try:               
                # Get the answer from the chain
                res = qa(query)    
                answer, docs = res['result'], res['source_documents']

                # Print the result
                print("\n\n> Question:")
                print(query)
                print("\n> Answer:")
                print(answer)
                
                # Print the relevant sources used for the answer
                for document in docs:
                    print("\n> " + document.metadata["source"] + ":")
                    print(document.page_content)
             
                # get docs from compression retriever
                if False:
                    print("\n> Get docs from compression retriever:")
                    compressed_docs = compression_retriever.get_relevant_documents(query)
                    pretty_print_docs(compressed_docs)
    
            except Exception as e:
                print(f"Query error occurred: {str(e)}")
                
    except Exception as e:
        print(f"An error occurred in main: {str(e)}")
        
if __name__ == "__main__":
    main()


# Installing collected packages: requests, numpy, colorama, cachetools, tqdm, openai, gptcache
#   WARNING: The script f2py.exe is installed in 'C:\Users\TEX\AppData\Roaming\Python\Python310\Scripts' which is not on PATH.
#   Consider adding this directory to PATH or, if you prefer to suppress this warning, use --no-warn-script-location.
#   WARNING: The script tqdm.exe is installed in 'C:\Users\TEX\AppData\Roaming\Python\Python310\Scripts' which is not on PATH.
#   Consider adding this directory to PATH or, if you prefer to suppress this warning, use --no-warn-script-location.
#   WARNING: The script openai.exe is installed in 'C:\Users\TEX\AppData\Roaming\Python\Python310\Scripts' which is not on PATH.
#   Consider adding this directory to PATH or, if you prefer to suppress this warning, use --no-warn-script-location.
#   WARNING: The script gptcache_server.exe is installed in 'C:\Users\TEX\AppData\Roaming\Python\Python310\Scripts' which is not on PATH.
#   Consider adding this directory to PATH or, if you prefer to suppress this warning, use --no-warn-script-location.
# ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
# open-clip-torch 1.3.0 requires torch>=1.9, which is not installed.
# open-clip-torch 1.3.0 requires torchvision, which is not installed.
# matplotlib 3.5.3 requires pillow>=6.2.0, which is not installed.
# lpips 0.1.4 requires torch>=0.4.0, which is not installed.
# lpips 0.1.4 requires torchvision>=0.2.1, which is not installed.
# jina 3.7.6 requires websockets, which is not installed.
# guided-diffusion-sdk 1.0.0 requires torch, which is not installed.
# Successfully installed cachetools-5.3.0 colorama-0.4.6 gptcache-0.1.23 numpy-1.24.3 openai-0.27.6 requests-2.30.0 tqdm-4.65.0