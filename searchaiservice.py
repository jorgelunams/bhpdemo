from azure.search.documents.indexes import SearchIndexClient
from dotenv import load_dotenv 
import Gen2Services as g2s
from azure.identity import DefaultAzureCredential 
from azure.core.credentials import AzureKeyCredential
import os
import json
from azure.search.documents import SearchClient  
from azure.search.documents.models import VectorizedQuery, VectorQuery 
from azure.core.credentials import AzureKeyCredential  
import os  
import tiktoken

from azure.search.documents import SearchClient    
from langchain.text_splitter import TokenTextSplitter
import uuid
from azure.search.documents.models import VectorQuery  
from azure.search.documents.indexes.models import (
    SimpleField,
    SearchFieldDataType,
    SearchableField,
    SearchField,
    VectorSearch,
    HnswAlgorithmConfiguration,
    VectorSearchProfile,
    SemanticConfiguration,
    SemanticPrioritizedFields,
    SemanticField,
    SemanticSearch,
    SearchIndex
)
text_splitter = TokenTextSplitter(chunk_size=2048, chunk_overlap=205)  
 

def create_search_index_manual_solutions(index_name):

# Create a search index
    endpoint = os.environ["AZURE_SEARCH_SERVICE_ENDPOINT"]
    credential = AzureKeyCredential(os.environ["AZURE_SEARCH_API_KEY"]) if len(os.environ["AZURE_SEARCH_API_KEY"]) > 0 else DefaultAzureCredential()
    index_client = SearchIndexClient(
    endpoint=endpoint, credential=credential)
    fields = [
        SimpleField(name="id", type=SearchFieldDataType.String, key=True, sortable=True, filterable=True, facetable=True),
        SearchableField(name="problema", type=SearchFieldDataType.String),
        SearchableField(name="tipo_problema", type=SearchFieldDataType.String),  
        SearchableField(name="soluciones", type=SearchFieldDataType.String),  
        SearchableField(name="paginas", type=SearchFieldDataType.String),  
        SearchableField(name="nombre_manual", type=SearchFieldDataType.String),    
        SearchableField(name="titulo", type=SearchFieldDataType.String),  
        SearchableField(name="subtitulo", type=SearchFieldDataType.String),
        SearchableField(name="nombre_opcion", type=SearchFieldDataType.String),
        SearchableField(name="texto_paginas", type=SearchFieldDataType.String), 
        
        SearchField(name="problemaVector", type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                searchable=True, vector_search_dimensions=1536, vector_search_profile_name="myHnswProfile"),
        SearchField(name="solucionesVector", type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                searchable=True, vector_search_dimensions=1536, vector_search_profile_name="myHnswProfile"),  
       
        
]
    
    # Configure the vector search configuration  
    vector_search = VectorSearch(
        algorithms=[
            HnswAlgorithmConfiguration(
                name="myHnsw"
            )
        ],
        profiles=[
            VectorSearchProfile(
                name="myHnswProfile",
                algorithm_configuration_name="myHnsw",
            )
        ]
    ) 
    semantic_config = SemanticConfiguration(
        name="manual-semantic-config",
        prioritized_fields=SemanticPrioritizedFields(
            title_field=SemanticField(field_name="paginas"),
            description_fields=[SemanticField(field_name="pregunta")]       
        )
    )
    # Create the semantic settings with the configuration
    semantic_search = SemanticSearch(configurations=[semantic_config])

    # Create the search index with the semantic settings
    index = SearchIndex(name=index_name, fields=fields,
                        vector_search=vector_search, semantic_search=semantic_search)
    result = index_client.create_or_update_index(index)
    print(f' {result.name} created') 
    

def create_search_index_kb_solutions(index_name):

# Create a search index
    endpoint = os.environ["AZURE_SEARCH_SERVICE_ENDPOINT"]
    credential = AzureKeyCredential(os.environ["AZURE_SEARCH_API_KEY"]) if len(os.environ["AZURE_SEARCH_API_KEY"]) > 0 else DefaultAzureCredential()
    index_client = SearchIndexClient(
    endpoint=endpoint, credential=credential)
    fields = [
        SimpleField(name="id", type=SearchFieldDataType.String, key=True, sortable=True, filterable=True, facetable=True),
        SearchableField(name="problema", type=SearchFieldDataType.String),
        SearchField(name="chunk_id", type=SearchFieldDataType.String, sortable=True, filterable=True, facetable=False, searchable=False),
        SearchableField(name="tipo_problema", type=SearchFieldDataType.String, sortable=True, filterable=True, facetable=False),  
        SearchableField(name="soluciones", type=SearchFieldDataType.String , sortable=False, filterable=True, facetable=False),  
        SearchableField(name="paginas", type=SearchFieldDataType.String , sortable=False, filterable=True, facetable=False),  
        SearchableField(name="nombre_manual", type=SearchFieldDataType.String, sortable=True, filterable=True, facetable=False),  
        SearchableField(name="referencia_url", type=SearchFieldDataType.String, sortable=False, filterable=True, facetable=False),      
        SearchableField(name="titulo", type=SearchFieldDataType.String ,sortable=True, filterable=True, facetable=False)  ,  
        SearchableField(name="subtitulo", type=SearchFieldDataType.String ,sortable=True, filterable=True, facetable=False),
        SearchableField(name="nombre_opcion", type=SearchFieldDataType.String, sortable=True, filterable=True, facetable=False),
        SearchableField(name="texto_paginas", type=SearchFieldDataType.String, sortable=True, filterable=True, facetable=False),
        SearchableField(name="texto_extraido", type=SearchFieldDataType.String, sortable=True, filterable=True, facetable=False),  
        SearchableField(name="entidades", type=SearchFieldDataType.String, sortable=False, filterable=True, facetable=False), 
        SearchableField(name="totalTokens", type=SearchFieldDataType.Int32, sortable=True, filterable=True, facetable=False), 
        SearchableField(name="preguntas_contexto", type=SearchFieldDataType.String, sortable=True, filterable=True, facetable=False), 
        SearchField(name="problemaVector", type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                searchable=True, vector_search_dimensions=1536, vector_search_profile_name="myHnswProfile"),
        SearchField(name="solucionesVector", type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                searchable=True, vector_search_dimensions=1536, vector_search_profile_name="myHnswProfile"), 
        SearchField(name="preguntasVector", type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                searchable=True, vector_search_dimensions=1536, vector_search_profile_name="myHnswProfile"), 
        SearchField(name="textoVector", type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                searchable=True, vector_search_dimensions=1536, vector_search_profile_name="myHnswProfile"), 
        SearchField(name="textoExtraidoVector", type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                searchable=True, vector_search_dimensions=1536, vector_search_profile_name="myHnswProfile"), 
       
        
        ]
    
    # Configure the vector search configuration  
    vector_search = VectorSearch(
        algorithms=[
            HnswAlgorithmConfiguration(
                name="myHnsw"
            )
        ],
        profiles=[
            VectorSearchProfile(
                name="myHnswProfile",
                algorithm_configuration_name="myHnsw",
            )
        ]
    ) 
    semantic_config = SemanticConfiguration(
    name="manual-semantic-config",
    prioritized_fields=SemanticPrioritizedFields(
        title_field=SemanticField(field_name="paginas"),
        content_field=SemanticField(field_name="pregunta")       
        )
    )
    # Create the semantic settings with the configuration
    semantic_search = SemanticSearch(configurations=[semantic_config])

    # Create the search index with the semantic settings
    index = SearchIndex(name=index_name, fields=fields,
                        vector_search=vector_search, semantic_search=semantic_search)
    result = index_client.create_or_update_index(index)
    print(f' {result.name} created') 
    
    
def insert_document(index_name):    
    import json    
    from azure.core.exceptions import HttpResponseError    
    import os    
   
    from azure.search.documents import SearchClient    
    credential = AzureKeyCredential(os.environ["AZURE_SEARCH_API_KEY"]) if len(os.environ["AZURE_SEARCH_API_KEY"]) > 0 else DefaultAzureCredential()
   
    # Download the manual with embeddings  
    connection_string = os.getenv('CONNECTION_STRING')    
    file_path = "KBDEV/ManualsDocuments/ManualMobilEmbeded.json"  
    manual = g2s.download_content_as_json(connection_string, "telefonicadata", file_path)   
    endpoint = os.environ["AZURE_SEARCH_SERVICE_ENDPOINT"]    
    credential = AzureKeyCredential(os.environ["AZURE_SEARCH_API_KEY"]) if len(os.environ["AZURE_SEARCH_API_KEY"]) > 0 else DefaultAzureCredential()    
    search_client = SearchClient(endpoint=endpoint, index_name=index_name, credential=credential)     
    for item in manual:
     item['nombre_manual'] = item.pop('nombre-manual')
     

    for item in manual:
       item['id'] = str(uuid.uuid4())
    documents = manual  
    try:    
        result = search_client.upload_documents(documents)    
        print(f"Uploaded {len(documents)} documents")    
    except HttpResponseError as e:    
        print(f"An error occurred: {e}")    
        print(f"Uploaded {len(documents)} documents")  
        
  
import os  
import uuid  
from azure.core.credentials import AzureKeyCredential  
from azure.search.documents import SearchClient  
from azure.search.documents.indexes.models import SearchIndex  
from openai import AzureOpenAI  
  
def generate_embedding(text, client, embedding_model_name="text-embedding-ada-002"):  
    """  
    Generates an embedding for the given text using Azure OpenAI Service.  
    """  
    try:  
        response = client.embeddings.create(input=text, model=embedding_model_name)  
        embedding = response.data[0].embedding  
        return embedding  
    except Exception as e:  
        print(f"Failed to generate embedding: {e}")  
        return []  
  
def insert_manual_with_embeddings(index_name):  
    """  
    Generates embeddings for specified fields in each document, then uploads the documents to Azure Cognitive Search.  
    """  
    # Azure OpenAI setup  
    from azure.core.exceptions import HttpResponseError    
    import os    
    openai_client = AzureOpenAI(  
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),  
        api_version="2024-02-01",  
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")  
    )  
        
     
    from azure.search.documents import SearchClient    
    credential = AzureKeyCredential(os.environ["AZURE_SEARCH_API_KEY"]) if len(os.environ["AZURE_SEARCH_API_KEY"]) > 0 else DefaultAzureCredential()
   
    # Download the manual with embeddings  
    connection_string = os.getenv('CONNECTION_STRING')    
    file_path = "KBDEV/ManualsDocuments/ProblemasFuncionamientoPR.json"  
    manual = g2s.download_content_as_json(connection_string, "telefonicadata", file_path)   
    # Azure Cognitive Search setup  
    endpoint = os.environ["AZURE_SEARCH_SERVICE_ENDPOINT"]  
    credential = AzureKeyCredential(os.environ["AZURE_SEARCH_API_KEY"])  
    search_client = SearchClient(endpoint=endpoint, index_name=index_name, credential=credential)  
  
    # Process documents  
    processed_documents = []  
    for doc in manual:  
        doc_id = str(uuid.uuid4())  
        problema_embedding = generate_embedding(doc["problema"], openai_client)  
        soluciones_embedding = generate_embedding(doc["soluciones"], openai_client)  
          
        processed_doc = {  
            "id": doc_id,  
            "problema": doc["problema"],  
            "tipo_problema": doc["tipo_problema"],  
            "soluciones": doc["soluciones"],  
            "paginas": doc.get("paginas", ""),  
            "nombre_manual": doc.get("nombre_manual", ""),  
            "titulo": doc.get("titulo", ""),  
            "subtitulo": doc.get("subtitulo", ""),  
            "nombre_opcion": doc.get("nombre_opcion", ""),  
            "texto_paginas": doc.get("texto_paginas", ""),  
            "problemaVector": problema_embedding,  
            "solucionesVector": soluciones_embedding  
        }  
          
        processed_documents.append(processed_doc)  
      
    # Upload documents  
    try:  
        result = search_client.upload_documents(processed_documents)  
        print(f"Uploaded {len(processed_documents)} documents successfully.")  
    except Exception as e:  
        print(f"Failed to upload documents: {e}")  
  
def insert_kb_with_embeddings(index_name):  
    """  
    Generates embeddings for specified fields in each document, then uploads the documents to Azure Cognitive Search.  
    """  
    # Azure OpenAI setup  
    from azure.core.exceptions import HttpResponseError    
    import os    
    openai_client = AzureOpenAI(  
        api_key=os.getenv("AZURE_OPENAI_API_KEY4O"),  
        api_version="2024-02-01",  
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT4O")  
    )  
        
     
    from azure.search.documents import SearchClient    
    credential = AzureKeyCredential(os.environ["AZURE_SEARCH_API_KEY"]) if len(os.environ["AZURE_SEARCH_API_KEY"]) > 0 else DefaultAzureCredential()
   
    # Download the manual with embeddings  
    connection_string = os.getenv('CONNECTION_STRING')    
    file_path = "KBDEV/ManualsDocuments/ManualAcademiaAtencionesTecnicas.json"  
    manual = g2s.download_content_as_json(connection_string, "telefonicadata", file_path)   
    manual_json_string = json.dumps(manual)
    # Azure Cognitive Search setup  
    endpoint = os.environ["AZURE_SEARCH_SERVICE_ENDPOINT"]  
    credential = AzureKeyCredential(os.environ["AZURE_SEARCH_API_KEY"])  
    search_client = SearchClient(endpoint=endpoint, index_name=index_name, credential=credential)  
  
    # Process documents  
    chunk_id = 0
    processed_documents = [] 
    ofProceed = True 
    
      
    # Upload documents  
    try:  
        processed_documents = insert_documents_main(manual, openai_client , "ManualAcademiaAtencionesTecnicas" ) 
        result = search_client.upload_documents(documents=processed_documents)  
        print(f"Uploaded {len(processed_documents)} documents successfully.")  
    except Exception as e:  
        print(f"Failed to upload documents: {e}")  


def insert_manual_roaming_chunking_with_embeddings(index_name):  
    """  
    Generates embeddings for specified fields in each document, then uploads the documents to Azure Cognitive Search.  
    """  
    # Azure OpenAI setup  
    from azure.core.exceptions import HttpResponseError    
    import os    
    openai_client = AzureOpenAI(  
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),  
        api_version="2024-02-01",  
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")  
    )  
        
     
    from azure.search.documents import SearchClient    
    credential = AzureKeyCredential(os.environ["AZURE_SEARCH_API_KEY"]) if len(os.environ["AZURE_SEARCH_API_KEY"]) > 0 else DefaultAzureCredential()
   
    # Download the manual with embeddings  
    connection_string = os.getenv('CONNECTION_STRING')    
    file_path = "KBDEV/ManualsDocuments/RoamingPage110PR.json"  
    manual = g2s.download_content_as_json(connection_string, "telefonicadata", file_path)   
    manual_json_string = json.dumps(manual)
    # Azure Cognitive Search setup  
    endpoint = os.environ["AZURE_SEARCH_SERVICE_ENDPOINT"]  
    credential = AzureKeyCredential(os.environ["AZURE_SEARCH_API_KEY"])  
    search_client = SearchClient(endpoint=endpoint, index_name=index_name, credential=credential)  
  
    # Process documents  
    chunk_id = 0
    processed_documents = []    
    for doc_list in manual:  
        for doc in doc_list:  
            try:  
                doc_id = str(uuid.uuid4())  
                problema = doc["problema"]  
                chunks = text_splitter.split_text(doc["page_text"])
                
                print(f"Number of chunks: {len(chunks)}")
                for chunk in chunks: 
                    
                    text_embedding = generate_embedding(chunk, openai_client)  
                    soluciones_found = doc["soluciones"] 
                    Preguntas = ' '.join([json.dumps(pregunta, ensure_ascii=False) for pregunta in doc["PreguntasContexto"]])   
                    preguntas_embedding = generate_embedding(Preguntas, openai_client)  
                    chunk_id += 1 
                    Entidades = ' '.join([json.dumps(pregunta, ensure_ascii=False) for pregunta in doc["entidades"]])  
                    processed_doc = {  
                            "id": doc_id,  
                            "problema": doc["problema"],  
                            "tipo_problema": doc["tipo_problema"],  
                        ##    "soluciones": doc["soluciones"],  
                            "paginas": doc.get("paginas", ""),  
                            "nombre_manual": doc.get("nombre_manual", ""),  
                            "titulo": doc.get("titulo", ""),  
                            "subtitulo": doc.get("subtitulo", ""),  
                            "nombre_opcion": doc.get("nombre_opcion", ""),  
                            "texto_paginas": doc.get("page_text", ""),   
                            "referencia_url": doc.get("referencia_url", ""),   
                            "preguntasVector": preguntas_embedding,
                            "preguntas_contexto": Preguntas,
                            "textoVector": text_embedding,
                            "entidades": Entidades, 
                            "chunk_id": str(chunk_id)
                        }  
                    
                    processed_documents.append(processed_doc)  
    
            except KeyError as e:  
                     print(f"Key {str(e)} not found in dictionary: {doc}")   
    # Upload documents  
    try:  
        result = search_client.upload_documents(processed_documents)  
        print(f"Uploaded {len(processed_documents)} documents successfully.")  
    except Exception as e:  
        print(f"Failed to upload documents: {e}")  
  


def VectorQASearch(index_name, question):    
    from azure.search.documents.models import VectorizedQuery  
   
    
    credential = AzureKeyCredential(os.environ["AZURE_SEARCH_API_KEY"]) if len(os.environ["AZURE_SEARCH_API_KEY"]) > 0 else DefaultAzureCredential()
    endpoint = os.environ["AZURE_SEARCH_SERVICE_ENDPOINT"]   
    query = question
    from openai import AzureOpenAI
    client = AzureOpenAI(
    api_key = os.getenv("AZURE_OPENAI_API_KEY"),  
    api_version = "2024-02-01",
    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    )
    embedding_model_name = "text-embedding-ada-002"
    embedding = client.embeddings.create(input=query, model=embedding_model_name).data[0].embedding
    vector_query = VectorizedQuery(vector=embedding, k_nearest_neighbors=3, fields="preguntaVector")
    search_client = SearchClient(endpoint=endpoint, index_name=index_name, credential=credential)
    results = search_client.search(  
        search_text=None,  
        vector_queries= [vector_query],
        select=["pregunta", "respuesta", "pagina"],
    )  
           
    results_list = [{'pagina': result['pagina'], 'respuesta': result['respuesta'], 'score': result['@search.score']} for result in results]
    return results_list

def VectorManualSolutionsSearch(index_name, question):    
    from azure.search.documents.models import VectorizedQuery   
    from openai import AzureOpenAI
    from azure.core.exceptions import AzureError

    try:
        credential = AzureKeyCredential(os.environ["AZURE_SEARCH_API_KEY"]) if len(os.environ["AZURE_SEARCH_API_KEY"]) > 0 else DefaultAzureCredential()
        endpoint = os.environ["AZURE_SEARCH_SERVICE_ENDPOINT"]   
        query = question
        client = AzureOpenAI(
            api_key = os.getenv("AZURE_OPENAI_API_KEY"),  
            api_version = "2024-02-01",
            azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        )
        embedding_model_name = "text-embedding-ada-002"
        embedding = client.embeddings.create(input=query, model=embedding_model_name).data[0].embedding
        vector_query = VectorizedQuery(vector=embedding, k_nearest_neighbors=2, fields="preguntasVector")
        vector_query_preguntas = VectorizedQuery(vector=embedding, k_nearest_neighbors=2, fields="preguntasVector")
        vector_query_texto = VectorizedQuery(vector=embedding, k_nearest_neighbors=4, fields="textoVector")
        search_client = SearchClient(endpoint=endpoint, index_name=index_name, credential=credential)  
        results = search_client.search(  
            search_text=None,   
            vector_queries= [  vector_query_texto ],
            select=[   "texto_paginas", "texto_paginas", "paginas",  "titulo", "subtitulo"],
           
        )  
               
        results_list = [{ 'titulo': result['titulo'],
                         'subtitulo': result['subtitulo'], 'paginas': result['paginas'],
                         'contexto:': result['texto_paginas'],
                         'score': result['@search.score']} for result in results]
        return results_list

    except AzureError as e:
        print(f"An Azure error occurred: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")
        



def VectorSemanticSearch(index_name, question):    
    from azure.search.documents.models import VectorizedQuery   
    from openai import AzureOpenAI
    from azure.core.exceptions import AzureError
    from azure.core.credentials import AzureKeyCredential
    from azure.search.documents import SearchClient
    from azure.search.documents.indexes.models import (
        SearchIndex,
        SearchField,
        SearchFieldDataType,
        SimpleField,
        SearchableField,
        VectorSearch,
        VectorSearchProfile,
        HnswAlgorithmConfiguration,
    )


    try:
        credential = AzureKeyCredential(os.environ["AZURE_SEARCH_API_KEY"]) if len(os.environ["AZURE_SEARCH_API_KEY"]) > 0 else DefaultAzureCredential()
        service_endpoint = os.environ["AZURE_SEARCH_SERVICE_ENDPOINT"]
         
        key = os.environ["AZURE_SEARCH_API_KEY"]

        search_client = SearchClient(service_endpoint, index_name, AzureKeyCredential(key))
        query = question
        client = AzureOpenAI(
            api_key = os.getenv("AZURE_OPENAI_API_KEY4O"),  
            api_version = "2024-02-01",
            azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT4O")
        )
        embedding_model_name = "text-embedding-ada-002"
        embedding = client.embeddings.create(input=query, model=embedding_model_name).data[0].embedding
       
       
        vector_query_texto = VectorizedQuery(vector=embedding, k_nearest_neighbors=3, fields="contentVector")
        search_client = SearchClient(service_endpoint, index_name, AzureKeyCredential(key))
        semantic_config_name = "manual-semantic-config"  # Set your semantic configuration name  
        
        
  
       
        # Perform the hybrid search  
        results = search_client.search(  
            search_text=query,  
            vector_queries=[   vector_query_texto],  
            ##filter=f"nombre_manual eq '{manualName}'",  
            select=["content", "numero_de_pagina", "titulo", "Sumario_Ejecutivo"],  
            top=10,  # Number of combined results to retrieve  
            semantic_configuration_name=semantic_config_name  # Specify the semantic configuration name  
        )  
         
        
        # Process and display the results  
        results_list = [  
            {  
                'titulo': result['titulo'],  
                'numero_de_pagina': result['numero_de_pagina'],  
                'sumario_ejecutivo': result['Sumario_Ejecutivo'],  
                'contexto': result['content'],  
                'score': result['@search.score']  
            } for result in results  
        ]  
   
        
        results_list = sorted(results_list, key=lambda x: x['score'], reverse=True)
        top_3_results = results_list[:3]
        return top_3_results

    except AzureError as e:
        print(f"An Azure error occurred: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")
        

import requests  
  
def bing_search(query,  custom_config_id=None, count=10, offset=0, mkt='en-US'):  
    
    subscription_key = ""
    """  
    Perform a Bing search using the Bing Search API.  
  
    Parameters:  
        query (str): The search query.  
        subscription_key (str): Your Bing Search API subscription key.  
        custom_config_id (str): Your custom configuration ID (optional).  
        count (int): The number of search results to return (default is 10).  
        offset (int): The number of search results to skip (default is 0).  
        mkt (str): The market to search in (default is 'en-US').  
  
    Returns:  
        dict: The search results in JSON format.  
    """  
    endpoint = "https://api.bing.microsoft.com/v7.0/search"  
    headers = {"Ocp-Apim-Subscription-Key": subscription_key}  
    params = {  
        "q": query,  
        "count": count,  
        "offset": offset,  
        "mkt": mkt  
    }  
  
    if custom_config_id:  
        params["customconfig"] = custom_config_id  
  
    response = requests.get(endpoint, headers=headers, params=params)  
    response.raise_for_status()  # Raise an exception for HTTP errors  
    
  
    results = response.json()  
    try:  
         
        formatted_results = []  
  
        for i, result in enumerate(results["webPages"]["value"], start=1):  
            formatted_result = {  
                "name": result['name'],  
                "url": result['url'],  
                "snippet": result['snippet']  
            }  
            formatted_results.append(formatted_result)  
          
        return json.dumps(formatted_results, indent=4)  # Convert list to JSON string with pretty printing  
  
    except Exception as e:  
        error_message = {"error": str(e)}  
        return json.dumps(error_message)  # Return error message as JSON string  
    
def num_tokens_from_string(content, model="gpt-3.5-turbo-0301"):
    encoding = tiktoken.encoding_for_model(model)
    num_tokens = 0
    num_tokens += 4  # every message follows <im_start>{role/name}\n{content}<im_end>\n
    num_tokens += len(encoding.encode(content))
    num_tokens += 2  # every reply is primed with <im_start>assistant
    return num_tokens

################

########################

import uuid  
import json  
  
def process_document(doc, chunk_id, openai_client, nombredocumento, text_embedding=None, chunk_text=None):
    
    problema = doc["problema"]  

    totalTokensText = num_tokens_from_string(chunk_text if chunk_text else doc["page_text"])  
    problema_embedding = generate_embedding(doc["problema"], openai_client)  
    soluciones_embedding = generate_embedding(doc["soluciones"], openai_client)   
    texto_extraido_embedding = generate_embedding(doc["texto_extraido"], openai_client)   
        
    if text_embedding is None:  
        text_embedding = generate_embedding(doc["page_text"], openai_client) 
    # Concatenate all questions into a single string  
    Preguntas = ' '.join([json.dumps(pregunta, ensure_ascii=False) for pregunta in doc["PreguntasContexto"]])  
    preguntas_embedding = generate_embedding(Preguntas, openai_client)  
      
    # Convert entidades dictionary into a string  
    Entidades = ' '.join([json.dumps(pregunta, ensure_ascii=False) for pregunta in doc["entidades"]])  
    Solutions = ' '.join([json.dumps(pregunta, ensure_ascii=False) for pregunta in doc["soluciones"]])  
    totalTokensText = str(totalTokensText)
        
    import uuid

    doc_id = str(uuid.uuid4())
    processed_doc = {  
        "id": doc_id,  
        "problema": doc["problema"],  
        "tipo_problema": doc["tipo_problema"],  
       ## "soluciones":Solutions,  
        "paginas": doc.get("paginas", ""),  
        "nombre_manual": nombredocumento,  
        "titulo": doc.get("titulo", ""),  
        "subtitulo": doc.get("subtitulo", ""),  
        "nombre_opcion": doc.get("nombre_opcion", ""),  
        "texto_paginas": doc.get("page_text", ""),  
        "problemaVector": problema_embedding,  
        ##"referencia_url": doc.get("referencia_url", ""),  
        "solucionesVector": soluciones_embedding,  
        "preguntasVector": preguntas_embedding,  
        "preguntas_contexto": Preguntas,  
        "textoVector": text_embedding,  
         "entidades": Entidades,  
        "totalTokens": totalTokensText,
        "textoExtraidoVector": texto_extraido_embedding,
        "texto_extraido": doc["texto_extraido"],
        "chunk_id": str(chunk_id)  
    }  
  
    return processed_doc  


def generate_doc_id(nombredocumento, titulo, subtitulo, nombre_opcion):
    # Remove special characters
    import re
    nombredocumento = remove_special_characters(nombredocumento)
    titulo = remove_special_characters(titulo)
    subtitulo = remove_special_characters(subtitulo)
    nombre_opcion = remove_special_characters(nombre_opcion)

    # Replace spaces with nothing
    nombredocumento = nombredocumento.replace(" ", "")
    titulo = titulo.replace(" ", "")
    subtitulo = subtitulo.replace(" ", "")
    nombre_opcion = nombre_opcion.replace(" ", "")

    # Concatenate the components with underscores
    doc_id = f"{nombredocumento}_{titulo}_{subtitulo}_{nombre_opcion}"

    # Remove forbidden characters
    forbidden_chars = "#?;'\"&"
    for char in forbidden_chars:
        doc_id = doc_id.replace(char, "")

    # Remove any other characters that are not letters, digits, underscore, dash, or equal sign
    doc_id = re.sub(r'[^a-zA-Z0-9_=().-]', '', doc_id)

    # Truncate to 1024 characters if necessary
    if len(doc_id) > 1024:
        doc_id = doc_id[:1024]

    return doc_id

 
def remove_special_characters(text):
    # Define the special characters you want to remove
    special_chars = "áéíóúüñÁÉÍÓÚÜÑ"
    
    # Define the characters you want to replace them with
    replace_chars = "aeiouunAEIOUUN"
    
    # Create a translation table
    trans = str.maketrans(special_chars, replace_chars)
    
    # Use the translation table to remove special characters from the text
    return text.translate(trans)
 
def process_large_document(doc, chunk_id, openai_client, processed_documents, nombredocumento):  
    chunks = text_splitter.split_text(doc["page_text"])
    
    print(f"Number of chunks: {len(chunks)}")  
      
    for chunk in chunks:  
        chunk_tokens = num_tokens_from_string(chunk)  
        text_embedding = generate_embedding(chunk, openai_client)  
          
        processed_doc = process_document(doc, chunk_id, openai_client, text_embedding=text_embedding, chunk_text=chunk, nombredocumento=nombredocumento) 
        processed_doc["textoVector"] = text_embedding  
        ##processed_doc["totalTokens"] = chunk_tokens  
        processed_doc["texto_paginas"] = chunk  
          
        processed_documents.append(processed_doc)  
          
        chunk_id += 1  
  
def insert_documents_main(manual, openai_client, nombredocumento=None):  
    manual2 = []  
    processed_documents = []  
    chunk_id = 0  
    counter=0
    # First pass: Separate large documents  
    for doc_list in manual:  
        for doc in doc_list:  
            try:  
                totalTokensText = num_tokens_from_string(doc["page_text"])  
                if totalTokensText > 8000:  
                    manual2.append(doc)  
                    counter += 1
                else:  
                    chunk_id += 1  
                    processed_doc = process_document(doc, chunk_id, openai_client, nombredocumento)  
                    processed_documents.append(processed_doc)  
                      
            except KeyError as e:  
                print(f"Key {str(e)} not found in dictionary: {doc}") 
    print(f"Number of large documents: {counter}") 
  
    # Second pass: Process
    for doc in manual2:  
        try:  
            chunk_id += 1  
            process_large_document(doc, chunk_id, openai_client, processed_documents, nombredocumento)  
        except KeyError as e:  
            print(f"Key {str(e)} not found in dictionary: {doc}")  
  
    return processed_documents  

from azure.core.credentials import AzureKeyCredential  
from azure.search.documents import SearchClient  
from azure.search.documents.indexes import SearchIndexClient  
   
  
# Initialize the search client   
def DeleteDocuments(index_name):
    import json    
    from azure.core.exceptions import HttpResponseError    
    import os    
   
    from azure.search.documents import SearchClient    
    credential = AzureKeyCredential(os.environ["AZURE_SEARCH_API_KEY"]) if len(os.environ["AZURE_SEARCH_API_KEY"]) > 0 else DefaultAzureCredential()
   
    # Download the manual with embeddings  
    connection_string = os.getenv('CONNECTION_STRING')    
  
    endpoint = os.environ["AZURE_SEARCH_SERVICE_ENDPOINT"]    
    credential = AzureKeyCredential(os.environ["AZURE_SEARCH_API_KEY"]) if len(os.environ["AZURE_SEARCH_API_KEY"]) > 0 else DefaultAzureCredential()    
    search_client = SearchClient(endpoint=endpoint, index_name=index_name, credential=credential)   
    manual_name_to_delete = "ManualAcademiaAtencionesTecnicas"  
  
 
    results = search_client.search(search_text="*", filter=f"nombre_manual eq '{manual_name_to_delete}'")  
    
    # Collect document keys to delete  
    documents_to_delete = []  
     
    for result in results:  
        if 'id' in result:
            documents_to_delete.append({"@search.action": "delete", "id": result["id"]})
        else:
            print(f"Document does not contain an 'id' field: {result}")

    # Delete the documents  
   
        # Delete the documents  
    if documents_to_delete:  
            index_operations = search_client.upload_documents(documents=documents_to_delete)  
            print(f"Deleted {len(documents_to_delete)} documents.")  
    else:  
            print("No documents found to delete.")  
        