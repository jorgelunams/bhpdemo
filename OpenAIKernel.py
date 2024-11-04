import asyncio
from queue import Full  
from jinja2 import Undefined
import semantic_kernel as sk
 
from semantic_kernel.utils.logging import setup_logging
from semantic_kernel.functions import kernel_function
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion  # Ensure this path is correct
from semantic_kernel.connectors.ai.function_call_behavior import FunctionCallBehavior
from semantic_kernel.connectors.ai.chat_completion_client_base import ChatCompletionClientBase
from semantic_kernel.contents.chat_history import ChatHistory
from semantic_kernel.functions.kernel_arguments import KernelArguments

from dotenv import load_dotenv 
from azure.core.credentials import AzureKeyCredential
import os
import Gen2Services as gen2
import tiktoken
from openai import AzureOpenAI 
import Tools as tools  
import searchaiservice as search 
from semantic_kernel.functions import KernelArguments
from semantic_kernel.connectors.ai.open_ai.prompt_execution_settings.open_ai_prompt_execution_settings import (
    OpenAIChatPromptExecutionSettings,
)
from semantic_kernel.prompt_template.input_variable import InputVariable
from semantic_kernel.prompt_template import PromptTemplateConfig
load_dotenv(override=True) # take environment variables from .env.
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion
# The following variables from your .env file are used in this notebook
endpoint = os.environ["AZURE_SEARCH_SERVICE_ENDPOINT"]
credential = AzureKeyCredential(os.environ["AZURE_SEARCH_ADMIN_KEY"])  
index_name = os.environ["AZURE_SEARCH_INDEX"]
azure_openai_endpoint = os.environ["AZURE_OPENAI_ENDPOINT"]
azure_openai_key = os.environ["AZURE_OPENAI_KEY"] if len(os.environ["AZURE_OPENAI_KEY"]) > 0 else None
azure_openai_embedding_deployment = os.environ["AZURE_OPENAI_EMBEDDING_DEPLOYMENT"]
embedding_model_name = os.environ["AZURE_OPENAI_EMBEDDING_MODEL_NAME"]
azure_openai_api_version = os.environ["AZURE_OPENAI_API_VERSION"]
azure_end_point = os.environ["AZURE_OPENAI_ENDPOINT4O"]
deployment = os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"]
api_key = os.environ["AZURE_OPENAI_API_KEY4O"]
from semantic_kernel.prompt_template import PromptTemplateConfig
from openai import AzureOpenAI
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
import json 



async def classify_question(question : str):
    import os, dotenv
    import openai
   
    max_response_tokens = 20000
    overall_max_tokens = 30000
    prompt_max_tokens = overall_max_tokens - max_response_tokens 
    storake_connection_string = os.environ["AZURE_STORAGE_CONNECTION_STRING"]
    
    json_data =gen2.download_content_as_json(os.environ["AZURE_STORAGE_CONNECTION_STRING"], "bhpcontainer", "Documents/march24Tables_v4.json")
        
    # Extract the first element from each sub-list  
    titles = [sub_list[0] for sub_list in json_data]   
    tilesstr = str(titles)
    extracted_info = []  
    extracted_info_all = []  
# Loop through each JSON string in the list  
    for json_str in titles:  
        try:  
            # Parse the string as JSON  
            json_obj = json.loads(json_str)  # Parse the string as JSON  
    
            # Extract the 'title' and 'Executive_summary' and add them to the list  
            extracted_info.append({  
                'title': json_obj['title'],  
                'Executive_summary': json_obj['Executive_summary']  
            })  
        except json.JSONDecodeError as e:  
            print(f"Error parsing JSON: {e}")  
            
    for json_str in titles:  
        try:  
            # Parse the string as JSON  
            json_obj = json.loads(json_str)  # Parse the string as JSON  
    
            # Extract the 'title' and 'Executive_summary' and add them to the list  
            extracted_info_all.append({  
                'title': json_obj['title'],  
                'Executive_summary': json_obj['Executive_summary'],
                'table_json': json_obj['table_json']  
            })  
        except json.JSONDecodeError as e:  
            print(f"Error parsing JSON: {e}")  
     
    listTables=  str(extracted_info)
    prompt = """    
                ##Systema: You are an expert in understanding the intent of a question.   
                 IMPORTANT: It is critical to evaluate if the quesiton is generic or not. Normally if the question does not require
                get data from tables, or perform calculaitons , statistics, etc. it is generic. If the question requires data from tables
                or calculations, it is not generic. DO IT CORRECTLY!!!!!
                Example of generic question:
                  1) When you do not find a table that has the answer.
                  2) When the question does not require data from tables.
                  3) When is about topics not related to production, finance, results of sales, revenues, expenses, etc.
                  4) Informaiton that is not in tables are accidents, incidents, etc. These are generic quesitons IMPORTANT
                # Can you help me understand the intent of the following question? ##END##  
                {{$question}}    
                ###END    
                Instructions:  
                INPUT: This is the list of tables that contain title and Executive_summary. Look into the question and the   
                title and the executive summary to select the correct table. Select the best table that will have all the details to answer the question.
                do not guess, just select the bext one.
                EXAMPLE: Quesiton: Look into Company Production and Performance Report that contains production guidance and give me all values for Escondida 
                Use this data to find the answer.     {{$list_tables}}  
                Response: 
                 {  
                    "question": "Took into Company Production and Performance Report that contains production guidance and give me all values for Escondida ",  
                    "classification": "The classification of the question",  
                    "title": "Company Production and Performance Report" Look carefully at the question and match the correct title of the table. This is very important!
                    "isGeneric":  "No" 
                    'require_calculations:"no" Look at the question and see if it requires calculations. If it does, indicate yes. If not, indicate no.
                     if the question is give me all values for X that does not require calculations. If the question is give me the average of X that requires calculations.
                    }   
                If the question is not related to any table, classify it as a generic question.  
                    ###START###  These are the titles for each report or table. Critical that you find the one closest to the question that will contain the answer
                    {{$list_tables}}  
                    ###END###  
                1) Classify the question into one of the following categories:  
                    - Generic question   
                    - Question related to a particular table   
                2) Your response should be a JSON object with the following structure. Never start your response with ```json. Use only json
                3)Important always respond in JSON format do not add any other text or comments. Do not add json``` at the begining or the end.
                4) Always start with { and end with }. Always provide the keys and values in the response.
                    {  
                    "question": "The question you are classifying",  
                    "classification": "The classification of the question",  
                    "title": "The name of the table if the question is related to a table" 
                    "isGeneric": "yes" or "no" 
                    'require_calculations: "yes" or "no"
                    }   
                """  

    kernel = sk.Kernel()

    # Add Azure OpenAI chat completion
    chat_completion = AzureChatCompletion(
        deployment_name="gpt-4o",
        api_key="b83ff56180a045f38715c9af7c63ccb6",
        base_url="https://workshopopenaisw.openai.azure.com/",
    )
    kernel.add_service(chat_completion)
     
    kernelLocal = sk.Kernel()  

    service_id = "default" 
    service_id = "aoai_chat_completion"
    
    kernelLocal.add_service(
        AzureChatCompletion(service_id=service_id, deployment_name=deployment,  api_version="2024-05-01-preview",
                            endpoint=azure_end_point, api_key=api_key)
    )
    req_settings = kernelLocal.get_prompt_execution_settings_from_service_id(service_id)
    req_settings.max_tokens = 4000
    req_settings.temperature = 0.0
    req_settings.top_p = 0.8 
    prompt_template_config = PromptTemplateConfig(
        template=prompt,
        name="chat",
        template_format="semantic-kernel",
        input_variables=[
            InputVariable(name="question", description="The user input", is_required=True),
            InputVariable(name="list_tables", description="list of tables", is_required=True),
            
        ],
        execution_settings=req_settings,
    )
        
    chat_function = kernelLocal.add_function(
        function_name="chat",
        plugin_name="chatPlugin",
        prompt_template_config=prompt_template_config,
    )

       
    answer = await kernelLocal.invoke(chat_function, KernelArguments(question=question, list_tables=listTables))  
    answer_str = str(answer)
    data_dict = json.loads(answer_str)
    # Parse the string as JSON to get a dictionary  
    data_dict = json.loads(answer_str)   
    # Access the 'isGeneric' key to retrieve its value  
    is_generic = data_dict['isGeneric'] 
    calculations = data_dict['require_calculations']
   
    if(is_generic == "yes"):  
       answer = await respond_generic_question(question)
       answer_str = str(answer)
       return str(answer_str)      
        
    # Parse the JSON string into a Python object  
    try:  
        data = json.loads(answer_str)  
    except json.JSONDecodeError:  
        print("Invalid JSON")  
        data = []  # or handle the error as appropriate  
    
    infostr=""
    # Assuming you want to extract the title of the first dictionary in the list  
    if data:  # Check if the data list is not empty  
        found_title = data['title']   
        info = find_by_title(found_title, extracted_info_all) 
        table_json = info['table_json']
        table_json_str = str(table_json)
        infostr = str(info)
        
    else:  
        print("No data available.") 
    import re 
    if(calculations == "no" or calculations == "No"):
        finalAnswer_no_calcs = await respond_no_calcs_question(question, table_json_str)
        answer_g = await respond_generic_question(question)
        answer_str_g = str(answer_g)
        combined = answer_str_g + " " + str(finalAnswer_no_calcs)
        finalAnswer = await respond_question(question, combined)
        finalAnswer_str = str(finalAnswer)
        return str(finalAnswer_str)
    code_str = await get_code(question, table_json_str)
    
   ## cleaned_code_str = clean_code_for_json(code_str)    
    # Parse the string as JSON to get a dictionary  
    data_dict = json.loads(code_str)   
    code = data_dict['code']   
  
    local_vars = {}  
    exec(code, {'re': re}, local_vars)  # We pass the 're' module to the exec environment  
    dynamic = local_vars['dynamic_function']   
    # Call the function with the data and the quarter you're interested in  
    quarter = "'v Q2 FY24'"  
    data=infostr
    try:  
        response = dynamic(table_json)  
        print(f"response is  {response}")  
    except ValueError as e:  
        print(e)  
    if(response == None):
        finalAnswer = await respond_no_calcs_question(question, table_json_str)
        finalAnswer_str = str(finalAnswer)
        return str(finalAnswer_str)
    answer_g = await respond_generic_question(question)
    answer_str_g = str(answer_g)
    combined = str(response) +  answer_str_g + " "
    finalAnswer = await respond_question(question, combined)
    finalAnswer_str = str(finalAnswer)
    return str(finalAnswer_str)
# Function to find a dictionary by title  
def find_by_title(title, info_list):  
    for info in info_list:  
        if info['title'] == title:  
            return info  
    return None  # Return None if the title is not found  


sample="""
'columns': [{'name': 'Production', 'type': 'string'}, {'name': 'Quarter performance Q3 FY24', 'type': 'numeric'}, {'name': 'v Q2 FY24', 'type': 'percentage'}, {'name': 'v Q3 FY23', 'type': 'percentage'}, {'name': 'YTD performance YTD Mar FY24', 'type': 'numeric'}, {'name': 'v YTD Mar FY23', 'type': 'percentage'}, {'name': 'FY24 production guidance Previous', 'type': 'range'}, {'name': 'FY24 production guidance Current', 'type': 'range'}, {'name': 'Guidance Status', 'type': 'string'}], 'rows': [{'Production': 'Copper (kt)', 'Quarter performance Q3 FY24': 465.9, 'v Q2 FY24': '7%', 'v Q3 FY23': '15%', 'YTD performance YTD Mar FY24': 1360.3, 'v YTD Mar FY23': '10%', 'FY24 production guidance Previous': '1,720 - 1,910', 'FY24 production guidance Current': '1,720 - 1,910', 'Guidance Status': ''}, {'Production': 'Escondida (kt)', 'Quarter performance Q3 FY24': 288.2, 'v Q2 FY24': '13%', 'v Q3 FY23': '15%', 'YTD performance YTD Mar FY24': 816.1, 'v YTD Mar FY23': '7%', 'FY24 production guidance Previous': '1,080 - 1,180', 'FY24 production guidance Current': '1,080 - 1,180', 'Guidance Status': 'Unchanged'}, {'Production': 'Pampa Norte (kt)', 'Quarter performance Q3 FY24': 61.6, 'v Q2 FY24': '3%', 'v Q3 FY23': '-16%', 'YTD performance YTD Mar FY24': 199.7, 'v YTD Mar FY23': '-9%', 'FY24 production guidance Previous': '210 - 250i', 'FY24 production guidance Current': '210 - 250', 'Guidance Status': 'Upper end'}, {'Production': 'Copper South Australia (kt)', 'Quarter performance Q3 FY24': 79.0, 'v Q2 FY24': '-4%', 'v Q3 FY23': '53%', 'YTD performance YTD Mar FY24': 232.7, 'v YTD Mar FY23': '49%', 'FY24 production guidance Previous': '310 - 340', 'FY24 production guidance Current': '310 - 340', 'Guidance Status': 'Unchanged'}, {'Production': 'Antamina (kt)', 'Quarter performance Q3 FY24': 33.9, 'v Q2 FY24': '-14%', 'v Q3 FY23': '15%', 'YTD performance YTD Mar FY24': 105.6, 'v YTD Mar FY23': '4%', 'FY24 production guidance Previous': '120 - 140', 'FY24 production guidance Current': '120 - 140', 'Guidance Status': 'Unchanged'}, {'Production': 'Iron ore (Mt)', 'Quarter performance Q3 FY24': 61.5, 'v Q2 FY24': '-7%', 'v Q3 FY23': '3%', 'YTD performance YTD Mar FY24': 190.5, 'v YTD Mar FY23': '-1%', 'FY24 production guidance Previous': '254 - 264.5', 'FY24 production guidance Current': '254 - 264.5', 'Guidance Status': ''}, {'Production': 'WAIO (Mt)', 'Quarter performance Q3 FY24': 60.3, 'v Q2 FY24': '-6%', 'v Q3 FY23': '3%', 'YTD performance YTD Mar FY24': 186.8, 'v YTD Mar FY23': '-1%', 'FY24 production guidance Previous': '250 - 260', 'FY24 production guidance Current': '250 - 260', 'Guidance Status': 'Unchanged'}, {'Production': 'WAIO (100% basis) (Mt)', 'Quarter performance Q3 FY24': 68.1, 'v Q2 FY24': '-6%', 'v Q3 FY23': '3%', 'YTD performance YTD Mar FY24': 210.2, 'v YTD Mar FY23': '-1%', 'FY24 production guidance Previous': '282 - 294', 'FY24 production guidance Current': '282 - 294', 'Guidance Status': 'Unchanged'}, {'Production': 'Samarco (Mt)', 'Quarter performance Q3 FY24': 1.2, 'v Q2 FY24': '-10%', 'v Q3 FY23': '12%', 'YTD performance YTD Mar FY24': 3.7, 'v YTD Mar FY23': '13%', 'FY24 production guidance Previous': '4 - 4.5', 'FY24 production guidance Current': '4 - 4.5', 'Guidance Status': 'Upper end'}, {'Production': 'Metallurgical coal - BMA (Mt)', 'Quarter performance Q3 FY24': 6.0, 'v Q2 FY24': '6%', 'v Q3 FY23': '-13%', 'YTD performance YTD Mar FY24': 17.4, 'v YTD Mar FY23': '-16%', 'FY24 production guidance Previous': '23 - 25', 'FY24 production guidance Current': '21.5 - 22.5', 'Guidance Status': 'Lowered'}, {'Production': 'BMA (100% basis) (Mt)', 'Quarter performance Q3 FY24': 12.1, 'v Q2 FY24': '6%', 'v Q3 FY23': '-13%', 'YTD performance YTD Mar FY24': 34.7, 'v YTD Mar FY23': '-16%', 'FY24 production guidance Previous': '46 - 50', 'FY24 production guidance Current': '43 - 45', 'Guidance Status': 'Lowered'}, {'Production': 'Energy coal - NSWEC (Mt)', 'Quarter performance Q3 FY24': 4.1, 'v Q2 FY24': '8%', 'v Q3 FY23': '5%', 'YTD performance YTD Mar FY24': 11.6, 'v YTD Mar FY23': '23%', 'FY24 production guidance Previous': '13 - 15', 'FY24 production guidance Current': '13 - 15', 'Guidance Status': 'Upper end'}, {'Production': 'Nickel - Western Australia Nickel (kt)', 'Quarter performance Q3 FY24': 18.8, 'v Q2 FY24': '-4%', 'v Q3 FY23': '-4%', 'YTD performance YTD Mar FY24': 58.6, 'v YTD Mar FY23': '1%', 'FY24 production guidance Previous': '77 - 87', 'FY24 production guidance Current': '77 - 87', 'Guidance Status': 'Lower half'}]}"

""" 

async def get_code(question : str, data):
    import os, dotenv
    import openai 
    max_response_tokens = 20000 
    prompt = """    
                ##Systema: You are an expert python developer
                # Can you help me understand  the question and create a python code to answer it? ##END##  
                {{$question}}    
                ###END    
                Instructions:  
                INPUT: This json contains names of columns and rows. You need to write a python method that an answer the quesiton.
                look  at the column names, names of columns in the rows as well and the content to make the code.
                when creating the code assume we are passing only the data to the function no other parameters like quarter, month etc.
                every thing most be contained inside the function. 
                Table data is in the following content use it to create the code.
                IMPORTANT: Write only python code, do not offer any explanations or comments. This method python will be later executed dynamically.
                from inside python code. Only pass the data and create all the logic of variables, searches, etc inside the function.
                ###START
                    {{$data}}  
                    ###END###  
                the data is n=only an exmaple of the data will be passed to he fucntion. Do not use the data in the code only use it
                to understand the structure of the data and cretae the code. Do nto offer comments, explanations or show the data in the code.
                keep it simple and only the code to answer the question. do not add ```josn at the begining or the end .
                your answer: This is an exmaple only
                IMPORTANT: NEVER START WITJ ```json. USE ONLY JSON
                always start with "code": " and end with "answer" ALWAYS do not add any other text or comments.
                Make sure you create a nice and clean code with JSON format. Avoid spaces and many new lines. Avoid errors
                specifi all variables, return the answer as a dictionary with the keys and values when required. Add text the the answer
                IMPORTANT : Always include the text for instance if you are calculating average indicate the column name and the values. Always
                provide in your code the answer details of columns and values. 
                IMPORTANT : Consider that some numeric values come like strings. You need to convert them to numbers before performing any calculations.
                {
                    "code": "import re  
                        def dynamic_function(data):  
                            return answer "}
                            
                This is an Exmaple only
import re   
def dynamic_function(data):  
    total_production = 0  
    total_unit_cost = 0  
    for row in data['rows']:  
        if 'values' in row and row['values'][0]['Content'] == 'Escondida':  
            production_range = row['values'][1]['Content']  
            unit_cost_range = row['values'][2]['Content']  
            production_values = re.findall(r'\d+', production_range)  
            unit_cost_values = re.findall(r'\d+\.\d+', unit_cost_range)  
            total_production += sum(map(int, production_values))  
            total_unit_cost += sum(map(float, unit_cost_values))  
    return {'total_production': total_production, 'total_unit_cost': total_unit_cost}  

                                        """  

     
    kernelLocal = sk.Kernel()  

    service_id = "default" 
    service_id = "aoai_chat_completion"
    
    kernelLocal.add_service(
        AzureChatCompletion(service_id=service_id, deployment_name=deployment,  api_version="2024-05-01-preview",
                            endpoint=azure_end_point, api_key=api_key)
    )
    req_settings = kernelLocal.get_prompt_execution_settings_from_service_id(service_id)
    req_settings.max_tokens = 4000
    req_settings.temperature = 0.0
    req_settings.top_p = 0.8 
    prompt_template_config = PromptTemplateConfig(
        template=prompt,
        name="chat",
        template_format="semantic-kernel",
        input_variables=[
            InputVariable(name="question", description="The user input", is_required=True),
            InputVariable(name="data", description="list of tables", is_required=True),
            
        ],
        execution_settings=req_settings,
    )
        
    chat_function = kernelLocal.add_function(
        function_name="chat",
        plugin_name="chatPlugin",
        prompt_template_config=prompt_template_config,
    )
 
    answer = await kernelLocal.invoke(chat_function, KernelArguments(question=question, data=data))  
    answer_str = str(answer)
    return str(answer)

async def respond_generic_question(question: str):
    import os, dotenv
    import openai 
    max_response_tokens = 20000 
    searchresults =search.VectorSemanticSearch("bhp-index-dev", question) 
    results_list = []  
    
    ##["content", "numero_de_pagina", "titulo", "Sumario_Ejecutivo"]
    for result in searchresults:  
            
            pagina = result.get("numero_de_pagina", "")  
            sumario = result.get("Sumario_Ejecutivo", "")   
            titulo = result.get("titulo", "")  
            contexto = json.dumps(result.get("contexto", ""), ensure_ascii=False)  
            score = result.get("@search.score", 0)  
    
            results_list.append({  
                'titulo': titulo,  
                'pagina': pagina,  
                'sumario': sumario,   
                'contexto': contexto,  
                'score': score  
            })  
    prompt = """    
                ##Systema: You are an expert in understanidng information about Operational review for the nine months ended 31 March 2024
                # Look at this quesiton ##END##  
                {{$question}}    
                ###END    
                Instructions:  
                INPUT:  I am giving you three documents - pages extracted from the Operational review for the nine months ended 31 March 2024.
                your job is to find the most relevant information to the question and answer the quesiton with all possible details.
                Do not perform any calculations, only provide an answer to the queestion. Do not invent any thing, use the data provided only.
                Provide all possible detailed, be nice, friendly and professional. Use a professional language and tone.
                INPUT: I am giving you these input 
                'titulo': titulo,  
                'pagina': pagina,  
                'sumario': sumario,   
                'contexto': contexto,  
                'score': score  
                ###START
                    {{$documents}}  
                    ###END###  
                RESPONSE: 
      """  

     
    kernelLocal = sk.Kernel()  

    service_id = "default" 
    service_id = "aoai_chat_completion"
    
    kernelLocal.add_service(
        AzureChatCompletion(service_id=service_id, deployment_name=deployment,  api_version="2024-05-01-preview",
                            endpoint=azure_end_point, api_key=api_key)
    )
    req_settings = kernelLocal.get_prompt_execution_settings_from_service_id(service_id)
    req_settings.max_tokens = 4000
    req_settings.temperature = 0.0
    req_settings.top_p = 0.8 
    prompt_template_config = PromptTemplateConfig(
        template=prompt,
        name="chat",
        template_format="semantic-kernel",
        input_variables=[
            InputVariable(name="question", description="The user input", is_required=True),
            InputVariable(name="response", description="list of tables", is_required=True),
            
        ],
        execution_settings=req_settings,
    )
        
    chat_function = kernelLocal.add_function(
        function_name="chat",
        plugin_name="chatPlugin",
        prompt_template_config=prompt_template_config,
    )
    
    documents = str(results_list) 
    answer = await kernelLocal.invoke(chat_function, KernelArguments(question=question, documents=documents))  
    answer_str = str(answer)
    return str(answer)

async def respond_no_calcs_question(question: str, data):
    import os, dotenv
    import openai 
    max_response_tokens = 20000 
    
    prompt = """    
                ##Systema: You are an expert in understanidng information about Operational review for the nine months ended 31 March 2024
                # Look at this quesiton ##END##  
                {{$question}}    
                ###END    
                Instructions:  
                INPUT:  I am giving you a table that contains a title, Executive_summary and a table with columns and rows.
                look at the quesiton and answer it with all possible details, organized in bullets or lists. Use tables when needed.
                do not invent the answer, use the data provided to create the response. Never invent any thing. USe the data provided only.
                Do not perform any calculations only provide the details from the table and answer the question.
                INPUT
                ###START
                    {{$documents}}  
                    ###END###  
                RESPONSE: 
      """  

     
    kernelLocal = sk.Kernel()  

    service_id = "default" 
    service_id = "aoai_chat_completion"
    
    kernelLocal.add_service(
        AzureChatCompletion(service_id=service_id, deployment_name=deployment,  api_version="2024-05-01-preview",
                            endpoint=azure_end_point, api_key=api_key)
    )
    req_settings = kernelLocal.get_prompt_execution_settings_from_service_id(service_id)
    req_settings.max_tokens = 4000
    req_settings.temperature = 0.0
    req_settings.top_p = 0.8 
    prompt_template_config = PromptTemplateConfig(
        template=prompt,
        name="chat",
        template_format="semantic-kernel",
        input_variables=[
            InputVariable(name="question", description="The user input", is_required=True),
            InputVariable(name="data", description="list of tables", is_required=True),
            
        ],
        execution_settings=req_settings,
    )
        
    chat_function = kernelLocal.add_function(
        function_name="chat",
        plugin_name="chatPlugin",
        prompt_template_config=prompt_template_config,
    )
    
    
 
    answer = await kernelLocal.invoke(chat_function, KernelArguments(question=question, documents=data))  
    answer_str = str(answer)
    return str(answer)


########## Respond to a question with a response
async def respond_question(question : str, response):
    import os, dotenv
    import openai 
    max_response_tokens = 20000 
    prompt = """    
                ##Systema: You are an expert in understanidng information about Operational review for the nine months ended 31 March 2024
                # Look at this quesiton ##END##  
                {{$question}}    
                ###END    
                Instructions:  
                INPUT: I am giving you the answer to the question. You need to write a set of sentences that will be the response to the question.
                with al ldetails, organized in nullets or lists, use tables when needed. Be nice and friendly.
                If response is empty or not clear, respond that you could not find the answer to the question.. Suggest how to ask the question.
                do not invent the answer, use the data provided to create the response. Never invent any thing. USe the data provided only.
                IMPORTANT: You will get information from the document and also possible from previous python calculations. Make sure you include 
                in your report reference to these calculaitons in your text. Include every thig provided in the answer.
                If the calculation in the response is incorrect fix it and create a new calculaiton.
                Use all the data to respond. Explain your logic, provide details and an executive summary
                ###START
                    {{$response}}  
                    ###END###  
                RESPONSE: 
      """  

     
    kernelLocal = sk.Kernel()  

    service_id = "default" 
    service_id = "aoai_chat_completion"
    
    kernelLocal.add_service(
        AzureChatCompletion(service_id=service_id, deployment_name=deployment,  api_version="2024-05-01-preview",
                            endpoint=azure_end_point, api_key=api_key)
    )
    req_settings = kernelLocal.get_prompt_execution_settings_from_service_id(service_id)
    req_settings.max_tokens = 4000
    req_settings.temperature = 0.0
    req_settings.top_p = 0.8 
    prompt_template_config = PromptTemplateConfig(
        template=prompt,
        name="chat",
        template_format="semantic-kernel",
        input_variables=[
            InputVariable(name="question", description="The user input", is_required=True),
            InputVariable(name="response", description="list of tables", is_required=True),
            
        ],
        execution_settings=req_settings,
    )
        
    chat_function = kernelLocal.add_function(
        function_name="chat",
        plugin_name="chatPlugin",
        prompt_template_config=prompt_template_config,
    )
 
    answer = await kernelLocal.invoke(chat_function, KernelArguments(question=question, response=response))  
    answer_str = str(answer)
    return str(answer)
def clean_code_for_json(code_str):  
    import re  
    # Escape backslashes first (replace \ with \\)  
    cleaned_code = code_str.replace("\\", "\\\\")  
  
    # Escape single quotes inside the strings (replace ' with \')  
    # We use a regular expression to replace only single quotes that are within strings  
    cleaned_code = re.sub(r"(\'.*?\')", lambda x: x.group(0).replace("'", "\\'"), cleaned_code)  
  
    # Escape double quotes outside of strings (replace " with \")  
    # We use a regular expression to replace only double quotes that are not within strings  
    cleaned_code = re.sub(r'(?<!\\)"', '\\"', cleaned_code)  
  
    # Unescape double backslashes in front of the double quotes  
    # (replace \\" with ")  
    cleaned_code = cleaned_code.replace("\\\\\"", '\\"')  
  
    return cleaned_code  
       