# Readme  
   
## Overview  
   
This code is a demo showcasing how to use `semantic_kernel`, a Python package for developing an AI assistant that can understand and respond to user queries. This demo is specifically tailored to extract and process information from documents such as Operational Reviews, and answer questions based on this information.  
   
The code makes use of various libraries and services such as OpenAI, Azure, and other custom services for data extraction, processing, and generating responses.  
   
## Pre-requisites  
   
To run this code, you need the following:  
   
1. Python environment with necessary libraries installed.  
2. An Azure account with necessary services enabled.  
3. Environment variables set up with the required Azure service keys.  
   
## Usage  
   
The code contains several functions:  
   
1. `classify_question(question: str)`: This function accepts a string question as input. It loads a JSON data file from an Azure storage account and parses the data to extract titles and summaries. It then uses the OpenAI service to classify the question based on the data and returns a response. If the question is classified as generic, it calls the `respond_generic_question` function to generate a response.  
   
2. `find_by_title(title, info_list)`: This helper function is used inside `classify_question`. It takes a title and a list of information (each item in the list is a dictionary with a 'title' key). The function returns the dictionary that has the matching title. If no match is found, it returns None.  
   
3. `get_code(question: str, data)`: This function accepts a string question and data as input. It uses the OpenAI service to generate Python code that can answer the question based on the data. The generated code is returned as a string.  
   
4. `respond_generic_question(question: str)`: This function accepts a string question as input. It uses the VectorSemanticSearch service to find relevant documents based on the question. It then uses the OpenAI service to generate a response based on the question and the found documents.  
   
5. `respond_no_calcs_question(question: str, data)`: This function accepts a string question and data as input. It uses the OpenAI service to generate a response to the question based on the data, without performing any calculations.  
   
6. `respond_question(question: str, response)`: This function accepts a string question and a response as input. It uses the OpenAI service to generate a set of sentences that will be the response to the question based on the provided response.  
   
Please note that this code is not ready for production. It is provided as a demo and does not include essential aspects such as security measures, testing, UAT, DevOps, etc.  
   
## Disclaimer  
   
THIS CODE IS PROVIDED AS IS WITHOUT WARRANTY OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING ANY IMPLIED WARRANTIES OF FITNESS FOR A PARTICULAR PURPOSE, MERCHANTABILITY, OR NON-INFRINGEMENT.  
   
Microsoft does not take any responsibility for, nor does it warrant that the functions contained in the work will meet your requirements or that the operation of the work will be error-free.  
   
Please review the code thoroughly before using it in a production environment.

Version from WorkShop Updated 