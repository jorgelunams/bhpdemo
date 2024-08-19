from azure.storage.filedatalake import DataLakeFileClient
from azure.storage.filedatalake import DataLakeFileClient, generate_file_sas
from azure.storage.filedatalake import FileSasPermissions
from datetime import datetime, timedelta
import json
def upload_to_datalake(connectionstring, file_system_name, file_path, data):
    file = DataLakeFileClient.from_connection_string(connectionstring,
                                                     file_system_name=file_system_name, file_path=file_path)
    file.create_file()
    data_bytes = data.encode('utf-8')  # Encode the data to bytes
    file.append_data(data_bytes, offset=0, length=len(data_bytes))
    file.flush_data(len(data_bytes))
    
    
def get_file_url(connectionstring, file_system_name, file_path):
    file = DataLakeFileClient.from_connection_string(connectionstring,
                                                     file_system_name=file_system_name, file_path=file_path)
   

    # Generate SAS token
    sas_token = generate_file_sas(
        account_name=file.account_name,
        file_system_name=file_system_name,
        file_name=file_path,
        directory_name="",      
        credential=file.credential.account_key,
        permission=FileSasPermissions(read=True),
        expiry=datetime.utcnow() + timedelta(hours=1)  # Token valid for 1 hour
    )

    # Construct the URL
    url = f"https://{file.account_name}.dfs.core.windows.net/{file.file_system_name}/{file_path}?{sas_token}"

    return url 
def download_content_as_json(connection_string, file_system_name, file_path):
    file_client = DataLakeFileClient.from_connection_string(connection_string, file_system_name, file_path)

    download = file_client.download_file()
    file_content = download.readall()

    return json.loads(file_content)