
""" Download pre-trained models from Google drive. """
import os
import argparse
import zipfile 
import logging
import requests
from tqdm import tqdm

logging.basicConfig(
		format="%(asctime)s - %(levelname)s - %(filename)s -   %(message)s",
		datefmt="%d/%m/%Y %H:%M:%S",
		level=logging.INFO)

MODEL_TO_URL = {
	'models': 'https://drive.google.com/open?id=14cmxzv40nIlsszVJrjSNN18KdVBh1ij5',
}


def download_file_from_google_drive(id, destination):
	URL = "https://docs.google.com/uc?export=download"

	session = requests.Session()

	response = session.get(URL, params={ 'id' : id }, stream=True)
	token = get_confirm_token(response)

	if token:
		params = { 'id' : id, 'confirm' : token }
		response = session.get(URL, params=params, stream=True)

	save_response_content(response, destination)

def get_confirm_token(response):
	for key, value in response.cookies.items():
		if key.startswith('download_warning'):
			return value

	return None

def save_response_content(response, destination):
	CHUNK_SIZE = 32768

	with open(destination, "wb") as f:
		for chunk in tqdm(response.iter_content(CHUNK_SIZE)):
			if chunk: # filter out keep-alive new chunks
				f.write(chunk)

def download_model(name='models'):
	project_dir = os.path.dirname(os.path.abspath(__file__))

	
	os.makedirs(os.path.join(project_dir, "..", name), exist_ok=True)
	file_destination = os.path.join(project_dir, "..", 'models.zip')
	file_id = MODEL_TO_URL[name].split('id=')[-1]
	logging.info(f'Downloading {name} model (~1300MB tar.xz archive)')
	download_file_from_google_drive(file_id, file_destination)

	logging.info('Extracting model from archive (~1300MB folder)')
	with zipfile.ZipFile(file_destination, 'r') as zip_ref:
		zip_ref.extractall(path=os.path.dirname(file_destination))

	logging.info('Removing archive')
	os.remove(file_destination)
	logging.info('Done.')

def main():
	download_model()

if __name__ == "__main__":
	main()



