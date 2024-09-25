import os
import gdown
import logging
import shutil
from typing import Optional

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def download_raw_recipes(target_filename: str = 'RAW_recipes.csv', folder_id: Optional[str] = None) -> None:
    """
    Download the RAW_recipes.csv file from Google Drive if it doesn't exist locally.
    
    Args:
    target_filename (str): The name of the file to download (default is 'RAW_recipes.csv')
    folder_id (str, optional): The Google Drive folder ID. If None, uses a default ID.
    """
    if folder_id is None:
        folder_id = '17FkgU6C6IKkibt1_2R2mX_XShHzfji9-'
    
    if os.path.exists(target_filename):
        logging.info(f"{target_filename} already exists in the project directory.")
        return
    
    logging.info(f"{target_filename} not found. Attempting to download...")
    
    # Construct the download URL
    url = f'https://drive.google.com/uc?id={folder_id}'
    
    try:
        # Create a temporary directory for downloading
        temp_dir = 'temp_download'
        os.makedirs(temp_dir, exist_ok=True)
        
        # Download the folder content
        gdown.download_folder(url, output=temp_dir, quiet=False, use_cookies=False)
        
        # Check if the target file is in the downloaded content
        downloaded_file = os.path.join(temp_dir, target_filename)
        if os.path.exists(downloaded_file):
            # Move the file to the project directory
            shutil.move(downloaded_file, target_filename)
            logging.info(f"Successfully downloaded and moved {target_filename} to the project directory.")
        else:
            logging.error(f"Download completed, but {target_filename} was not found in the downloaded content.")
        
    except Exception as e:
        logging.error(f"An error occurred while downloading: {str(e)}")
    
    finally:
        # Clean up: remove the temporary directory
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
            logging.info("Cleaned up temporary download directory.")

def main():
    # You can specify a different filename or folder ID here if needed
    download_raw_recipes()
    
    # Example of how to use with different parameters:
    # download_raw_recipes('CustomRecipes.csv', 'your_custom_folder_id')

if __name__ == "__main__":
    main()