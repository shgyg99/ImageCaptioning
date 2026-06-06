import os
import kagglehub
from src.logger import get_logger
import shutil
from src.custom_exception import CustomException
import zipfile
from utils.config_manager import ConfigManager


logger = get_logger(__name__)
class DataIngestion:

    def __init__(self, dataset_name:str, target_dir:str):
        self.dataset_name = dataset_name
        self.target_dir = target_dir

    def create_raw_dir(self):
        raw_dir = os.path.join(self.target_dir, "raw")
        if not os.path.exists(raw_dir):
            try:
                os.makedirs(raw_dir)
                logger.info(f"Created the {raw_dir}")
            except Exception as e:
                logger.error("Error while creating raw directory...")
                raise CustomException("Faile to create raw dir", e)
        return raw_dir
    
    def extract_images_and_labels(self, path:str, raw_dir:str):
        try:
            if path.endswith('.zip'):
                logger.info("Extracting zip file")

                with zipfile.ZipFile(path, 'r') as zip_ref:
                    zip_ref.extractall(path)

            images_folder = os.path.join(path, "Images")
            
            if os.path.exists(images_folder):
                shutil.move(images_folder,os.path.join(raw_dir,"Images"))
                logger.info("Images moved sucesfully..")
            else:
                logger.info("Imagees folder dont exist..")
        except Exception as e:
            logger.error("Error while extracting .")
            raise CustomException("Erro while extracting.." , e)
    
    def download_datset(self,raw_dir:str):
        try:
            path = kagglehub.dataset_download(self.dataset_name)
            logger.info(f"Download the data from {path}")

            self.extract_images_and_labels(path , raw_dir)

        except Exception as e:
                logger.error("Error while downlaoding data")
                raise CustomException("Erro while downloading data" , e)
        
    def run(self):
        try:
            raw_dir = self.create_raw_dir()
            self.download_datset(raw_dir)

        except Exception as e:
                logger.error("Error while data ingestion pipeline")
                raise CustomException("Erro while data ingestion pipeline" , e)
        



if __name__=="__main__":
    config_manager = ConfigManager.from_yaml()
    dataset_name = config_manager.get("data.dataset_name", "adityajn105/flickr8k")
    target_dir = config_manager.get("data.target_dir", "artifacts")

    data_ingestion = DataIngestion(dataset_name, target_dir)
    data_ingestion.run()