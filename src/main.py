import os
import random
from src.config import Config
from src.data_utils import DataExtractor
from src.preprocessor import ImageProcessor, DataPreprocessor
from src.dataset_manager import DatasetManager
from src.engine import ModelEngine
from src.evaluator import ModelEvaluator
from src.model import build_tiny_cnn
from src.export_mif import export_model_to_mif

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['OPENCV_LOG_LEVEL'] = 'OFF'


def main():
    cfg = Config()
    extractor = DataExtractor()
    img_processor = ImageProcessor(cfg.IMG_SIZE)
    data_preprocessor = DataPreprocessor(cfg, img_processor, extractor)
    ds_manager = DatasetManager(cfg)
    evaluator = ModelEvaluator(cfg)
    random.seed(42)

    print("\n" + "=" * 50 + "\n         INICIANDO PIPELINE       \n" + "=" * 50)

    tar_path = cfg.RAW_DIR / "Selfie-dataset.tar.gz"
    if tar_path.exists():
        extractor.extract_tar(tar_path, cfg.RAW_DIR / "selfies", limit=7000)

    data_preprocessor.clear_interim()
    data_preprocessor.process_authorized(max_fotos=400)
    data_preprocessor.process_unknowns(ratio=2.0, num_fundos=300)

    ds_manager.clean_processed()
    ds_manager.split_data(list(cfg.NEGADOS_INTERIM_DIR.glob("*.jpg")), "0_desconhecido")
    ds_manager.split_data(list(cfg.INTERIM_AUTORIZADO_DIR.rglob("*.jpg")), "1_autorizado")

    engine = ModelEngine(cfg, build_tiny_cnn)
    history, model = engine.train()

    evaluator.plot_training_history(history)
    evaluator.evaluate_on_test_set()

    export_model_to_mif()

    print("\n" + "=" * 50 + "\n      PIPELINE CONCLUÍDA COM SUCESSO      \n" + "=" * 50)


if __name__ == "__main__":
    main()