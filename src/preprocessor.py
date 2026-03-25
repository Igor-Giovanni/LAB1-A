import cv2
import numpy as np
import random
import shutil
from pathlib import Path

class ImageProcessor:
    def __init__(self, img_size=32):
        self.img_size = img_size
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    def detect_and_crop(self, frame):
        if frame is None: return None
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.2, 5, minSize=(60, 60))
        for (x, y, w, h) in faces:
            pad = int(w * 0.15)
            y1, y2 = max(0, y - pad), min(gray.shape[0], y + h + pad)
            x1, x2 = max(0, x - pad), min(gray.shape[1], x + w + pad)
            roi = gray[y1:y2, x1:x2]
            return cv2.resize(roi, (self.img_size, self.img_size))
        return None

    def apply_augmentation(self, image):
        img_aug = image.copy()
        if random.random() > 0.5: img_aug = cv2.flip(img_aug, 1)
        alpha = random.uniform(0.9, 1.1)
        beta = random.randint(-10, 10)
        return cv2.convertScaleAbs(img_aug, alpha=alpha, beta=beta)

    def generate_synthetic_background(self):
        cor_base = random.randint(40, 230)
        imagem = np.full((self.img_size, self.img_size), cor_base, dtype=np.float32)
        tipo_gradiente = random.choice(['horizontal', 'vertical', 'nenhum'])
        if tipo_gradiente != 'nenhum':
            intensidade_luz = random.uniform(-40, 40)
            gradiente = np.linspace(0, intensidade_luz, self.img_size)
            if tipo_gradiente == 'horizontal':
                imagem = imagem + gradiente
            else:
                imagem = imagem + gradiente[:, np.newaxis]
        ruido = np.random.normal(0, random.uniform(2.0, 15.0), (self.img_size, self.img_size))
        return np.clip(imagem + ruido, 0, 255).astype(np.uint8)

class DataPreprocessor:
    def __init__(self, config, processor, extractor):
        self.cfg = config
        self.processor = processor
        self.extractor = extractor

    def clear_interim(self):
        if self.cfg.INTERIM_DIR.exists():
            shutil.rmtree(self.cfg.INTERIM_DIR)
        self.cfg.INTERIM_DIR.mkdir(parents=True, exist_ok=True)
        self.cfg.INTERIM_AUTORIZADO_DIR.mkdir(parents=True, exist_ok=True)
        self.cfg.NEGADOS_INTERIM_DIR.mkdir(parents=True, exist_ok=True)

    def process_authorized(self, max_fotos=400):
        for item in self.cfg.RAW_AUTORIZADO_DIR.iterdir():
            nome_limpo = self.extractor.sanitize_name(item.name)
            if nome_limpo != item.name:
                item.rename(item.with_name(nome_limpo))
        for item in self.cfg.RAW_AUTORIZADO_DIR.iterdir():
            nome_pessoa = item.stem
            output_dir = self.cfg.INTERIM_AUTORIZADO_DIR / nome_pessoa
            output_dir.mkdir(parents=True, exist_ok=True)
            rostos = []
            if item.suffix.lower() == '.mp4':
                cap = cv2.VideoCapture(str(item))
                while len(rostos) < max_fotos:
                    ret, frame = cap.read()
                    if not ret: break
                    f = self.processor.detect_and_crop(frame)
                    if f is not None: rostos.append(f)
                cap.release()
            elif item.is_dir():
                fotos = [p for p in item.iterdir() if p.suffix.lower() in ['.jpg', '.png', '.jpeg']]
                for f_p in fotos:
                    if len(rostos) >= max_fotos: break
                    img = cv2.imread(str(f_p))
                    f = self.processor.detect_and_crop(img)
                    if f is not None: rostos.append(f)
            for i, r in enumerate(rostos[:max_fotos]):
                cv2.imwrite(str(output_dir / f"{i:04d}.jpg"), r)
            if 0 < len(rostos) < max_fotos:
                for i in range(max_fotos - len(rostos)):
                    cv2.imwrite(str(output_dir / f"aug_{i:04d}.jpg"), self.processor.apply_augmentation(random.choice(rostos)))

    def process_unknowns(self, ratio=2.0, num_fundos=300):
        total_auth = len(list(self.cfg.INTERIM_AUTORIZADO_DIR.rglob("*.jpg")))
        meta_rostos = int(total_auth * ratio) - num_fundos
        src_dir = self.cfg.RAW_DIR / "selfies"
        image_paths = [p for p in src_dir.rglob('*') if p.is_file() and p.suffix.lower() in ['.jpg', '.jpeg', '.png']]
        random.shuffle(image_paths)
        count_0 = 0
        for img_path in image_paths:
            if count_0 >= meta_rostos: break
            try:
                img = cv2.imdecode(np.fromfile(str(img_path), np.uint8), cv2.IMREAD_COLOR)
                f = self.processor.detect_and_crop(img)
                if f is not None:
                    cv2.imwrite(str(self.cfg.NEGADOS_INTERIM_DIR / f"selfie_{count_0:05d}.jpg"), f)
                    count_0 += 1
            except: continue
        for i in range(num_fundos):
            cv2.imwrite(str(self.cfg.NEGADOS_INTERIM_DIR / f"fundo_{i:04d}.jpg"), self.processor.generate_synthetic_background())