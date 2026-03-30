import tarfile
import zipfile
import unicodedata
import re
from pathlib import Path

class DataExtractor:
    @staticmethod
    def extract_tar(file_path: Path, dest_path: Path, limit=7000):
        if not file_path.exists():
            raise FileNotFoundError(f"Arquivo {file_path} não encontrado.")

        dest_path.mkdir(parents=True, exist_ok=True)
        count = 0
        
        # Checa se o arquivo é ZIP
        if file_path.suffix.lower() == '.zip':
            with zipfile.ZipFile(file_path, 'r') as z:
                for member in z.infolist():
                    # Aceita jpg, jpeg e png
                    if not member.is_dir() and member.filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                        if count >= limit: break
                        with z.open(member) as f:
                            with open(dest_path / Path(member.filename).name, "wb") as out:
                                out.write(f.read())
                        count += 1
        else:
            # Assume que é tar.gz
            with tarfile.open(file_path, "r:gz") as tar:
                for member in tar:
                    if member.isfile() and member.name.lower().endswith(('.jpg', '.jpeg', '.png')):
                        if count >= limit: break
                        f = tar.extractfile(member)
                        if f:
                            with open(dest_path / Path(member.name).name, "wb") as out:
                                out.write(f.read())
                            count += 1
        return count

    @staticmethod
    def sanitize_name(name: str) -> str:
        nfd = unicodedata.normalize('NFD', name)
        clean = ''.join([c for c in nfd if not unicodedata.combining(c)])
        clean = clean.replace(' ', '_')
        return re.sub(r'[^a-zA-Z0-9_.]', '', clean)