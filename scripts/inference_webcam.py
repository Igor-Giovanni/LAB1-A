import cv2
from src.config import Config
from src.pipeline import FechaduraBiometricaPipeline

cfg = Config()


def iniciar_inferencia():
    model_path = str(cfg.PROJECT_ROOT / "models" / "tiny_cnn_binaria_final.h5")

    try:
        pipeline = FechaduraBiometricaPipeline(model_path=model_path, img_size=cfg.IMG_SIZE, threshold=0.5)
    except FileNotFoundError as e:
        print(f"Erro Crítico: {e}")
        return

    cap = cv2.VideoCapture(0)
    print("Iniciando Webcam... Clique na janela do vídeo e pressione 'q' para sair.")

    while True:
        ret, frame = cap.read()
        if not ret: break

        resultado = pipeline.predizer_imagem(frame)

        if resultado["status"] == "Sucesso":
            if resultado["autorizado"]:
                label = f"AUTORIZADO ({resultado['probabilidade'] * 100:.1f}%)"
                color = (0, 255, 0)
            else:
                label = f"NEGADO ({resultado['probabilidade'] * 100:.1f}%)"
                color = (0, 0, 255)

            cv2.putText(frame, label, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)

            if resultado.get("face_crop") is not None:
                cv2.imshow('Visao da CNN (32x32)', cv2.resize(resultado["face_crop"], (160, 160)))
        else:
            cv2.putText(frame, "Procurando rosto...", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        cv2.imshow('Fechadura Biometrica - Pipeline Oficial', frame)

        key = cv2.waitKey(20) & 0xFF
        if key == ord('q') or key == ord('Q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    iniciar_inferencia()