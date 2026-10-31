# Зависимости и лицензии

Собственная лицензия проекта пока не выбрана; файл LICENSE не добавлялся автоматически.

Основные зависимости и источники условий:

- Ultralytics 8.4.194: метаданные установленного пакета указывают AGPL-3.0;
  [исходники и LICENSE](https://github.com/ultralytics/ultralytics).
- OpenCV Python 4.14.0.94: Apache 2.0 по метаданным;
  [исходники и лицензии сборки](https://github.com/opencv/opencv-python).
- Transformers 4.57.6: Apache 2.0 по метаданным;
  [исходники и LICENSE](https://github.com/huggingface/transformers).
- PyTorch 2.14.1 и TorchVision 0.29.1:
  [PyTorch LICENSE](https://github.com/pytorch/pytorch/blob/main/LICENSE),
  [TorchVision LICENSE](https://github.com/pytorch/vision/blob/main/LICENSE).
  Колёса включают компоненты с отдельными условиями; перечень есть в метаданных и исходниках.
- `yolo11m.pt`: [описание YOLO11](https://docs.ultralytics.com/models/yolo11/)
  и условия поставки Ultralytics. Веса не создавались в этом проекте.
- Depth Anything: [карточка конкретного чекпойнта](https://huggingface.co/depth-anything/Depth-Anything-V2-Metric-Outdoor-Small-hf)
  и [базовый репозиторий](https://github.com/DepthAnything/Depth-Anything-V2).
  Условия конкретных весов и базовой модели необходимо сверить перед распространением релиза.

Используется установленный в системе шрифт Segoe UI либо DejaVu Sans;
файлы шрифтов в проект не копируются.
