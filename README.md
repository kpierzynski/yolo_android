# YOLO classificator runner for Android

This is simple app to run yolov11-cls models on Android device, including fine-tuned models.

## Export Yolo
App required model in `onnx` format. You can export finetuned model:

`yolo export model=path/to/model.pt format=onnx`

## Export labels
For classification, labels are required. Generate labels:

```python
from ultralytics import YOLO

model = YOLO("path/to/model.pt")
names = model.names

with open("labels.txt", "w") as f:
    if isinstance(names, dict):
        for i in range(len(names)):
            f.write(names[i] + "\n")
    else:
        for name in names:
            f.write(name + "\n")
```
