from ultralytics import YOLO

if __name__ == '__main__':
    model_yaml = r"D:\yolo\ultralytics-main\yolov8n.yaml"
    data_yaml = r"D:\yolo\ultralytics-main\data.yaml"
    pre_model = r"D:\yolo\ultralytics-main\runs\detect\train\weights\best.pt"
    model = YOLO(model_yaml, task='detect').load(pre_model)
    results = model.train(data = data_yaml, epochs = 15, imgsz = 640, batch = 4, workers = 2)