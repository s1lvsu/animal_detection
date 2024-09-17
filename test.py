from ultralytics import YOLO

# Load a model
model = YOLO('best.pt')  # pretrained YOLOv8n model 100 epoh

# Run batched inference on a list of images
results = model.predict('name.jpg', save=True)

# Process results list
for result in results:
     boxes = result.boxes  # Boxes object for bbox outputs

     if len(boxes.xyxy) > 0:
         print(boxes.xyxy)

