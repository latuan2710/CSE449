import numpy as np
import cv2

coco_names = ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
              "fire hydrant", "street sign", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse",
              "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "hat", "backpack", "umbrella", "shoe",
              "eye glasses", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
              "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
              "plate", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
              "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed",
              "mirror", "dining table", "window", "desk", "toilet", "door", "tv", "laptop", "mouse", "remote",
              "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "blender", "book",
              "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush", "hair brush"]

weightsPath = "frozen_inference_graph.pb"
configPath = "mask_rcnn_inception_v2_coco_2018_01_28.pbtxt"

net = cv2.dnn.readNetFromTensorflow(weightsPath, configPath)

image = cv2.imread("doggo.jpg")
(H, W) = image.shape[:2]

blob = cv2.dnn.blobFromImage(image, swapRB=True, crop=False)

net.setInput(blob)

boxes = net.forward(["detection_out_final"])

output = boxes[0].squeeze()

num = np.argwhere(output[:, 2] > 0.8).shape[0]

font = cv2.FONT_HERSHEY_SIMPLEX

img = cv2.imread("doggo.jpg")
for i in range(num):
    x1n, y1n, x2n, y2n = output[i, 3:]
    x1 = int(x1n * W)
    y1 = int(y1n * H)
    x2 = int(x2n * W)
    y2 = int(y2n * H)
    img = cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 3)
    class_name = coco_names[int(output[i, 1])]
    img = cv2.putText(img, class_name, (x1, y1 - 10), font, 0.5,
                      (255, 0, 0), 1, cv2.LINE_AA)

cv2.imshow("Result", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
