import cv2
import numpy as np

whT = 320
cap = cv2.VideoCapture(0)
confThreshold = 0.5
nmsThreshold = 0.3

classFile = "maske.names"
classNames = []

with open(classFile, "rt") as f:
    classNames = f.read().rstrip("\n").split("\n")

modelConfiguration = "maske.cfg"
modelWeights = "maske_final.weights"

model = cv2.dnn.readNetFromDarknet(modelConfiguration, modelWeights)

model.setPreferableBackend(cv2.dnn.DNN_TARGET_CPU)
model.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)


def findObject(detectionLayers, img):
    hT, wT, cT = img.shape
    bbox = []
    classIds = []
    confs = []

    for detectionLayer in detectionLayers:
        for objectDetection in detectionLayer:
            scores = objectDetection[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            if confidence > confThreshold:
                w, h = int(objectDetection[2] * wT), int(objectDetection[3] * hT)
                x, y = int((objectDetection[0] * wT) - w / 2), int((objectDetection[1] * hT) - h / 2)
                bbox.append([x, y, w, h])
                classIds.append(classId)
                confs.append(float(confidence))
    indices = cv2.dnn.NMSBoxes(bbox, confs, confThreshold, nmsThreshold)

    for i in indices:
        i = i[0]
        box = bbox[i]
        x, y, w, h = box[0], box[1], box[2], box[3]

        if classNames[classIds[i]].upper() == "NO-MASK":
            g, b, r = 0, 0, 255
        else:
            g, b, r = 0, 255, 0

        cv2.rectangle(img, (x, y), (x + w, y + w), (g, b, r), 3)
        cv2.putText(img, f'{classNames[classIds[i]].upper()} {int(confs[i] * 100)}%',
                    (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (g, b, r), 2)


while True:
    success, img = cap.read()
    blob = cv2.dnn.blobFromImage(img, 1 / 255, (whT, whT), [0, 0, 0], 1, crop=False)  
    model.setInput(blob)

    layerNames = model.getLayerNames()
    outputLayers = [layerNames[i[0] - 1] for i in model.getUnconnectedOutLayers()]
    detectionLayers = model.forward(outputLayers)
    findObject(detectionLayers, img)
    cv2.imshow("Mask Detection", img)
    k = cv2.waitKey(20) & 0xFF
    if k == 27:
        break

    cv2.waitKey(50)
