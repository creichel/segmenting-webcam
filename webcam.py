import segmentation_models as sm
import cv2
import numpy as np
import utils

cam = cv2.VideoCapture(0)

BACKBONE = 'mobilenetv2'
CLASSES = ['car', 'pedestrian']

n_classes = 1 if len(CLASSES) == 1 else (len(CLASSES) + 1)  # case for binary and multiclass segmentation

cv2.namedWindow("WebcamSegmentation")

# pr_mask = model.predict(image)

model = sm.Unet(BACKBONE, classes=n_classes)
model.load_weights('best_model.h5')

while True:
    ret, frame = cam.read()

    input_height, input_width, _ = frame.shape
    frame = cv2.resize(frame, (320, 320))
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = np.expand_dims(frame, axis=0)

    result = model.predict(frame)
    result = result.squeeze()
    result = cv2.resize(result, (input_width, input_height))

    # result = utils.denormalize(frame.squeeze())

    cv2.imshow("WebcamSegmentation", result.squeeze())
    if not ret:
        break

    # Closing
    k = cv2.waitKey(1)

    if k % 256 == 27:
        # ESC pressed
        print("Escape hit, closing...")
        break

cam.release()

cv2.destroyAllWindows()
