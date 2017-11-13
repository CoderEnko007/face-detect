from localbinarypatterns import LocalBinaryPatterns
from sklearn.svm import LinearSVC
from imutils import paths
import argparse
import cv2

ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image', required=False)
ap.add_argument('-v', '--video', required=False)
ap.add_argument("-t", "--training", required=True)
# ap.add_argument("-e", "--testing", required=True)
args = vars(ap.parse_args())


def detectFace(i):
    gray = cv2.cvtColor(i, cv2.COLOR_BGR2GRAY)
    detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    rects = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5,
                                      minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)
    # rects = [(int(x), int(y), int(x + w), int(y + h)) for (x, y, w, h) in rects]
    return rects


desc = LocalBinaryPatterns(24, 8)
data = []
labels = []

for imagePath in paths.list_images(args["training"]):
    print(imagePath)
    image = cv2.imread(imagePath)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hist = desc.describe(gray)

    print(imagePath.split("\\"))
    labels.append(imagePath.split("\\")[-2])
    data.append(hist)

model = LinearSVC(C=100.0, random_state=42)
model.fit(data, labels)

if args.get('image', False):
    image = cv2.imread(args['image'])
    rect = detectFace(image)
    if len(rect) > 0:
        cv2.rectangle(image, rect[:2], rect[2:], (0, 255, 255), 2)
    cv2.imshow("face", image)
    cv2.waitKey(0)
else:
    if not args.get('video', False):
        camera = cv2.VideoCapture(0)
    else:
        camera = cv2.VideoCapture(args['video'])
    i = 0
    while True:
        ret, image = camera.read()
        if not ret:
            break
        rect = detectFace(image)
        if len(rect) > 0:
            for (x, y, w, h) in rect:
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                # rect = np.array([[x, y, x+w, y+h] for (x, y, w, h) in rect])
                # print(rect)
                face = image[y:y + h, x:x + w]
                gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
                hist = desc.describe(gray)
                prediction = model.predict([hist])[0]

                cv2.putText(image, prediction, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
                cv2.imshow("face", face)
        cv2.imshow("image", image)
        key = cv2.waitKey(1) & 0xff
        if key == ord('w'):
            cv2.imwrite("face%d.jpg" % i, face)
            i += 1
        elif key == ord('q'):
            break
