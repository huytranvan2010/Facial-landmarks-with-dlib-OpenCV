# USAGE
# python facial_landmarks.py --shape_predictor shape_predictor_68_face_landmarks.dat --image input_image.png
from imutils import face_utils
import argparse
import cv2 
import dlib
import imutils

ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape_predictor", required=True, help="path to facial landmark predictor")     # dlib’s pre-trained facial landmark detector (phát hiện 68 landmarks)
ap.add_argument("-i", "--image", required=True, help="path to the input image")
args = vars(ap.parse_args())

# khởi tạo dlib's face detector (HOG-based)
detector = dlib.get_frontal_face_detector()   # dựa trên HOG + Linear SVM tìm face, xem thêm bài face recognition

# Tạo the facial landmerk predictor
predictor = dlib.shape_predictor(args["shape_predictor"])

# Vẫn phải detect được khuôn mặt trước khi tìm facial landmarks
# load ảnh, resize, convert to gray (cần cho HOG)
image = cv2.imread(args["image"])
image = imutils.resize(image, width=500)    # giữ nguyên aspect ratio
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)   # dùng cho HOG detector bên dưới

# detect faces in the grayscale image
# nhận 2 tham số ảnh (có thể ảnh màu), 2-nd parameter - số image pyramids tạo ra trước khi detect faces (upsample)
# nó giúp phóng ảnh lên để có thể phát hiện khuôn mặt nhỏ hơn, dùng thì chạy lâu hơn
rects = detector(gray, 1)   # trả về list các rectangle chứa khuôn mặt (left, top, right, bottom) <=> (xmin, ymin, xmax, ymax)

# duyệt qua các detections
for (i, rect) in enumerate(rects):
    # xác định facial landmarks for the face region sau đó convert các facial landmarks (x,y)
    # về numpy array, mỗi hàng là một cặp tọa độ
    shape = predictor(gray, rect)   # nhận 2 tham số là ảnh đầu vào và vùng phát hiện khuôn mặt, shape.part(i) là cặp tọa độ thứ i

    # chuyển về dạng numpy các coordinates
    shape = face_utils.shape_to_np(shape)   # numpy array (68, 2)

    # Chuyển dlib's rectange (left, top, right, botttom) = (xmin, ymin, xmax, ymax) to OpenCV-style bounding box (xmin, ymin, w, h)
    # Dễ dàng chuyển được 
    (x, y, w, h) = face_utils.rect_to_bb(rect)
    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 1)    # vẽ rectangle quanh khuôn mặt

    # hiển thị số khuôn mặt trong ảnh, chú ý ở đây đang duyệt qua các detections
    cv2.putText(image, "Face #{}".format(i+1), (x-10, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # duyệt qua các coordinates of facial landmarks (x, y) và vẽ chúng lên ảnh
    for (x, y) in shape:
        cv2.circle(image, (x, y), 1, (255, 0, 255), -1)

# hiển thị ảnh đầu ta với face detections + facial landmarks
cv2.imshow("Output", image)

# lưu ảnh có đánh dấu facial landmarks
cv2.imwrite("output_image.png", image)

# nhấn phím bất kì để thoát
cv2.waitKey(0)



