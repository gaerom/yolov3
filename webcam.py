import cv2
import numpy as np
import darknet as dn
from cv2 import VideoWriter
from cv2 import VideoWriter_fourcc

def convert(r):
    boxes = []

    for i in range(len(r)):
        width = r[i][2][2]
        height = r[i][2][3]
        center_x = r[i][2][0]
        center_y = r[i][2][1]
        topLeft_x = center_x - (width / 2)
        topLeft_y = center_y - (height / 2)

        x, y, w, h = topLeft_x, topLeft_y, width, height

        boxes.append((x, y, w, h))

    return boxes

def draw_Bbox(image, boxes):
    for i in range(len(boxes)):

        x, y, w, h = boxes[i]

        top = max(0, np.floor(x + 0.5).astype(int))
        left = max(0, np.floor(y + 0.5).astype(int))
        right = min(image.shape[1], np.floor(x + w + 0.5).astype(int))
        bottom = min(image.shape[0], np.floor(y + h + 0.5).astype(int))

        cv2.rectangle(image, (top, left), (right, bottom), (255, 0, 0), 2)

    # for category, score, bounds in r:
    #     x, y, w, h = bounds
    #     cv2.rectangle(image, (int(x - w / 2), int(y - h / 2)), (int(x + w / 2), int(y + h / 2)), (255, 0, 0), thickness=2)

        for j in range(len(r)):
            className = str(r[j][0])
            className = className[2:len(className) - 1]
        cv2.putText(image, str(className), (int(x), int(y)), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)


net = dn.load_net(b"yolov3.cfg", b"yolov3.weights", 0)
meta = dn.load_meta(b"coco.data")


# video 처리
cap = cv2.VideoCapture('/home/ksr/바탕화면/yolo/darknet/video.mp4')
video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
video_fps = int(cap.get(cv2.CAP_PROP_FPS))
# detection 결과 저장
video = cv2.VideoWriter('../results/output_video.avi', cv2.VideoWriter_fourcc(*'XVID'), video_fps, (video_width, video_height))

'''
cap = cv2.VideoCapture(0) # 0으로 하면 웹캡
#detection 결과 저장, webcam.py와 같은 dir
video = cv2.VideoWriter('webcam.avi', VideoWriter_fourcc(*'XVID'), 25.0, (640, 480))


while (cap.isOpened()):

    ret, frame = cap.read()
    
    r = dn.detect(net, meta, frame)
    boxes = convert(r)
    draw_Bbox(frame, boxes)
    
    if ret:
    	video.write(frame)

    if not ret:
        break

  
    
    cv2.imshow('result', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
'''

# image detection
image_path = "/home/ksr/바탕화면/yolo/darknet/person.jpg"
frame = cv2.imread(image_path)

# Detection 수행
r = dn.detect(net, meta, frame)
boxes = convert(r)
draw_Bbox(frame, boxes)

output_path = "/home/ksr/바탕화면/yolo/darknet/img_result/output_image.jpg"  # 저장할 이미지 파일의 경로
cv2.imwrite(output_path, frame)

# Detection 결과를 화면에 출력
cv2.imshow('result', frame)
cv2.waitKey(0)
cv2.destroyAllWindows()
     
        
cap.release()
video.release()       
cv2.destroyAllWindows()
