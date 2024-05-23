import cv2
import numpy as np
import requests

# 車牌辨識AIP、金讑
API_ENDPOINT = 'https://api.platerecognizer.com/v1/plate-reader/'
API_KEY = 'YOUR_API_KEY_HERE'

# 開影片
video_path = "licensev.mp4"  
cap = cv2.VideoCapture(video_path)

# 調整影像像素
new_width = 640  
new_height = 480  

# 初始化儲存偵測到的車牌座標的變數
tracked_plate_coordinates = []

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (new_width, new_height))

    # 將影格轉換為 JPEG 格式
    _, img_encoded = cv2.imencode('.jpg', frame)
    img_bytes = img_encoded.tobytes()

    # 發送到車牌識別器API進行識別
    headers = {'Authorization': f'Token {API_KEY}'}
    response = requests.post(API_ENDPOINT, files={'upload': img_bytes}, headers=headers)

    # 處理API
    if response.status_code == 200:
        result = response.json()
        for plate in result['results']:
            plate_text = plate['plate']
            plate_confidence = plate['score']
            print("Plate: %s" % plate_text)
            print("Confidence: %f" % plate_confidence)

            # 儲存偵測到的車牌
            vertices = plate['box']
            tracked_plate_coordinates.append(vertices)

    # 繪製偵測區域
    for vertices in tracked_plate_coordinates:
        pts = np.array(vertices, np.int32)
        pts = pts.reshape((-1, 1, 2))
        cv2.polylines(frame, [pts], isClosed=True, color=(0, 255, 0), thickness=2)

    # 顯示影像
    cv2.imshow('Frame', frame)

    # 離開
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 關閉opencv
cap.release()
cv2.destroyAllWindows()
