import cv2
import mediapipe as mp
import time
import math

class HandDetector:
    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands,
                                        min_detection_confidence=float(self.detectionCon), 
                                        min_tracking_confidence=float(self.trackCon))
        self.mpDraw = mp.solutions.drawing_utils
        self.tipIds = [4, 8, 12, 16, 20]  # ID ujung jari (Ibu Jari, Telunjuk, Tengah, Manis, Kelingking)

    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)

        return img

    def findPosition(self, img, handNo=0, draw=True):
        xList = []
        yList = []
        self.lmList = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                xList.append(cx)
                yList.append(cy)
                self.lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)
            bbox = min(xList), min(yList), max(xList), max(yList)
            if draw:
                cv2.rectangle(img, (bbox[0] - 20, bbox[1] - 20), (bbox[2] + 20, bbox[3] + 20), (0, 255, 0), 2)
            return self.lmList, bbox
        return [], []

    def fingersUp(self):
        fingers = []
        if len(self.lmList) == 0:
            return []

        # Logika untuk ibu jari
        if self.lmList[self.tipIds[0]][1] > self.lmList[self.tipIds[0] - 1][1]:  # Posisi X ibu jari
            fingers.append(1)
        else:
            fingers.append(0)

        # Logika untuk jari lainnya
        for id in range(1, 5):
            if self.lmList[self.tipIds[id]][2] < self.lmList[self.tipIds[id] - 2][2]:  # Posisi Y ujung jari lebih tinggi dari sendi tengahnya
                fingers.append(1)
            else:
                fingers.append(0)

        return fingers

def main():
    pTime = 0
    cap = cv2.VideoCapture(0)
    detector = HandDetector()

    while cap.isOpened():
        success, img = cap.read()
        if not success:
            print("Failed to capture image")
            break

        img = detector.findHands(img)
        lmList, bbox = detector.findPosition(img)

        if lmList:
            fingers = detector.fingersUp()  # Dapatkan status jari
            totalFingers = fingers.count(1)  # Hitung jumlah jari yang terangkat

            # Tampilkan jumlah jari di layar
            cv2.putText(img, f'Jari: {totalFingers}', (50, 100), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)

        # Hitung FPS
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv2.putText(img, f'FPS: {int(fps)}', (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
        cv2.imshow("Image", img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
