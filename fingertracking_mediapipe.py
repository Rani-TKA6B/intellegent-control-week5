import cv2
import mediapipe as mp
import time

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
        self.tipIds = [4, 8, 12, 16, 20]  # ID ujung jari

        self.fingerNames = ["Ibu Jari", "Telunjuk", "Tengah", "Manis", "Kelingking"]  # Nama jari

    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)

        return img

    def findPosition(self, img, handNo=0, draw=True):
        self.lmList = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                self.lmList.append([id, cx, cy])
                if draw and id in self.tipIds:
                    cv2.circle(img, (cx, cy), 10, (255, 0, 255), cv2.FILLED)
            return self.lmList
        return []

    def fingersUp(self):
        fingers = []
        if len(self.lmList) == 0:
            return []

        # Ibu jari (dibandingkan dengan titik sebelumnya di sepanjang sumbu X)
        if self.lmList[self.tipIds[0]][1] > self.lmList[self.tipIds[0] - 1][1]:  
            fingers.append(1)  # Terangkat
        else:
            fingers.append(0)  # Turun

        # Jari lain (dibandingkan dengan sendi tengah)
        for id in range(1, 5):
            if self.lmList[self.tipIds[id]][2] < self.lmList[self.tipIds[id] - 2][2]:  
                fingers.append(1)  # Terangkat
            else:
                fingers.append(0)  # Turun

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
        lmList = detector.findPosition(img)

        if lmList:
            fingers = detector.fingersUp()  # Dapatkan status jari
            totalFingers = fingers.count(1)  # Hitung jumlah jari yang terangkat

            # Tentukan nama jari yang terangkat
            raisedFingers = [detector.fingerNames[i] for i in range(5) if fingers[i] == 1]
            if raisedFingers:
                fingersText = ", ".join(raisedFingers)
            else:
                fingersText = "Tidak ada jari terangkat"

            # Tampilkan hasil di layar
            cv2.putText(img, fingersText, (50, 100), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 3)

        # Hitung FPS
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv2.putText(img, f'FPS: {int(fps)}', (10, 50), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2)
        cv2.imshow("Image", img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
