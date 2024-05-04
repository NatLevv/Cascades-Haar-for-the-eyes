import cv2

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    eye_cascade = cv2.CascadeClassifier('eyes.xml')

    eyes = eye_cascade.detectMultiScale(gray)

    if len(eyes) > 0:
        x = [x for (x, y, w, h) in eyes]
        y = [y for (x, y, w, h) in eyes]

        x_min = min(x)
        y_min = min(y)
        x_max = max([x + w for (x, y, w, h) in eyes])
        y_max = max([y + h for (x, y, w, h) in eyes])

        margin_w = int(0.3 * (x_max - x_min))
        margin_h = int(0.2 * (y_max - y_min))
        roi = frame[y_min - margin_h:y_max + margin_h, x_min - margin_w:x_max + margin_w]    # Созд.пр-ка

        if roi.size != 0:    # Проверка на пуст.обл.
            roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            roi_bgr = cv2.cvtColor(roi_gray, cv2.COLOR_GRAY2BGR)

            overlay = roi_bgr.copy()

            cv2.rectangle(frame, (x_min - margin_w, y_min - margin_h),
                          (x_max + margin_w, y_max + margin_h), (192, 192, 192), -1)

            alpha = 0.4
            cv2.addWeighted(overlay, alpha, roi, 1.3 - alpha, 0, roi)    #коэф.проз-ти

    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
