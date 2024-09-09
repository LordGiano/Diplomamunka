import cv2

def play_dav_video(file_path):
    # Videó beolvasása
    cap = cv2.VideoCapture(file_path)

    if not cap.isOpened():
        print("Nem sikerült megnyitni a fájlt.")
        return

    # Videó lejátszása
    while True:
        ret, frame = cap.read()

        if not ret:
            print("Videó vége.")
            break

        # Kép méretének csökkentése 20%-kal
        width = int(frame.shape[1] * 0.8)
        height = int(frame.shape[0] * 0.8)
        resized_frame = cv2.resize(frame, (width, height))

        # Kép megjelenítése
        cv2.imshow('DAV Video', resized_frame)

        # Kilépés az 'q' gomb lenyomásával
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

    # Erőforrások felszabadítása
    cap.release()
    cv2.destroyAllWindows()

# Használat példa
play_dav_video('videos\\Rat1\\01.08.17-01.08.37[M][0@0][0].dav')
