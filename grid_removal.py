import cv2 as cv
import numpy as np

# Videó megnyitása
capture = cv.VideoCapture('videos\\Rat2\\00.21.38-00.26.49[M][0@0][0].dav')

# Ugrás a 20. képkockára
capture.set(cv.CAP_PROP_POS_FRAMES, 10)

# A 20. képkocka beolvasása
ret, img = capture.read()

if not ret:
    print("Nem sikerült beolvasni a 20. képkockát.")
    capture.release()
    exit(0)

# Kép méretének csökkentése 20%-kal
width = int(img.shape[1] * 0.8)
height = int(img.shape[0] * 0.8)
resized_img = cv.resize(img, (width, height))

# Szürkeárnyalatos kép készítése
gray = cv.cvtColor(resized_img, cv.COLOR_BGR2GRAY)

# Élek detektálása
edges = cv.Canny(gray, 50, 150, apertureSize=3)

# Kontúrok keresése
contours, _ = cv.findContours(edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

# Kontúrok szűrése hosszúság alapján
filtered_contours = [cnt for cnt in contours if cv.arcLength(cnt, True) > 100]

# Maszk készítése a szűrt kontúrok alapján
mask = np.zeros_like(gray)
cv.drawContours(mask, filtered_contours, -1, 255, thickness=cv.FILLED)

# Maszk megjelenítése
cv.imshow('Filtered Mask', mask)
if cv.waitKey(0) & 0xFF == ord('q'):
    cv.destroyAllWindows()

# Inpainting alkalmazása a rács eltávolítására
inpainted_img = cv.inpaint(resized_img, mask, 10, cv.INPAINT_TELEA)

# Inpainted kép megjelenítése
cv.imshow('Inpainted Image', inpainted_img)
if cv.waitKey(0) & 0xFF == ord('q'):
    cv.destroyAllWindows()

# Erőforrások felszabadítása
capture.release()
