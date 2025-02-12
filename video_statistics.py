import cv2
import os
import glob
import pandas as pd

# A fő mappa, ahol a videók találhatók
root_folder = "videos"  # Ha máshol van, módosítsd ennek megfelelően

# Eredmények tárolása
video_stats = []

# Végigmegyünk minden almappán (Rat1, Rat2, stb.)
for rat_folder in sorted(os.listdir(root_folder)):
    rat_path = os.path.join(root_folder, rat_folder)

    # Ellenőrizzük, hogy mappa-e
    if not os.path.isdir(rat_path):
        continue

    # Videófájlok keresése a mappában
    video_files = glob.glob(os.path.join(rat_path, "*.dav"))  # Ha más formátum is van, bővíthető

    total_videos = len(video_files)
    total_duration = 0  # Összes videóhossz másodpercben
    video_lengths = []  # Egyedi videók hossza

    for video_path in video_files:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"⚠️ Nem lehet megnyitni a videót: {video_path}")
            continue

        # Videó hosszának lekérése másodpercben
        fps = cap.get(cv2.CAP_PROP_FPS)  # Képkocka per másodperc
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # Képkockák száma
        duration = frame_count / fps if fps > 0 else 0  # Másodpercben
        video_lengths.append(round(duration, 2))
        total_duration += duration

        cap.release()

    # Eredmények mentése egy listába
    video_stats.append([rat_folder, total_videos, round(total_duration, 2), video_lengths])

# 📊 Táblázatba konvertálás
df = pd.DataFrame(video_stats,
                  columns=["Patkány/Ketrec", "Videók száma", "Teljes hossz (mp)", "Egyedi videóhosszak (mp)"])
df.set_index("Patkány/Ketrec", inplace=True)

# Kiírás konzolra
print(df)

# CSV fájlba mentés
csv_output_path = "video_statistics.csv"
df.to_csv(csv_output_path)
print(f"\n📂 Az eredmények elmentve: {csv_output_path}")
