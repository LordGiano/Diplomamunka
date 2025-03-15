import os
import subprocess

# 🔹 FFmpeg pontos elérési útja (frissítsd, ha máshová telepítetted)
ffmpeg_path = r"C:\ffmpeg\bin\ffmpeg.exe"

# 🔹 Forrás és célmappa beállítása
source_dir = r"C:\Programozás\Diplomamunka\videos"  # Itt vannak a .dav fájlok
target_dir = r"C:\Programozás\Diplomamunka\converted_videos"  # Ide kerülnek az MP4 vagy AVI fájlok
convert_to = "mp4"  # Válaszd: "mp4" vagy "avi"

# 🔹 Ellenőrizzük, hogy létezik-e a célmappa
if not os.path.exists(target_dir):
    os.makedirs(target_dir)

# 🔹 Végigmegyünk az összes patkánymappán (Rat1, Rat2, stb.)
for rat_folder in os.listdir(source_dir):
    rat_path = os.path.join(source_dir, rat_folder)

    if os.path.isdir(rat_path):  # Csak mappákkal dolgozunk
        output_rat_folder = os.path.join(target_dir, rat_folder)
        os.makedirs(output_rat_folder, exist_ok=True)

        # 🔹 Végigmegyünk az adott mappában lévő összes .dav fájlon
        for file in os.listdir(rat_path):
            if file.endswith(".dav"):
                input_path = os.path.join(rat_path, file)
                output_file = os.path.join(output_rat_folder, os.path.splitext(file)[0] + f".{convert_to}")

                # 🔹 FFmpeg konvertálás MP4 vagy AVI formátumba (audio kihagyásával)
                if convert_to == "mp4":
                    ffmpeg_command = f'"{ffmpeg_path}" -i "{input_path}" -c:v libx264 -crf 18 -preset slow -an -r 25 "{output_file}"'
                elif convert_to == "avi":
                    ffmpeg_command = f'"{ffmpeg_path}" -i "{input_path}" -c:v mjpeg -q:v 3 -an -r 25 "{output_file}"'

                print(f"🎥 Konvertálás folyamatban: {input_path} → {output_file}")
                subprocess.run(ffmpeg_command, shell=True)

print("✅ Az összes .dav fájl konvertálása befejeződött! Csak az elkészült fájlok lettek átmásolva.")
