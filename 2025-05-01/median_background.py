import cv2
import numpy as np
import argparse


def create_median_background(input_video, num_samples=200):
    # Videó betöltése
    cap = cv2.VideoCapture(input_video)

    if not cap.isOpened():
        print(f"Hiba: Nem sikerült a videó megnyitása: {input_video}")
        return None

    # Videó jellemzőinek kinyerése
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Mintavételi időköz kiszámítása
    sample_interval = max(1, frame_count // num_samples)

    # Minták gyűjtése
    print(f"Mintavételezés a videóból (körülbelül {min(num_samples, frame_count)} minta)...")
    sample_frames = []

    for i in range(0, frame_count, sample_interval):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if ret:
            sample_frames.append(frame)

            # Folyamatjelző
            if len(sample_frames) % 20 == 0:
                print(f"  - {len(sample_frames)} minta összegyűjtve")

    # Erőforrások felszabadítása
    cap.release()

    if not sample_frames:
        print("Hiba: Nem sikerült mintákat venni a videóból.")
        return None

    # Medián háttérkép létrehozása
    print(f"Medián háttérkép létrehozása {len(sample_frames)} mintából...")
    samples_array = np.array(sample_frames)
    median_background = np.median(samples_array, axis=0).astype(np.uint8)

    return median_background


def main():
    # Argumentum parser beállítása
    parser = argparse.ArgumentParser(description='Medián háttérkép készítése videóból')
    parser.add_argument('--input', type=str,
                        help='Path to input video',
                        default='../videos/Rat4/04.18.08-04.20.13[M][0@0][0].dav')
    parser.add_argument('--output', type=str,
                        help='Path to output background image',
                        default='background_FINAL.png')
    parser.add_argument('--samples', type=int,
                        help='Number of samples to take from video',
                        default=200)

    args = parser.parse_args()

    # Medián háttérkép létrehozása
    background = create_median_background(args.input, args.samples)

    if background is not None:
        # Háttérkép mentése
        cv2.imwrite(args.output, background)
        print(f"Medián háttérkép elmentve: {args.output}")


if __name__ == "__main__":
    main()