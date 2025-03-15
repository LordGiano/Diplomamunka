import create_mask
import remove_grids_from_mask
import create_cleaned_mask
import cv2
import os

def process_video(video_path: str, output_filename="final_mask.png"):
    """
    A teljes inpainting folyamat egy adott videóra, kétszeres futtatással.

    :param video_path: Az input videófájl elérési útja.
    :param output_filename: A végső mentett maszk neve.
    """
    if not os.path.exists(video_path):
        raise ValueError(f"A fájl nem található: {video_path}")

    print(f"Feldolgozás kezdete: {video_path}")

    # 1. Háttérmodell tanítása
    bg_subtractor = create_mask.train_background_model(video_path)

    # 2. Mozgásmaszk generálása (a betanított háttérmodell alapján)
    movement_mask = create_mask.create_movement_mask(video_path, bg_subtractor)

    # 3. Rácsok eltávolítása
    cleaned_mask = remove_grids_from_mask.remove_grids_from_mask(movement_mask)

    # 4. Végső maszk generálása
    final_mask = create_cleaned_mask.create_cleaned_mask(movement_mask, cleaned_mask)

    # 5. Csak a végső maszkot mentjük fájlba
    cv2.imwrite(output_filename, final_mask)
    print(f"Feldolgozás befejezve. A végső maszk elmentve: {output_filename}")

    # 6. A végső maszk megjelenítése egy OpenCV ablakban
    cv2.imshow("Végső Maszk", final_mask)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    process_video(r"C:\Programozás\Diplomamunka\videos\Rat2\05.13.23-05.30.35[M][0@0][0].dav")
