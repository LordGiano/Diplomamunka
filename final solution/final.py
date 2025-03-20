import create_mask
import remove_grids_from_mask
import create_cleaned_mask
from inpainting import inpaint_video  # Most már numpy.ndarray-et fogad
import inpainting_test
import inpaint_parameters
import os

def process_video(video_path: str):
    """
    A teljes inpainted_videos folyamat egy adott videóra, beleértve a mozgásmaszk készítését és az inpaintinget.

    :param video_path: Az input videófájl elérési útja.
    """
    if not os.path.exists(video_path):
        raise ValueError(f"A fájl nem található: {video_path}")

    print(f"Feldolgozás kezdete: {video_path}")

    # 1. Mozgásmaszk létrehozása
    base_mask = create_mask.process_video(video_path)
    if base_mask is None:
        raise RuntimeError("Hiba történt a mozgásmaszk létrehozásakor!")

    # 2. Rácsok eltávolítása
    cleaned_mask = remove_grids_from_mask.remove_grids_from_mask(base_mask)

    # 3. Végső maszk generálása
    final_mask = create_cleaned_mask.create_cleaned_mask(base_mask, cleaned_mask)

    # 4. Inpainting videó létrehozása
    #inpaint_video(video_path, final_mask)  # Most közvetlenül `numpy.ndarray`-et adunk át
    #inpainting_test.inpaint_random_frames(video_path, final_mask)
    inpaint_parameters.inpaint_test(video_path, final_mask)
    print("Feldolgozás kész.")

if __name__ == "__main__":
    process_video(r"C:/Programozás/Diplomamunka/videos/Rat3/14.16.33-14.27.00[M][0@0][0].dav")
