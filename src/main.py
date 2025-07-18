from knn_keypoint_alpha_shape import process_video_with_knn_keypoint_alpha_shape
from base_inpainting import process_video_with_base_inpainting
from knn_inpainting import process_video_with_knn_inpainting
import cv2 as cv


def create_base_inpainting(input_video_path, output_folder):
    output_path = f"{output_folder}/base_inpainting_output.mp4"

    process_video_with_base_inpainting(input_video_path, output_path)


def create_knn_inpainting(input_video_path, output_folder):
    output_path = f"{output_folder}/knn_inpainting_output.mp4"
    history = 400
    dist2Threshold = 300.0
    detectShadows = False
    kernel_size = (3, 3)
    dilation_iterations = 2
    inpaint_radius = 7
    method = cv.INPAINT_TELEA

    process_video_with_knn_inpainting(
        input_video_path,
        output_path,
        history=history,
        dist2Threshold=dist2Threshold,
        detectShadows=detectShadows,
        kernel_size=kernel_size,
        dilation_iterations=dilation_iterations,
        inpaint_radius=inpaint_radius,
        method=method
    )


def create_knn_keypoint_alpha_shape(input_video_path, output_folder):
    output_video_path = f"{output_folder}/knn_keypoint_alpha_shape_output.mp4"
    history = 200
    dist2Threshold = 100
    vicinity_size = 50
    detector_type = 'AKAZE'
    keypoint_size = 8
    keypoint_radius = 5
    filter_grid_keypoints = True
    grid_vicinity = 5

    process_video_with_knn_keypoint_alpha_shape(
        input_video_path,
        output_video_path,
        history=history,
        dist2Threshold=dist2Threshold,
        vicinity_size=vicinity_size,
        detector_type=detector_type,
        keypoint_size=keypoint_size,
        keypoint_radius=keypoint_radius,
        filter_grid_keypoints=filter_grid_keypoints,
        grid_vicinity=grid_vicinity
    )


if __name__ == "__main__":
    def main():
        input_video_path = r"C:/Programozás/Diplomamunka/videos/Rat5/04.12.01-04.13.41[M][0@0][0].dav"
        output_folder = r"C:/folder"

        #create_base_inpainting(input_video_path, output_folder) # eredeti képkocka és az inpaintingelt képkocka öszehasonlítása
        #create_knn_inpainting(input_video_path, output_folder) # eredeti képkocka és az inpaintingelt mozgási maszk öszehasonlítása
        create_knn_keypoint_alpha_shape(input_video_path, output_folder) # alpha shape alkalmazása az eredeti képkockán és a kulcspontdetektálással kiegészített mozgási maszkon

    main()