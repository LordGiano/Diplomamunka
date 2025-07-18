import numpy as np
from scipy.spatial import Delaunay
from shapely.geometry import LineString, MultiLineString, Polygon, MultiPolygon
from shapely.ops import polygonize, unary_union
import cv2
import os


def alpha_shape(points, alpha, return_largest_only=False):
    if len(points) < 4:
        # Not enough points to create a valid shape
        return None

    # Compute the Delaunay triangulation
    tri = Delaunay(points)

    # Initialize edge dictionary
    edges = {}

    # For each Delaunay triangle
    for i in range(tri.simplices.shape[0]):
        simplex = tri.simplices[i]
        # For each edge in the triangle
        for j in range(3):
            # Get indices of the edge's endpoints
            edge = (simplex[j], simplex[(j + 1) % 3])
            edge = tuple(sorted(edge))

            # Calculate the squared edge length
            p1 = points[edge[0]]
            p2 = points[edge[1]]
            edge_length_sq = np.sum((p1 - p2) ** 2)

            # Keep track of the edge if it passes the alpha test
            if edge_length_sq < alpha ** 2:
                edges[edge] = edge_length_sq

    # Create LineStrings from the edges
    boundary_lines = []
    for edge, _ in edges.items():
        boundary_lines.append(LineString([points[edge[0]], points[edge[1]]]))

    # Create the final shape
    boundary = MultiLineString(boundary_lines)

    # Try to get polygons from the lines
    polygons = list(polygonize(boundary))

    if len(polygons) == 0:
        return None

    # Merge all polygons
    result = unary_union(polygons)

    # Ha return_largest_only=True és MultiPolygon az eredmény, csak a legnagyobbat adjuk vissza
    if return_largest_only and isinstance(result, MultiPolygon) and len(result.geoms) > 0:
        return max(result.geoms, key=lambda p: p.area)

    return result


def draw_alpha_shape(img, alpha=20.0, return_largest_polygon=False):
    if img is None:
        raise ValueError("Input image is None")

    # Kép másolása színes formában, hogy lehessen színt rajzolni
    color_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    # Fehér pontok kinyerése
    white_pixels = np.where(img > 128)
    points = np.column_stack((white_pixels[1], white_pixels[0]))

    if len(points) < 4:
        print("Nincs elég fehér pont az alpha shape-hez (minimum 4 szükséges)")
        return color_img, points, None

    # Alpha shape kiszámítása
    shape = alpha_shape(points, alpha, return_largest_only=return_largest_polygon)

    if shape is None:
        print("Nem sikerült érvényes alpha shape-et létrehozni.")
        return color_img, points, None

    # Körvonal kirajzolása OpenCV-vel
    if isinstance(shape, Polygon):
        # Külső kontúr
        exterior = np.array(shape.exterior.coords, dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(color_img, [exterior], isClosed=True, color=(0, 0, 255), thickness=2)

        # Lyukak
        for interior in shape.interiors:
            hole = np.array(interior.coords, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(color_img, [hole], isClosed=True, color=(0, 0, 255), thickness=2)

    elif isinstance(shape, MultiPolygon):
        for poly in shape.geoms:
            exterior = np.array(poly.exterior.coords, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(color_img, [exterior], isClosed=True, color=(0, 0, 255), thickness=2)

            for interior in poly.interiors:
                hole = np.array(interior.coords, dtype=np.int32).reshape((-1, 1, 2))
                cv2.polylines(color_img, [hole], isClosed=True, color=(0, 0, 255), thickness=2)

    return color_img, points, shape


def apply_existing_alpha_shape(img, shape, color=(0, 0, 255), thickness=2):
    # Ellenőrizzük, hogy a kép színes-e, ha nem, konvertáljuk
    if len(img.shape) == 2 or img.shape[2] == 1:
        img_with_shape = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    else:
        img_with_shape = img.copy()

    # Ha nincs érvényes shape, visszaadjuk az eredeti képet
    if shape is None:
        return img_with_shape

    # Körvonal kirajzolása OpenCV-vel
    if isinstance(shape, Polygon):
        # Külső kontúr
        exterior = np.array(shape.exterior.coords, dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(img_with_shape, [exterior], isClosed=True, color=color, thickness=thickness)

        # Lyukak
        for interior in shape.interiors:
            hole = np.array(interior.coords, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(img_with_shape, [hole], isClosed=True, color=color, thickness=thickness)

    elif isinstance(shape, MultiPolygon):
        for poly in shape.geoms:
            exterior = np.array(poly.exterior.coords, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(img_with_shape, [exterior], isClosed=True, color=color, thickness=thickness)

            for interior in poly.interiors:
                hole = np.array(interior.coords, dtype=np.int32).reshape((-1, 1, 2))
                cv2.polylines(img_with_shape, [hole], isClosed=True, color=color, thickness=thickness)

    return img_with_shape


# Példa használat - Felhasználóbarát verzió
if __name__ == "__main__":
    # Közvetlenül állítsd be a paramétereket itt
    image_path = "base_bg_frame.png"  # A feldolgozandó kép elérési útja
    alpha = 15.0                # Alpha érték az algoritmushoz
    output_path = "chatgpt.png"  # Ide menti az eredményt
    return_largest_polygon = True  # False: minden poligont megtart, True: csak a legnagyobb poligont tartja meg

    print(f"Processing image: {image_path}")
    print(f"Alpha value: {alpha}")
    print(f"Output will be saved to: {output_path}")
    print(f"Return largest polygon only: {return_largest_polygon}")

    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    try:
        # Ha nincs meg a file, ellenőrizzük
        if not os.path.exists(image_path):
            print(f"Hiba: A megadott kép ({image_path}) nem található!")
        else:
            res_img, points, shape = draw_alpha_shape(img, alpha, return_largest_polygon)
            cv2.imshow("Image", res_img)
            cv2.imwrite(output_path, res_img)
            cv2.waitKey(0)
            print(f"Feldolgozás sikeres! Az eredmény mentve: {output_path}")
    except Exception as e:
        print(f"Hiba történt a feldolgozás során: {e}")