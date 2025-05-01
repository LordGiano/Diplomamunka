import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
from shapely.geometry import LineString, MultiLineString, Polygon, MultiPolygon
from shapely.ops import polygonize, unary_union
import alphashape
import cv2
import os


def alpha_shape(points, alpha, return_largest_only=False):
    """
    Compute the alpha shape (concave hull) of a set of points.

    Parameters:
    points : numpy array of shape (n, 2)
        The coordinates of the points
    alpha : float
        Alpha value determines the 'tightness' of the resulting shape.
        Lower values result in a tighter fit around the points.
    return_largest_only : bool
        If True, only return the largest polygon from the result.
        If False, return all polygons as a MultiPolygon.

    Returns:
    shapely.geometry.Polygon or MultiPolygon : The alpha shape polygon(s)
    """
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


def draw_alpha_shape(image_path, alpha=20.0, output_path="alpha_shape_result.png", return_largest_polygon=False):
    """
    Draw an alpha shape around white points in a black and white image.
    Can focus only on the largest connected component.

    Parameters:
    image_path : str
        Path to the black and white image
    alpha : float
        Alpha value for the alpha shape algorithm
    output_path : str
        Path where the result will be saved
    largest_only : bool
        If True, only process the largest connected component
    return_largest_polygon : bool
        If True, only return the largest polygon from the alpha shape result

    Returns:
    None
    """
    # Read the image
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        raise ValueError(f"Cannot read image from {image_path}")

    # Extract all white points - minden fehér pontot figyelembe veszünk
    white_pixels = np.where(img > 128)
    points = np.column_stack((white_pixels[1], white_pixels[0]))
    img_shape = img.shape

    if len(points) < 4:
        print("Not enough white points to create an alpha shape (minimum 4 required)")
        return

    # Compute alpha shape with the new parameter
    shape = alpha_shape(points, alpha, return_largest_only=return_largest_polygon)

    if shape is None:
        print("Could not create a valid alpha shape. Try adjusting the alpha value.")
        return

    # Create figure
    plt.figure(figsize=(10, 10))

    # Plot original image as background
    plt.imshow(img, cmap='gray')

    # Plot the alpha shape(s)
    if isinstance(shape, Polygon):
        # Single polygon
        x, y = shape.exterior.xy
        plt.plot(x, y, 'r-', linewidth=2)

        # Plot any holes
        for interior in shape.interiors:
            x, y = interior.xy
            plt.plot(x, y, 'r-', linewidth=2)

    elif isinstance(shape, MultiPolygon):
        # Multiple polygons
        for polygon in shape.geoms:
            x, y = polygon.exterior.xy
            plt.plot(x, y, 'r-', linewidth=2)

            # Plot any holes in each polygon
            for interior in polygon.interiors:
                x, y = interior.xy
                plt.plot(x, y, 'r-', linewidth=2)

    # Set the axis limits and invert y-axis (image coordinates)
    plt.xlim(0, img_shape[1])
    plt.ylim(img_shape[0], 0)

    # Remove axes
    plt.axis('off')

    # Set equal aspect ratio
    plt.gca().set_aspect('equal', adjustable='box')

    # Save the result
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0, dpi=300)

    # Show the result
    plt.show()

    print(f"Result saved to {output_path}")

    # Return the processed image, largest contour, and shape for potential further processing
    return img, points, shape


# Példa használat - Felhasználóbarát verzió
if __name__ == "__main__":
    # Közvetlenül állítsd be a paramétereket itt
    image_path = "base_bg_frame.png"  # A feldolgozandó kép elérési útja
    alpha = 10.0                # Alpha érték az algoritmushoz
    output_path = "eredmeny.png"  # Ide menti az eredményt
    return_largest_polygon = True  # False: minden poligont megtart, True: csak a legnagyobb poligont tartja meg

    print(f"Processing image: {image_path}")
    print(f"Alpha value: {alpha}")
    print(f"Output will be saved to: {output_path}")
    print(f"Return largest polygon only: {return_largest_polygon}")

    try:
        # Ha nincs meg a file, ellenőrizzük
        if not os.path.exists(image_path):
            print(f"Hiba: A megadott kép ({image_path}) nem található!")
        else:
            # Meghívjuk a funkciót a megadott paraméterekkel
            draw_alpha_shape(image_path, alpha, output_path, return_largest_polygon)
            print(f"Feldolgozás sikeres! Az eredmény mentve: {output_path}")
    except Exception as e:
        print(f"Hiba történt a feldolgozás során: {e}")