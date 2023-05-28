import argparse
from PIL import Image
import numpy as np
import math

from camera import Camera
from light import Light
from material import Material
from scene_settings import SceneSettings
from surfaces.cube import Cube
from surfaces.infinite_plane import InfinitePlane
from surfaces.sphere import Sphere


def parse_scene_file(file_path):
    objects = []
    camera = None
    scene_settings = None
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            obj_type = parts[0]
            params = [float(p) for p in parts[1:]]
            if obj_type == "cam":
                camera = Camera(params[:3], params[3:6], params[6:9], params[9], params[10])
            elif obj_type == "set":
                scene_settings = SceneSettings(params[:3], params[3], params[4])
            elif obj_type == "mtl":
                material = Material(params[:3], params[3:6], params[6:9], params[9], params[10])
                objects.append(material)
            elif obj_type == "sph":
                sphere = Sphere(params[:3], params[3], int(params[4]))
                objects.append(sphere)
            elif obj_type == "pln":
                plane = InfinitePlane(params[:3], params[3], int(params[4]))
                objects.append(plane)
            elif obj_type == "box":
                cube = Cube(params[:3], params[3], int(params[4]))
                objects.append(cube)
            elif obj_type == "lgt":
                light = Light(params[:3], params[3:6], params[6], params[7], params[8])
                objects.append(light)
            else:
                raise ValueError("Unknown object type: {}".format(obj_type))
    return camera, scene_settings, objects


def save_image(image_array):
    image = Image.fromarray(np.uint8(image_array))

    # Save the image to a file
    image.save("scenes/Spheres.png")


"""
The general process of ray tracing is:

1:
Shoot a ray through each pixel in the image. For each pixel, you should:
1.1. Discover the location of the pixel on the camera’s screen (using camera parameters).
1.2. Construct a ray from the camera through that pixel.

2:
Check the intersection of the ray with all surfaces in the scene 
(you can add optimizations such as BSP trees if you wish but they are not mandatory).

3:
Find the nearest intersection of the ray. This is the surface that will be seen in the image.

4:
Compute the color of the surface:
4.1. Go over each light in the scene.
4.2. Add the value it induces on the surface.

5:
Find out whether the light hits the surface or not:
5.1. Shoot rays from the light towards the surface.
5.2. Find whether the ray intersects any other surfaces before the required surface - if so,
 the surface is occluded from the light and the light does not affect it
  (or partially affects it because of the shadow intensity parameter).

6:
Produce soft shadows, as explained below:
6.1. Shoot several rays from the proximity of the light to the surface.
6.2. Find out how many of them hit the required surface.
"""

"""output color = (background color) · transparency + (diffuse + specular)·(1−transparency) + (reflection color)"""
"""light intensity = (1− shadow intensity)*1 +shadow intensity*(% of rays that hit the points from the light source)"""


def main():
    parser = argparse.ArgumentParser(description='Python Ray Tracer')
    parser.add_argument('scene_file', type=str, help='Path to the scene file')
    parser.add_argument('output_image', type=str, help='Name of the output image file')
    parser.add_argument('--width', type=int, default=500, help='Image width')
    parser.add_argument('--height', type=int, default=500, help='Image height')
    args = parser.parse_args()

    # Parse the scene file
    camera, scene_settings, objects = parse_scene_file(args.scene_file)

    # TODO: Implement the ray tracer

    our_image_array = np.zeros((args.width, args.height, 3))

    for i in range(args.width):
        for j in range(args.height):
            # Step 1:
            # Step 1.1. Discover the location of the pixel on the camera’s screen (using camera parameters).
            # Step 1.2. Construct a ray from the camera through that pixel.
            # we have the following parameters: position, look_at, up_vector, screen_distance, screen_width

            # calculate the center of the image (Pc)
            image_center = camera.position + camera.look_at * camera.screen_distance

            # calculate Vright and Vup
            v_right = np.cross(camera.look_at, camera.up_vector)
            v_right = v_right / np.linalg.norm(v_right)
            v_up = np.cross(v_right, camera.look_at)
            v_up = v_up / np.linalg.norm(v_up)

            # calculate the ratio between the screen width and the image width (R)
            ratio = camera.screen_width / args.width

            # calculate the ray direction (R)
            ray = image_center + v_right * ratio * (i - math.floor(args.width / 2)) - v_up * ratio * (
                    j - math.floor(args.height / 2)) - camera.position
            # we subtract the camera position because we want the ray to start from the camera position

            # Step 2:
            # Check the intersection of the ray with all surfaces in the scene
            closest_surface = (None, float('inf'))
            closest_intersection_distance = float('inf')
            for surface in objects:
                # check if the surface is a sphere
                if type(surface) in [Light, Material]:
                    pass
                elif type(surface) == Sphere:
                    """
                    Solve quadratic equation:
                    at2 + bt + c = 0
                    where:
                        a = 1
                        b = 2 V • (P0 - O)
                        c = |P0 - O|2 - r 2 = 0
                    """

                    # ray is a 3D vector, sphere is a Sphere object which has a center point and a radius
                    # V is the ray direction, P0 is the ray origin, O is the sphere center, r is the sphere radius
                    # the result (t) is the distance from the ray origin to the intersection point

                    coefficients = [1, 2 * np.dot(ray, camera.position - surface.center),
                                    np.linalg.norm(camera.position - surface.center) ** 2 - surface.radius ** 2]
                    roots = np.roots(coefficients)
                    for t in roots:
                        if 0 < t < closest_intersection_distance:
                            point_of_intersection = camera.position + t * ray
                            closest_intersection_distance = t
                            closest_surface = (surface, point_of_intersection)

                elif type(surface) == InfinitePlane:
                    """
                    t = -(P0 • N - d) / (V • N)
                    """
                    # check if the ray is not parallel to the plane, otherwise there is no intersection, we can skip
                    # and we avoid division by zero
                    if np.dot(ray, surface.normal) != 0:
                        t = -(np.dot(camera.position, surface.normal) - surface.offset) / np.dot(ray, surface.normal)
                        if 0 < t < closest_intersection_distance:
                            point_of_intersection = camera.position + t * ray
                            closest_intersection_distance = t
                            closest_surface = (surface, point_of_intersection)

                elif type(surface) == Cube:
                    # we will use the slab method, we know the cube is axis aligned, we can check each axis separately
                    # we will create 6 planes, one for each side of the cube, and check if the ray intersects with them
                    # instead of creating objects for each plane, we will use the plane equation:
                    # ax + by + cz + d = 0
                    # where a, b, c are the normal vector of the plane, and d is the offset
                    # we will use the normal vector and the offset to create the planes
                    center = surface.position
                    edge_length = surface.scale
                    # infinite planes are defined by (normal, offset, material)
                    plane1 = InfinitePlane(np.array([1, 0, 0]), center[0] + edge_length / 2, None)
                    plane2 = InfinitePlane(np.array([-1, 0, 0]), -center[0] + edge_length / 2, None)
                    plane3 = InfinitePlane(np.array([0, 1, 0]), center[1] + edge_length / 2, None)
                    plane4 = InfinitePlane(np.array([0, -1, 0]), -center[1] + edge_length / 2, None)
                    plane5 = InfinitePlane(np.array([0, 0, 1]), center[2] + edge_length / 2, None)
                    plane6 = InfinitePlane(np.array([0, 0, -1]), -center[2] + edge_length / 2, None)

                    planes = [plane1, plane2, plane3, plane4, plane5, plane6]

                    for plane in planes:
                        # check if the ray is not parallel to the plane, otherwise there is no intersection, we can skip
                        # and we avoid division by zero
                        if np.dot(ray, plane.normal) != 0:
                            t = -(np.dot(camera.position, plane.normal) - plane.offset) / np.dot(ray, plane.normal)
                            if t > 0:
                                point_of_intersection = camera.position + t * ray
                                # check if the point is inside the cube
                                if center[0] - edge_length / 2 <= point_of_intersection[0] <= center[
                                    0] + edge_length / 2 and center[1] - edge_length / 2 <= point_of_intersection[1] <= \
                                        center[1] + edge_length / 2 and center[2] - edge_length / 2 <= \
                                        point_of_intersection[2] <= center[2] + edge_length / 2:
                                    if 0 < t < closest_intersection_distance:
                                        closest_intersection_distance = t
                                        closest_surface = (surface, point_of_intersection)

    # Dummy result
    image_array = np.zeros((500, 500, 3))

    # Save the output image
    save_image(image_array)


if __name__ == '__main__':
    main()
