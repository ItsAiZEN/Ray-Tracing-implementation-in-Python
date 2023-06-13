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

    image_array = np.zeros((args.width, args.height, 3))

    # Step 1:
    # Step 1.1. Discover the location of the pixel on the camera’s screen (using camera parameters).
    # Step 1.2. Construct a ray from the camera through that pixel.
    # we have the following parameters: position, look_at, up_vector, screen_distance, screen_width

    # calculate the center of the image (Pc)
    camera.look_at = camera.look_at / np.linalg.norm(camera.look_at)
    camera.up_vector = camera.up_vector / np.linalg.norm(camera.up_vector)
    image_center = camera.position + np.array(camera.look_at) * camera.screen_distance

    # calculate Vright and Vup
    v_right = np.cross(camera.look_at, camera.up_vector)
    v_right = v_right / np.linalg.norm(v_right)
    v_up = np.cross(v_right, camera.look_at)
    v_up = v_up / np.linalg.norm(v_up)

    # calculate the ratio between the screen width and the image width (R)
    ratio = camera.screen_width / args.width

    for i in range(args.width):
        for j in range(args.height):
            # Step 1:
            # Step 1.1. Discover the location of the pixel on the camera’s screen (using camera parameters).
            # Step 1.2. Construct a ray from the camera through that pixel.
            # we have the following parameters: position, look_at, up_vector, screen_distance, screen_width

            # calculate the ray direction (R)

            # !!! we change the placement of i and j according to the presentation !!! #

            ray = image_center - v_right * ratio * (j - math.floor(args.width / 2)) - v_up * ratio * (
                    i - math.floor(args.height / 2)) - camera.position
            ray = ray / np.linalg.norm(ray)
            # we subtract the camera position because we want the ray to start from the camera position
            # !!! maybe subtracting the camera position is wrong !!! #

            # Step 2 & 3:
            # Check the intersection of the ray with all surfaces in the scene, then sift through the results for the
            # closest intersection of the ray. This is the surface that will be seen in the image.
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
                        b = 2 *(V • (P0 - O))
                        c = ||P0 - O||****2 - r**2 (= 0 ?)
                    """

                    # ray is a 3D vector, sphere is a Sphere object which has a center point and a radius
                    # V is the ray direction, P0 is the ray origin, O is the sphere center, r is the sphere radius
                    # the result (t) is the distance from the ray origin to the intersection point

                    coefficients = [1, np.dot(2*ray, np.array(camera.position) - np.array(surface.position)),
                                    np.linalg.norm(np.array(camera.position) - np.array(
                                        surface.position)) ** 2 - surface.radius ** 2]
                    # !!! change to 0 from np.linalg.norm(np.array(camera.position) - np.array( surface.position)) ** 2 - surface.radius ** 2
                    # in the third coefficient !!!
                    # get only real roots
                    # check if the discriminant is negative, if so, there are no real roots
                    discriminant = coefficients[1] ** 2 - 4 * coefficients[0] * coefficients[2]
                    if discriminant >= 0:
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
                    surface_normal = np.array(surface.normal)
                    surface_normal = surface_normal / np.linalg.norm(surface_normal)
                    if np.dot(ray, surface_normal) != 0:
                        t = -(np.dot(camera.position, surface_normal) - surface.offset) / np.dot(ray, surface_normal)
                        # !!! maybe change normal to -normal !!!
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
                        plane_normal = np.array(plane.normal)
                        plane_normal = plane_normal / np.linalg.norm(plane_normal)
                        if np.dot(ray, plane_normal) != 0:
                            t = -(np.dot(camera.position, plane_normal) - plane.offset) / np.dot(ray, plane_normal)
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

            # Step 4:
            # Calculate the color of the pixel based on the surface properties and the light sources in the scene.
            # 4.1. Go over each light in the scene.
            # 4.2. Add the value it induces on the surface.

            # if the ray intersects with a surface
            if closest_surface[0] is None:
                image_array[i][j] = scene_settings.background_color
            else:
                print(type(closest_surface[0]), "point: ", closest_surface[1], "distance: ", closest_intersection_distance)  # !!! good for debugging !!!
                # calculate the normal vector of the surface at the intersection point
                if type(closest_surface[0]) == Sphere:
                    normal = closest_surface[1] - closest_surface[0].position
                    normal = normal / np.linalg.norm(normal)

                elif type(closest_surface[0]) == InfinitePlane:
                    normal = closest_surface[0].normal
                    normal = normal / np.linalg.norm(normal)

                elif type(closest_surface[0]) == Cube:
                    center = closest_surface[0].position
                    edge_length = closest_surface[0].scale
                    # check which plane is the closest to the intersection point
                    closet_plane = -1
                    minimal_distance = float('inf')
                    if abs(closest_surface[1][0] - center[0] + edge_length / 2) < minimal_distance:
                        closet_plane = 0
                    if abs(closest_surface[1][0] + center[0] + edge_length / 2) < minimal_distance:
                        closet_plane = 1
                    if abs(closest_surface[1][1] - center[1] + edge_length / 2) < minimal_distance:
                        closet_plane = 2
                    if abs(closest_surface[1][1] + center[1] + edge_length / 2) < minimal_distance:
                        closet_plane = 3
                    if abs(closest_surface[1][2] - center[2] + edge_length / 2) < minimal_distance:
                        closet_plane = 4
                    if abs(closest_surface[1][2] + center[2] + edge_length / 2) < minimal_distance:
                        closet_plane = 5

                    if closet_plane == 0:
                        normal = np.array([1, 0, 0])
                    elif closet_plane == 1:
                        normal = np.array([-1, 0, 0])
                    elif closet_plane == 2:
                        normal = np.array([0, 1, 0])
                    elif closet_plane == 3:
                        normal = np.array([0, -1, 0])
                    elif closet_plane == 4:
                        normal = np.array([0, 0, 1])
                    elif closet_plane == 5:
                        normal = np.array([0, 0, -1])
                    normal = normal / np.linalg.norm(normal)

                # calculate the intersection to the camera
                view = camera.position - closest_surface[1]
                view = view / np.linalg.norm(view)

                material_index = closest_surface[0].material_index
                material_counter = 0
                surface_material = None
                for object in objects:
                    if type(object) == Material:
                        material_counter += 1
                        if material_counter == material_index:
                            surface_material = object
                            break

                material_diffuse = surface_material.diffuse_color
                material_specular = surface_material.specular_color
                final_color = np.array([0, 0, 0])
                for light in objects:
                    if type(light) is not Light:
                        continue
                    else:
                        # light_intensity = 1 / light.radius ** 2
                        light_intensity = 1
                        # TODO calculate true light intensity
                        # we will calculate the light intensity by creating a grid relative to the location of the light
                        # and its radius, the grid will be scene_settings.root_number_shadow_rays wide, then we will
                        # calculate the intensity of the light by
                        # light intensity = (1− shadow intensity) * 1 + shadow intensity *
                        # ( % of rays that hit the points from the light source)
                        shadow_intensity = light.shadow_intensity
                        # SigmaL(Kd(N.L)+Ks(V.R)^n)SLIL
                        for color in range(3):
                            intersection_to_light = light.position - closest_surface[1]
                            intersection_to_light = intersection_to_light / np.linalg.norm(intersection_to_light)

                            intersection_to_reflected_light = 2 * np.dot(intersection_to_light,
                                                                         normal) * normal - intersection_to_light
                            intersection_to_reflected_light = intersection_to_reflected_light / np.linalg.norm(
                                intersection_to_reflected_light)

                            # !!! intersection to light might be a bad calculation !!!

                            diffusion_and_specular = (material_diffuse[color] * np.dot(normal, intersection_to_light) + \
                                                      material_specular[color] * np.dot(view,
                                                                                        intersection_to_reflected_light) ** surface_material.shininess)
                                                     # !!! * (1 - shadow_intensity) * light_intensity !!!
                            # TODO shininess is makes the result way too high
                            # TODO if its 1- shadow_intensity or  just shadow_intensity
                            # TODO also maybe we need to add an if statement for this case

                            # TODO calculate the light intensity based on the distance from the light source with the
                            # TODO add relevant calculation for specular intensity and specular color together
                            # or maybe just update the color and specular color and diffuse beforehand
                            # TODO maybe the color within the color formula instead of at the end
                            # formula 1/(a + b*d + c*d^2)?
                            """light intensity = (1− shadow intensity)*1 +shadow intensity*(% of rays that hit the points from the light source)"""

                            final_color[
                                color] += scene_settings.background_color[
                                             color] * surface_material.transparency + diffusion_and_specular * \
                                          (1 - surface_material.transparency) * light.color[color] * 255 + \
                                          surface_material.reflection_color[color]
                            print("diffusion and specular", diffusion_and_specular)

            image_array[
                i, j] = final_color  # !!! changed the multiplication by * light.color to inside the final color calculation
            # print(f"pixel {i} {j} {final_color}")

    # # Dummy result
    # image_array = np.zeros((500, 500, 3))
    for i in range(args.height):
        for j in range(args.width):
            for k in range(3):
                image_array[i][j][k] = int(image_array[i][j][k])
    print(image_array)
    image_array = image_array.clip(0, 255)

    # Save the output image
    save_image(image_array)


if __name__ == '__main__':
    main()
