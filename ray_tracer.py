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

import time


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


def main():
    parser = argparse.ArgumentParser(description='Python Ray Tracer')
    parser.add_argument('scene_file', type=str, help='Path to the scene file')
    parser.add_argument('output_image', type=str, help='Name of the output image file')
    parser.add_argument('--width', type=int, default=500, help='Image width')
    parser.add_argument('--height', type=int, default=500, help='Image height')
    args = parser.parse_args()

    # Parse the scene file
    camera, scene_settings, objects = parse_scene_file(args.scene_file)
    image_array = np.zeros((args.width, args.height, 3))

    camera.look_at = np.array(camera.look_at) - np.array(camera.position)
    camera.look_at = camera.look_at / np.linalg.norm(camera.look_at)
    camera.up_vector = camera.up_vector / np.linalg.norm(camera.up_vector)
    image_center = camera.position + np.array(camera.look_at) * camera.screen_distance

    v_right = np.cross(camera.look_at, camera.up_vector)
    v_right = v_right / np.linalg.norm(v_right)
    v_up = np.cross(v_right, camera.look_at)
    v_up = v_up / np.linalg.norm(v_up)

    ratio = camera.screen_width / args.width

    for i in range(args.height):
        for j in range(args.width):
            ray = image_center - v_right * ratio * (j - math.floor(args.width / 2)) - v_up * ratio * (
                    i - math.floor(args.height / 2)) - camera.position
            ray = ray / np.linalg.norm(ray)

            ray_tracer(ray, i, j, image_array, objects, scene_settings, camera.position, 1)

    for i in range(args.height):
        for j in range(args.width):
            for k in range(3):
                image_array[i][j][k] = int(image_array[i][j][k])
    image_array = image_array.clip(0, 255)

    # Save the output image
    save_image(image_array)


def ray_tracer(ray, i, j, image_array, objects, scene_settings, origin_point, depth):
    if depth > 1:
        return np.array([0, 0, 0])
    closest_surface = (None, float('inf'))
    closest_intersection_distance = float('inf')
    for surface in objects:
        # check if the surface is a sphere
        if type(surface) in [Light, Material]:
            pass
        elif type(surface) == Sphere:

            coefficients = [1, np.dot(2 * ray, np.array(origin_point) - np.array(surface.position)),
                            np.linalg.norm(np.array(origin_point) - np.array(
                                surface.position)) ** 2 - surface.radius ** 2]

            discriminant = (coefficients[1] ** 2) - (4 * coefficients[0] * coefficients[2])
            if discriminant >= 0:
                roots = [(-coefficients[1] - math.sqrt(discriminant)) / (2 * coefficients[0]), ( -coefficients[1] + math.sqrt(discriminant)) / (2 * coefficients[0])]
                for t in roots:
                    if 0.00001 < t < closest_intersection_distance:
                        point_of_intersection = origin_point + t * ray
                        closest_intersection_distance = t
                        closest_surface = (surface, point_of_intersection)

        elif type(surface) == InfinitePlane:

            surface_normal = np.array(surface.normal)
            surface_normal = surface_normal / np.linalg.norm(surface_normal)
            if np.dot(ray, surface_normal) != 0:
                t = -(np.dot(origin_point, surface_normal) - surface.offset) / np.dot(ray, surface_normal)
                if 0.00001 < t < closest_intersection_distance:
                    point_of_intersection = origin_point + t * ray
                    closest_intersection_distance = t
                    closest_surface = (surface, point_of_intersection)

        elif type(surface) == Cube:

            center = surface.position
            edge_length = surface.scale
            plane1 = InfinitePlane(np.array([1, 0, 0]), center[0] + edge_length / 2, None)
            plane2 = InfinitePlane(np.array([-1, 0, 0]), center[0] + edge_length / 2, None)
            plane3 = InfinitePlane(np.array([0, 1, 0]), center[1] + edge_length / 2, None)
            plane4 = InfinitePlane(np.array([0, -1, 0]), center[1] + edge_length / 2, None)
            plane5 = InfinitePlane(np.array([0, 0, 1]), center[2] + edge_length / 2, None)
            plane6 = InfinitePlane(np.array([0, 0, -1]), center[2] + edge_length / 2, None)

            planes = [plane1, plane2, plane3, plane4, plane5, plane6]

            for plane in planes:

                plane_normal = np.array(plane.normal)
                plane_normal = plane_normal / np.linalg.norm(plane_normal)
                if np.dot(ray, plane_normal) != 0:
                    t = -(np.dot(origin_point, plane_normal) - plane.offset) / np.dot(ray, plane_normal)
                    if t > 0.00001:
                        point_of_intersection = origin_point + t * ray
                        # check if the point is inside the cube
                        if (center[0] - (edge_length / 2) - 0.00001 <= point_of_intersection[0] <= center[0] + (edge_length / 2) + 0.00001)\
                            and \
                            (center[1] - (edge_length / 2) - 0.00001 <= point_of_intersection[1] <= center[1] + (edge_length / 2) + 0.00001)\
                            and \
                            (center[2] - (edge_length / 2) - 0.00001 <= point_of_intersection[2] <= center[2] + (edge_length / 2) + 0.00001):
                            if 0.00001 < t < closest_intersection_distance:
                                closest_intersection_distance = t
                                closest_surface = (surface, point_of_intersection)

# TODO maybe add thresholds

    if closest_surface[0] is None:
        if depth == 1:
            image_array[i][j] = np.array(scene_settings.background_color) * 255
        return np.array(scene_settings.background_color) * 255
    else:
        if type(closest_surface[0]) == Sphere:
            normal = closest_surface[1] - closest_surface[0].position
            normal = normal / np.linalg.norm(normal)

        elif type(closest_surface[0]) == InfinitePlane:
            normal = closest_surface[0].normal
            normal = normal / np.linalg.norm(normal)

        elif type(closest_surface[0]) == Cube:
            center = closest_surface[0].position
            edge_length = closest_surface[0].scale
            closet_plane = -1
            minimal_distance = float('inf')
            if abs(closest_surface[1][0] - (center[0] + edge_length / 2)) < minimal_distance:
                closet_plane = 0
                minimal_distance = abs(closest_surface[1][0] - (center[0] + edge_length / 2))

            if abs(closest_surface[1][0] - (center[0] - edge_length / 2)) < minimal_distance:
                closet_plane = 1
                minimal_distance = abs(closest_surface[1][0] - (center[0] - edge_length / 2))

            if abs(closest_surface[1][1] - (center[1] + edge_length / 2)) < minimal_distance:
                closet_plane = 2
                minimal_distance = abs(closest_surface[1][1] - (center[1] + edge_length / 2))

            if abs(closest_surface[1][1] - (center[1] - edge_length / 2)) < minimal_distance:
                closet_plane = 3
                minimal_distance = abs(closest_surface[1][1] - (center[1] - edge_length / 2))

            if abs(closest_surface[1][2] - (center[2] + edge_length / 2)) < minimal_distance:
                closet_plane = 4
                minimal_distance = abs(closest_surface[1][2] - (center[2] + edge_length / 2))

            if abs(closest_surface[1][2] - (center[2] - edge_length / 2)) < minimal_distance:
                closet_plane = 5
                minimal_distance = abs(closest_surface[1][2] - (center[2] - edge_length / 2))

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

        view = -(origin_point - closest_surface[1])
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
        return_color = np.zeros(3)

        for light in objects:
            if type(light) is not Light:
                continue
            else:

                shadow_intensity = light.shadow_intensity

                intersection_to_light = light.position - closest_surface[1]
                intersection_to_light = intersection_to_light / np.linalg.norm(intersection_to_light)

                intersection_to_reflected_light = 2 * np.dot(intersection_to_light,
                                                             normal) * normal - intersection_to_light
                intersection_to_reflected_light = intersection_to_reflected_light / np.linalg.norm(
                    intersection_to_reflected_light)

                reflected_ray = ray - 2 * np.dot(ray, normal) * normal
                reflected_ray = reflected_ray / np.linalg.norm(reflected_ray)

                # bounding_box_width = light.radius
                #
                # grid_ratio = bounding_box_width / scene_settings.root_number_shadow_rays
                #
                # light_v_right = np.random.randn(3)
                # light_v_right -= np.dot(light_v_right, -intersection_to_light) * (-intersection_to_light) / np.linalg.norm(-intersection_to_light) ** 2
                # light_v_right /= np.linalg.norm(light_v_right)
                #
                # light_v_up = np.cross(-intersection_to_light, light_v_right)
                # light_v_up /= np.linalg.norm(light_v_up)
                #
                # shadow_rays_count = 0
                #
                # for x in range(int(scene_settings.root_number_shadow_rays)):
                #     for y in range(int(scene_settings.root_number_shadow_rays)):
                #
                #         point_on_grid = light.position - light_v_right * grid_ratio * (
                #                 x - math.floor(scene_settings.root_number_shadow_rays / 2)) - light_v_up * grid_ratio * (y - math.floor(
                #                     scene_settings.root_number_shadow_rays) / 2) + ((
                #                            np.random.rand() - 0.5) * grid_ratio * light_v_right + (
                #                            np.random.rand() - 0.5) * grid_ratio * light_v_up)
                #         grid_ray = - (point_on_grid - closest_surface[1])
                #         grid_ray = grid_ray / np.linalg.norm(grid_ray)
                #
                #         is_hit = ray_tracer_shadow(grid_ray, objects, closest_surface[1], point_on_grid)
                #         if is_hit:
                #             shadow_rays_count += 1
                #
                # light_intensity = (1 - shadow_intensity) * 1 + shadow_intensity * (
                #             shadow_rays_count / (scene_settings.root_number_shadow_rays ** 2))

                light_intensity = 1

                diffusion_and_specular = (np.array(material_diffuse) * np.dot(normal, intersection_to_light) + \
                                          np.array(material_specular) * np.dot(view,
                                                                               intersection_to_reflected_light) ** surface_material.shininess) * light_intensity * light.specular_intensity

                return_color += np.array(diffusion_and_specular) * \
                                (1 - surface_material.transparency) * np.array(light.color) * 255

        recursion_color = ray_tracer(reflected_ray, i, j, image_array, objects, scene_settings, closest_surface[1], depth + 1)
        return_color += 255 * np.array(scene_settings.background_color) * np.array(surface_material.transparency) + np.array(surface_material.reflection_color) * recursion_color

        if depth == 1:
            image_array[i, j] = return_color
        return return_color


def ray_tracer_shadow(ray, objects, original_intersection_point, point_on_grid):
    closest_surface = (None, float('inf'))
    closest_intersection_distance = float('inf')
    point_of_intersection = None
    for surface in objects:
        if type(surface) in [Light, Material]:
            pass
        elif type(surface) == Sphere:

            coefficients = [1, np.dot(2 * ray, np.array(point_on_grid) - np.array(surface.position)),
                            np.linalg.norm(np.array(point_on_grid) - np.array(
                                surface.position)) ** 2 - surface.radius ** 2]

            discriminant = (coefficients[1] ** 2) - (4 * coefficients[0] * coefficients[2])
            if discriminant >= 0:
                roots = [(-coefficients[1] - math.sqrt(discriminant)) / (2 * coefficients[0]), ( -coefficients[1] + math.sqrt(discriminant)) / (2 * coefficients[0])]
                for t in roots:
                    if 0.00001 < t < closest_intersection_distance:
                        point_of_intersection = point_on_grid + t * ray
                        closest_intersection_distance = t
                        closest_surface = (surface, point_of_intersection)

        elif type(surface) == InfinitePlane:

            surface_normal = np.array(surface.normal)
            surface_normal = surface_normal / np.linalg.norm(surface_normal)
            if np.dot(ray, surface_normal) != 0:
                t = -(np.dot(point_on_grid, surface_normal) - surface.offset) / np.dot(ray, surface_normal)

                if 0.00001 < t < closest_intersection_distance:
                    point_of_intersection = point_on_grid + t * ray
                    closest_intersection_distance = t
                    closest_surface = (surface, point_of_intersection)

        elif type(surface) == Cube:

            center = surface.position
            edge_length = surface.scale

            plane1 = InfinitePlane(np.array([1, 0, 0]), center[0] + edge_length / 2, None)
            plane2 = InfinitePlane(np.array([-1, 0, 0]), -center[0] + edge_length / 2, None)
            plane3 = InfinitePlane(np.array([0, 1, 0]), center[1] + edge_length / 2, None)
            plane4 = InfinitePlane(np.array([0, -1, 0]), -center[1] + edge_length / 2, None)
            plane5 = InfinitePlane(np.array([0, 0, 1]), center[2] + edge_length / 2, None)
            plane6 = InfinitePlane(np.array([0, 0, -1]), -center[2] + edge_length / 2, None)

            planes = [plane1, plane2, plane3, plane4, plane5, plane6]

            for plane in planes:

                plane_normal = np.array(plane.normal)
                plane_normal = plane_normal / np.linalg.norm(plane_normal)
                if np.dot(ray, plane_normal) != 0:
                    t = -(np.dot(point_on_grid, plane_normal) - plane.offset) / np.dot(ray, plane_normal)
                    if t > 0.00001:
                        point_of_intersection = point_on_grid + t * ray

                        if center[0] - edge_length / 2 <= point_of_intersection[0] <= center[
                            0] + edge_length / 2 and center[1] - edge_length / 2 <= point_of_intersection[1] <= \
                                center[1] + edge_length / 2 and center[2] - edge_length / 2 <= \
                                point_of_intersection[2] <= center[2] + edge_length / 2:
                            if 0.00001 < t < closest_intersection_distance:
                                closest_intersection_distance = t
                                closest_surface = (surface, point_of_intersection)

    is_hit = True
    for coord in range(3):
        if abs(closest_surface[1][coord] - original_intersection_point[coord]) > 0.00001:
            is_hit = False
            break

    return is_hit


if __name__ == '__main__':
    start = time.time()
    main()
    end = time.time()
    print("time :", end - start)


# TODO: comments and documentation
# TODO: transparency bonus
