import argparse
from PIL import Image
import numpy as np

from camera import Camera
from light import Light
from material import Material
from scene_settings import SceneSettings
from surfaces.cube import Cube
from surfaces.infinite_plane import InfinitePlane
from surfaces.sphere import Sphere

class Intersection:
    def __init__(self, t, primitive=None, point=None, normal=None):
        self.t = t
        self.primitive = primitive
        self.point = point
        self.normal = normal

class Ray:
    def __init__(self, origin, direction):
        self.origin = origin
        self.direction = direction


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
    # image_array is in [0,1]; scale to [0,255] for saving
    image = Image.fromarray(np.clip(image_array * 255.0, 0, 255).astype(np.uint8))
    # Save the image to a file
    image.save("scenes/Spheres.png")


def ConstructRayThroughPixel(camera, i, j, half_width, half_height, plane_center , pixel_size , right, up):
    u = ((i + 0.5) * pixel_size) - half_width   
    v = ((j + 0.5) * pixel_size) - half_height    

    pixel_pos = plane_center + u * right + v * up
    ray_direction = pixel_pos - camera.position
    ray_direction = ray_direction / np.linalg.norm(ray_direction)

    return Ray(camera.position,ray_direction)

def IntersectWithSphere(ray, primitive):
    L = primitive.position - ray.origin # separate per coordinate
    t_ca = np.dot(L, ray.direction)
    if (t_ca < 0):
        return float('inf'), None
    d_sqrd = np.dot(L, L) - t_ca**2
    r_sqrd = primitive.radius **2
    if (d_sqrd > r_sqrd):
        return float('inf'), None
    t_hc = np.sqrt(r_sqrd-d_sqrd)
    t = min(t_ca + t_hc , t_ca - t_hc)
    P = ray.origin + t*ray.direction
    return t,  (P-primitive.position)/np.linalg.norm(P-primitive.position) # second var is Normal


def IntersectWithCube(ray, primitive): 
    half_scale = primitive.scale / 2
    box_min = primitive.position - half_scale 
    box_max = primitive.position + half_scale 
    t_near = -float('inf') 
    t_far = float('inf')
    normal = np.zeros(3)
    normal_out = np.zeros(3) # if in cube

    for i in range(3):
        if ray.direction[i] == 0: 
            if ray.origin[i] < box_min[i] or ray.origin[i] > box_max[i]:
                return float('inf'), None
        else:
            t1 = (box_min[i] - ray.origin[i]) / ray.direction[i]
            t2 = (box_max[i] - ray.origin[i]) / ray.direction[i]
            t_in = min(t1, t2)
            t_out = max(t1, t2)
            if t_in > t_near:
                t_near = t_in
                normal = np.zeros(3)
                if (ray.direction[i] > 0):
                    normal[i] = -1 
                else:
                    normal[i] = 1
            
            #CHANGE
            # if we are inside cube
            if t_out < t_far:
                t_far = t_out
                normal_out = np.zeros(3)
                if (ray.direction[i] < 0):
                    normal_out[i] = -1 
                else:
                    normal_out[i] = 1

    if t_near > t_far:
        return float('inf'), None
    if t_far < 0:
        return float('inf'), None
    if t_near > 0:
        return t_near, normal
    else: 
        return t_far, normal_out #CHANGE only if needed

def IntersectWithPlane(ray, primitive):
    denominator = np.dot(ray.direction, primitive.normal)
    if abs(denominator) < 1e-6: # if ray parallel to plane CHANGE to global eps
        return float('inf'), None
    numerator = primitive.offset - np.dot(ray.origin, primitive.normal)
    t = numerator / denominator
    if t < 0:
        return float('inf'), None
    return t, primitive.normal

def Intersect(ray, primitive):
    if (isinstance(primitive,Sphere)):
        return IntersectWithSphere(ray, primitive)
    if (isinstance(primitive,Cube)):
        return IntersectWithCube(ray, primitive)
    else: #infinte plane
        return IntersectWithPlane(ray, primitive)


def FindIntersection(ray, primitives):
    min_t = float('inf')
    min_primitive = None
    min_normal = None
    for primitive in primitives:
        t, normal = Intersect(ray, primitive)
        if (t != float('inf') and t < min_t):
            min_primitive = primitive
            min_t = t
            min_normal = normal
    if min_primitive is None:
        return Intersection(float('inf'), None, None, None)
    min_point = ray.origin + min_t*ray.direction
    return Intersection(min_t, min_primitive, min_point, min_normal)

def filter_by_type(objects, type_names):
    result = []
    for obj in objects:
        for type in type_names:
            if obj.__class__.__name__ == type:
                result.append(obj)
    return result


def CalculateSoftShadow(hit, N , light, primitives, eps): # CHANGE Global eps
    p_bias = hit.point + hit.normal * eps

    L = p_bias - light.position
    L = L / np.linalg.norm(L)

    if abs(L[0]) < 0.9:
        helper = np.array([1.0, 0.0, 0.0])
    else:
        helper = np.array([0.0, 1.0, 0.0])
    U = np.cross(L, helper)
    U = U / np.linalg.norm(U)
    V = np.cross(L, U)
    V = V / np.linalg.norm(V)
    
    r = light.radius
    cell = (2.0 * r) / N
    unblocked = 0
    total = int(N)*int(N)
    for i in range(int(N)):
        for j in range(int(N)):
            bias_x = np.random.rand()
            bias_y = np.random.rand()
            x = -r + (i + bias_x) * cell
            y = -r + (j + bias_y) * cell
            sample_point = light.position + x * U + y * V
            sample_direction = sample_point - p_bias
            dist = np.linalg.norm(sample_direction)
            if dist == 0:
                unblocked += 1
                continue
            d = sample_direction / dist
            t_hit = FindIntersection(Ray(p_bias, d), primitives).t
            if t_hit <= eps or t_hit >= dist - eps:
                unblocked += 1

    weight = unblocked / float(total)
    return weight

def CreateRayReflection(ray, hit):
    # Reflect the incoming ray.direction about the hit normal
    dir_in = ray.direction
    R = dir_in - 2 * np.dot(dir_in, hit.normal) * hit.normal
    R = R / np.linalg.norm(R)
    return Ray(hit.point, R)


def diff_and_spec(hit, light , material, scene_settings, V): # Calc for specific light
    
    P = hit.point 
    N = hit.normal
    color = np.zeros(3)
    L = light.position - P
    L = L / np.linalg.norm(L)

    NdotL = max(0, np.dot(N, L))
    diffuse = NdotL * material.diffuse_color * light.color
    R = 2 * NdotL * N - L
    R = R / np.linalg.norm(R)
    RdotV = max(0, np.dot(R, V))
    specular = material.specular_color * light.color * light.specular_intensity * (RdotV ** material.shininess)
    color = diffuse + specular
        
    return color

def GetColor(ray, hit, lights, depth, scene_settings, primitives, materials):
    
    if ((depth >= scene_settings.max_recursions) or (hit.primitive is None)):
        return scene_settings.background_color
    
    mat = materials[hit.primitive.material_index - 1]
    final_color = np.zeros(3)

    shadow_samples = int(scene_settings.root_number_shadow_rays)

    for light in lights:
        percent_hits = CalculateSoftShadow(hit , shadow_samples , light , primitives, eps = 1e-6) #CHANGE Global eps
        light_intensity = (1-light.shadow_intensity) + light.shadow_intensity*(percent_hits)
        curr_diff_and_spec = diff_and_spec(hit, light, mat, scene_settings, -ray.direction)
        final_color += curr_diff_and_spec*light_intensity
    
    final_color = final_color*(1-mat.transparency)
    
    if mat.transparency >0:
        exclude_curr_primitives = [primitive for primitive in primitives if primitive != hit.primitive]
        transparency_hit = FindIntersection(ray, exclude_curr_primitives)
        color_behind = GetColor(ray, transparency_hit, lights, depth+1, scene_settings, exclude_curr_primitives, materials)
        final_color += color_behind*mat.transparency

    if np.any(mat.reflection_color):
        ref_ray = CreateRayReflection(ray, hit)
        ref_hit = FindIntersection(ref_ray, primitives)
        ref_color = GetColor(ref_ray, ref_hit, lights, depth+1, scene_settings, primitives, materials)
        final_color += ref_color*mat.reflection_color

    return final_color


def main():
    parser = argparse.ArgumentParser(description='Python Ray Tracer')
    parser.add_argument('scene_file', type=str, help='Path to the scene file')
    parser.add_argument('output_image', type=str, help='Name of the output image file')
    parser.add_argument('--width', type=int, default=500, help='Image width')
    parser.add_argument('--height', type=int, default=500, help='Image height')
    args = parser.parse_args()

    # Parse the scene file
    camera, scene_settings, objects = parse_scene_file(args.scene_file)

    # Convert list params to numpy arrays for vector math
    camera.position = np.array(camera.position)
    camera.look_at = np.array(camera.look_at)
    camera.up_vector = np.array(camera.up_vector)
    scene_settings.background_color = np.array(scene_settings.background_color)

    for obj in objects:
        if isinstance(obj, Light):
            obj.position = np.array(obj.position)
            obj.color = np.array(obj.color)
        elif isinstance(obj, Material):
            obj.diffuse_color = np.array(obj.diffuse_color)
            obj.specular_color = np.array(obj.specular_color)
            obj.reflection_color = np.array(obj.reflection_color)
        elif isinstance(obj, Sphere):
            obj.position = np.array(obj.position)
        elif isinstance(obj, Cube):
            obj.position = np.array(obj.position)
        elif isinstance(obj, InfinitePlane):
            obj.normal = np.array(obj.normal)

    image_width, image_height = args.width, args.height

    image_array = np.zeros((image_height, image_width, 3))

    pixel_size = camera.screen_width / image_width
    screen_height = pixel_size * image_height

    forward = camera.look_at - camera.position
    forward = forward / np.linalg.norm(forward)

    right = np.cross(camera.up_vector, forward)
    right = right / np.linalg.norm(right)

    up = np.cross(right, forward)
    up = up / np.linalg.norm(up)

    half_width = camera.screen_width * 0.5
    half_height = screen_height * 0.5
    plane_center = camera.position + forward * camera.screen_distance

    materials = filter_by_type(objects, ["Material"])
    lights = filter_by_type(objects, ["Light"])
    primitives = filter_by_type(objects, ["Cube", "Sphere", "InfinitePlane"])

    for i in range(image_width): # Width
        for j in range(image_height): # Height
            ray = ConstructRayThroughPixel(camera, i, j, half_width, half_height, plane_center , pixel_size , right, up);
            hit = FindIntersection(ray, primitives)
            image_array[j][i] = GetColor(ray, hit, lights, 0, scene_settings, primitives, materials)
        if i % 100 == 0:
            print(f"Progress: column {i}/{image_width}", flush=True)
    
    # Save the output image
    save_image(image_array)


if __name__ == '__main__':
    main()