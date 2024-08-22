from PIL import Image, ImageDraw, ImageFont, ImageFilter
import numpy as np
import os
import matplotlib.pyplot as plt

# Define the grayscale gradient (from least dense to most dense)
gscale = np.asarray(list("+*=coarb%@#%&8@#"))
gscale = np.asarray(list(' .`-_\':,;^=+/"|)\\<>)iv%xclrs{*}I?!][1taeo7zjLunT#JCwfy325Fp6mqSghVd4EgXPGZbYkOA&8U$@KHDBWNMR0Q'))

# Edge characters
edge_chars = {
    'h': '-',
    'v': '|',
    'r': '/',
    'l': '\\'
}

# Gaussian Filter
def gaussian_filter(image_array, sigma=1):
    size = int(6 * sigma + 1)
    if size % 2 == 0:
        size += 1
    kernel = np.fromfunction(
        lambda x, y: (1 / (2 * np.pi * sigma ** 2)) * np.exp(
            - ((x - (size - 1) / 2) ** 2 + (y - (size - 1) / 2) ** 2) / (2 * sigma ** 2)
        ),
        (size, size)
    )
    kernel /= np.sum(kernel)
    pad_size = size // 2
    padded_image = np.pad(image_array, pad_size, mode='constant', constant_values=0)
    smooth_img = np.zeros_like(image_array)

    for i in range(image_array.shape[0]):
        for j in range(image_array.shape[1]):
            smooth_img[i, j] = np.sum(kernel * padded_image[i:i + size, j:j + size])
    return smooth_img

# Difference of Gaussians
def difference_of_gaussians(image_array, sigma1=1, sigma2=2):
    dog1 = gaussian_filter(image_array, sigma1)
    dog2 = gaussian_filter(image_array, sigma2)
    return dog1 - dog2

# Sobel Filter
def sobel_filter(image_array):
    sobel_x = np.array([
        [1, 0, -1],
        [2, 0, -2],
        [1, 0, -1]
    ])
    sobel_y = np.array([
        [1, 2, 1],
        [0, 0, 0],
        [-1, -2, -1]
    ])

    pad_size = 1
    padded_image = np.pad(image_array, pad_size, mode='constant', constant_values=0)
    edges = np.zeros_like(image_array, dtype=float)
    gx = np.zeros_like(image_array, dtype=float)
    gy = np.zeros_like(image_array, dtype=float)

    for i in range(image_array.shape[0]):
        for j in range(image_array.shape[1]):
            region = padded_image[i:i + 3, j:j + 3]
            gx[i, j] = np.sum(sobel_x * region)
            gy[i, j] = np.sum(sobel_y * region)
            edges[i, j] = np.hypot(gx[i, j], gy[i, j])

    return edges, gx, gy

# Non-Maximum Suppression
def non_maximum_suppression(edges, gx, gy):
    directions = np.arctan2(gy, gx) * (180 / np.pi)
    directions[directions < 0] += 180
    nms = np.zeros_like(edges, dtype=np.uint8)

    for i in range(1, edges.shape[0] - 1):
        for j in range(1, edges.shape[1] - 1):
            angle = directions[i, j]
            q = 255
            r = 255

            if (0 <= angle < 22.5) or (157.5 <= angle <= 180):
                q = edges[i, j + 1]
                r = edges[i, j - 1]
            elif 22.5 <= angle < 67.5:
                q = edges[i + 1, j - 1]
                r = edges[i - 1, j + 1]
            elif 67.5 <= angle < 112.5:
                q = edges[i + 1, j]
                r = edges[i - 1, j]
            elif 112.5 <= angle < 157.5:
                q = edges[i - 1, j - 1]
                r = edges[i + 1, j + 1]

            if (edges[i, j] >= q) and (edges[i, j] >= r):
                nms[i, j] = edges[i, j]
            else:
                nms[i, j] = 0

    return nms

# Apply smoothing to the image
def smooth_image(image_array, sigma=1):
    return gaussian_filter(image_array, sigma)

# Map grayscale values to ASCII characters with smoothing
def map_to_ascii(image_array, gscale, sigma=1, smoothed=False):
    if smoothed:
        smoothed_image = smooth_image(image_array, sigma)
        ascii_array = (smoothed_image / 255 * (len(gscale) - 1)).astype(int)
    else:
        ascii_array = (image_array / 255 * (len(gscale) - 1)).astype(int)
    return gscale[ascii_array]

# Apply edges to ASCII art
def apply_edges_to_ascii(ascii_art, edges, directions, edge_chars):
    for y in range(ascii_art.shape[0]):
        for x in range(ascii_art.shape[1]):
            if edges[y, x] > 400:  # Apply threshold here
                print(1)
                angle = directions[y, x]
                if (0 <= angle < 5.5) or (157.5 <= angle <= 180):
                    ascii_art[y, x] = edge_chars['h']
                elif 67.5 <= angle < 112.5:
                    ascii_art[y, x] = edge_chars['v']
                elif 22.5 <= angle < 67.5:
                    ascii_art[y, x] = edge_chars['r']
                elif 112.5 <= angle < 157.5:
                    ascii_art[y, x] = edge_chars['l']
    return ascii_art

# Apply depth map adjustments
def apply_depth_map(ascii_art, depth_map_array, gscale, edge_chars):
    max_depth = np.max(depth_map_array)
    min_depth = np.min(depth_map_array)
    depth_range = max_depth - min_depth

    if depth_range > 0:
        normalized_depth = (depth_map_array - min_depth) / depth_range
    else:
        normalized_depth = np.zeros_like(depth_map_array)

    for y in range(ascii_art.shape[0]):
        for x in range(ascii_art.shape[1]):
            depth_value = normalized_depth[y, x]
            gscale_index = int(depth_value * (len(gscale) - 1))
            ascii_art[y, x] = gscale[gscale_index]

            if depth_value > 0.8:  # Example threshold for closer objects
                ascii_art[y, x] = edge_chars.get(ascii_art[y, x], ascii_art[y, x])

    return ascii_art

# Draw ASCII image
def draw_ascii_image(ascii_art, img_rgb_array, char_size):
    ascii_img_width = char_size * ascii_art.shape[1]
    ascii_img_height = char_size * ascii_art.shape[0]

    ascii_img = Image.new('RGB', (ascii_img_width, ascii_img_height), color=(0, 0, 0))
    draw = ImageDraw.Draw(ascii_img)

    font = ImageFont.load_default()

    for y in range(ascii_art.shape[0]):
        for x in range(ascii_art.shape[1]):
            char = ascii_art[y, x]
            color = tuple(img_rgb_array[y, x])
            draw.text((x * char_size, y * char_size), char, font=font, fill=color)

    return ascii_img

# Process images
def process_images(input_folder, output_folder, depthmap_folder):
    for img_file in os.listdir(input_folder):
        img_path = os.path.join(input_folder, img_file)
        img = Image.open(img_path)
        img_rgb = img
        img_gray = img.convert('L')
        width, height = img.size
        scale = 4
        nscale = (width // scale, height // scale)
        img_gray = img_gray.resize(nscale)
        img_rgb = img_rgb.resize(nscale)

        img_array = np.asarray(img_gray)

        sigma1_default = 0.1
        sigma2_default = 0.4

        dog_img = difference_of_gaussians(img_array, sigma1_default, sigma2_default)
        edges, gx, gy = sobel_filter(dog_img)
        nms_edges = non_maximum_suppression(edges, gx, gy)

        ascii_art = map_to_ascii(img_array, gscale, sigma=2, smoothed=False)  # Adjust sigma as needed

        ascii_art_with_edges = apply_edges_to_ascii(ascii_art, nms_edges, np.arctan2(gy, gx) * (180 / np.pi), edge_chars)

        depth_map_path = os.path.join(depthmap_folder, img_file)
        if os.path.exists(depth_map_path):
            depth_map = Image.open(depth_map_path).convert('L')
            depth_map = depth_map.resize(nscale)
            depth_map_array = np.asarray(depth_map)
            ascii_art_with_edges = apply_depth_map(ascii_art_with_edges, depth_map_array, gscale, edge_chars)

        img_rgb_array = np.asarray(img_rgb)

        char_size = 7

        ascii_img = draw_ascii_image(ascii_art_with_edges, img_rgb_array, char_size)

        output_path = os.path.join(output_folder, f'ascii_{os.path.basename(img_file)}')
        ascii_img.save(output_path)
        print(f'Saved ASCII art image: {output_path}')

# Specify input, depth map, and output folders
input_folder = r'C:\Users\marko\PycharmProjects\smalltestv2\inputs'
output_folder = r'C:\Users\marko\PycharmProjects\smalltestv2\results'
depthmap_folder = r'C:\Users\marko\PycharmProjects\smalltestv2\inputs_depth'

# Process the images
process_images(input_folder, output_folder, depthmap_folder)
