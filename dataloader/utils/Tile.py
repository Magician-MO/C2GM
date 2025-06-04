import math
from collections import defaultdict

import numpy as np
from PIL import Image


def deg2num(lat_deg, lon_deg, zoom):
    lat_rad = math.radians(lat_deg)
    n = 1 << zoom
    xtile = int((lon_deg + 180.0) / 360.0 * n)
    ytile = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
    return xtile, ytile

def num2deg(xtile, ytile, zoom):
    # This returns the NW-corner of the square. Use the function with xtile+1 and/or ytile+1 to get the other corners.
    # With xtile+0.5 & ytile+0.5 it will return the center of the tile.
    n = 1 << zoom
    lon_deg = xtile / n * 360.0 - 180.0
    lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * ytile / n)))
    lat_deg = math.degrees(lat_rad)
    return lat_deg, lon_deg

def parse_filename(filename):
    parts = filename.split('_')
    z = int(parts[0])
    x = int(parts[1])
    y = int(parts[2].split('.')[0])  # 注意这里我们需要去除文件扩展名
    return z, x, y

def get_sub_tiles(xtile, ytile, zoom):
    return [(xtile * 2, ytile * 2, zoom + 1),
            (xtile * 2 + 1, ytile * 2, zoom + 1),
            (xtile * 2, ytile * 2 + 1, zoom + 1),
            (xtile * 2 + 1, ytile * 2 + 1, zoom + 1)]

def get_parent_tile(xtile, ytile, zoom):
    if xtile % 2 == 0 and ytile % 2 == 0:
        position = 'top_left'
    elif xtile % 2 == 1 and ytile % 2 == 0:
        position = 'top_right'
    elif xtile % 2 == 0 and ytile % 2 == 1:
        position = 'bottom_left'
    elif xtile % 2 == 1 and ytile % 2 == 1:
        position = 'bottom_right'
    return (xtile // 2, ytile // 2, zoom - 1, position)

def get_crop_coordinates(x_start, y_start, img_size, position):
    # 根据 position 计算裁剪区域的左上角坐标
    if position == 'top_left':
        x, y = x_start, y_start
    elif position == 'top_right':
        x, y = x_start + img_size // 2, y_start
    elif position == 'bottom_left':
        x, y = x_start, y_start + img_size // 2
    elif position == 'bottom_right':
        x, y = x_start + img_size // 2, y_start + img_size // 2

    return x, y, img_size // 2

def get_filename_levels_group(filenames):
    # 创建一个默认为列表的字典
    tiles_by_zoom = defaultdict(list)

    # 遍历每个文件名
    for filename in filenames:
        # 解析 z 值
        z = parse_filename(filename)[0]

        # 将文件名添加到对应的列表中
        tiles_by_zoom[z].append(filename)

    return tiles_by_zoom


def get_filename_sub_tile(filename):
    z, x, y = parse_filename(filename)

    # 获取下级瓦片
    subtiles = get_sub_tiles(x, y, z)

    # 将下级瓦片转换为文件名格式
    subtile_filenames = [f"{z}_{x}_{y}.png" for (x, y, z) in subtiles]

    return subtile_filenames

def get_filename_sub_tiles(filename, target_zoom=18):
    # 解析文件名
    parts = filename.split('_')
    z = int(parts[0])

    # 创建一个列表来保存结果，并将当前瓦片添加到结果中
    result = [filename]

    # 如果当前缩放级别还没有达到目标缩放级别，获取下级瓦片并递归调用此函数
    if z < target_zoom:
        subtile_filenames = get_filename_sub_tile(filename)
        for subtile_filename in subtile_filenames:
            result.extend(get_filename_sub_tiles(subtile_filename, target_zoom))

    return result

def get_filename_sub_tiles_group(filename, target_zoom=18):
    z_start, _, _ = parse_filename(filename)
    filenames = get_filename_sub_tiles(filename, target_zoom)

    tiles_by_zoom = defaultdict(list)
    tiles_by_zoom = get_filename_levels_group(filenames)
    tiles_by_zoom.pop(z_start)

    return tiles_by_zoom

def get_filename_sub_target_tiles(filename, target_zoom=18):
    # 解析文件名
    parts = filename.split('_')
    z = int(parts[0])

    # 如果当前缩放级别已经达到目标缩放级别，直接返回当前瓦片
    if z == target_zoom:
        return [filename]

    # 获取下级瓦片
    subtile_filenames = get_filename_sub_tile(filename)

    # 对每个下级瓦片递归调用此函数，直到达到目标缩放级别
    result = []
    for subtile_filename in subtile_filenames:
        result.extend(get_filename_sub_target_tiles(subtile_filename, target_zoom))

    return result

def get_filename_parent_tiles_recursion(filename, target_zoom=0):
    # 解析文件名以获取当前瓦片的编号和缩放级别
    parts = filename.split('_')
    xtile = int(parts[1])
    ytile = int(parts[2].split('.')[0])
    zoom = int(parts[0])

    # 如果缩放级别大于目标最小缩放级别，递归地获取父瓦片
    if zoom > target_zoom + 1:
        parent_x, parent_y, parent_z, position = get_parent_tile(xtile, ytile, zoom)
        parent_filename = '{}_{}_{}.png'.format(parent_z, parent_x, parent_y)
        parent_filenames, positions = get_filename_parent_tiles_recursion(parent_filename, target_zoom)
        return [parent_filename] + parent_filenames, [position] + positions
    else:
        parent_x, parent_y, parent_z, position = get_parent_tile(xtile, ytile, zoom)
        parent_filename = '{}_{}_{}.png'.format(parent_z, parent_x, parent_y)
        return [parent_filename], [position]

def get_filename_parent_tile(filename):
    # 解析文件名以获取当前瓦片的编号和缩放级别
    parts = filename.split('_')
    xtile = int(parts[1])
    ytile = int(parts[2].split('.')[0])
    zoom = int(parts[0])

    parent_x, parent_y, parent_z, position = get_parent_tile(xtile, ytile, zoom)
    parent_filename = '{}_{}_{}.png'.format(parent_z, parent_x, parent_y)
    return parent_filename, position

def get_filename_parent_tiles(filename, target_zoom=0):
    parts = filename.split('_')
    zoom = int(parts[0])

    parent_filenames = []
    positions = []
    for i in range(zoom - target_zoom):
        parent_filename, position = get_filename_parent_tile(filename)
        parent_filenames.append(parent_filename)
        positions.append(position)
        filename = parent_filename
    parent_list = list(zip(parent_filenames, positions))
    return parent_list

def get_filename_parent_tiles_scale(filename, zoom_scale=1):
    # 缩放尺度 SCALE = 2^zoom_scale
    parts = filename.split('_')
    zoom = int(parts[0])
    target_zoom = zoom - zoom_scale
    return get_filename_parent_tiles(filename, target_zoom)

def get_filename_parent_tile_scale(filename, img_size=256, zoom_scale=1):
    parent_list = get_filename_parent_tiles_scale(filename, zoom_scale)
    parent_filename = parent_list[-1][0]
    positions = [position for _, position in parent_list]
    positions.reverse()

    x, y, size = 0, 0, img_size
    # 计算所有的裁剪坐标
    for position in positions:
        x, y, size = get_crop_coordinates(x, y, size, position)

    return parent_filename, x, y, size

def get_image_parent_tile_scale(parent_path, coord_x, coord_y, img_size):
    parent_img = Image.open(parent_path).convert("RGB")
    cropped_img = parent_img.crop((coord_x, coord_y, coord_x + img_size, coord_y + img_size))
    resized_img = cropped_img.resize(parent_img.size, Image.BILINEAR)
    resized_img = np.asarray(resized_img) / 255.0
    resized_img = resized_img.astype(np.float32)
    return resized_img

def find_min_max(filenames, z_value):
    x_values = []
    y_values = []
    for filename in filenames:
        parts = filename.split('_')
        z = int(parts[0])
        x = int(parts[1])
        y = int(parts[2].split('.')[0])  # 注意这里我们需要去除文件扩展名
        if z == z_value:
            x_values.append(x)
            y_values.append(y)
    return min(x_values), max(x_values), min(y_values), max(y_values)

def find_min_max_sub_tiles(filename, target_zoom):
    parent_list = get_filename_sub_tiles(filename, target_zoom)
    return find_min_max(parent_list, target_zoom)


if __name__ == "__main__":
    # 使用示例
    # filename = "14_8189_5447.png"
    filename = "15_9374_12537.png"
    target_zoom = 17

    # print(get_filename_sub_tile(filename))
    # print(get_filename_parent_tile(filename))
    # print(get_filename_parent_tiles(filename, target_zoom=14))
    # print(get_filename_parent_tile_scale(filename, zoom_scale=1))
    # print(get_filename_parent_tiles_scale(filename, zoom_scale=3))

    # print(get_filename_sub_tiles(filename, target_zoom=18))
    print(get_filename_sub_target_tiles(filename, target_zoom))
    print(find_min_max_sub_tiles(filename, target_zoom))