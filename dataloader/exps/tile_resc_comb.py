import os
import re
import shutil
import sys

sys.path.append('/home/scx/project/GEN/Generate/SGDM/dataloader/utils')

from PIL import Image
from Tile import find_min_max, find_min_max_sub_tiles, get_filename_levels_group, get_filename_sub_target_tiles, get_filename_sub_tiles_group
from tile_convert import tile_edges
from tiles_to_tiff import convert_tile_local
from tqdm import tqdm


def parse_filename(filename):
    match = re.match(r'(\d+)_(\d+)_(\d+).png', filename)
    if match:
        z, x, y = map(int, match.groups())
        return z, x, y
    else:
        raise ValueError(f"Invalid filename: {filename}")

def stitch_tiles(filenames, folder_path, output_filename):
    tiles = []
    for filepath in tqdm(filenames):
        filename = os.path.basename(filepath)
        z, x, y = parse_filename(filename)
        # print(f"Reading tile {filename} at z={z}, x={x}, y={y}")
        tile = Image.open(os.path.join(folder_path, filename))
        tiles.append((z, x, y, tile))

    # Sort the tiles by z, x, y
    tiles.sort()

    # Assume all tiles have the same size
    tile0 = tiles[0][3]
    tile_width, tile_height = tile0.size
    # print(f"Tile size: {tile_width}x{tile_height}")

    # Get the min and max x and y values
    min_x = min(x for z, x, y, tile in tiles)
    max_x = max(x for z, x, y, tile in tiles)
    min_y = min(y for z, x, y, tile in tiles)
    max_y = max(y for z, x, y, tile in tiles)

    # Create the output image
    output = Image.new('RGB',
                       (tile_width * (max_x - min_x + 1),
                        tile_height * (max_y - min_y + 1)))
    # print(f"Output size: {output.size}")

    # Paste each tile into the correct position in the output image
    for z, x, y, tile in tiles:
        abs_x = x - min_x
        abs_y = y - min_y
        # print(f"Pasting tile at z={z}, x={abs_x}, y={abs_y}")
        output.paste(tile, (abs_x * tile_width, abs_y * tile_height))

    output.save(output_filename)


def get_coord_box(tiles_box, target_zoom):
    x_min, x_max, y_min, y_max = tiles_box
    coord_box_min = tile_edges(x_min, y_min, target_zoom, False)
    coord_box_max = tile_edges(x_max, y_max, target_zoom, False)

    lon_min = min(coord_box_min[0], coord_box_max[0])
    lat_min = min(coord_box_min[1], coord_box_max[1])
    lon_max = max(coord_box_min[2], coord_box_max[2])
    lat_max = max(coord_box_min[3], coord_box_max[3])

    return [lon_min, lat_min, lon_max, lat_max]


def combine_cell_tiff(FILENAME, TARGET_ZOOM, TILE_DIR, OUPUT_DIR, JUMP_ZOOM=None, RES_DIR=None):
    filenames_group = get_filename_sub_tiles_group(FILENAME, TARGET_ZOOM)
    for z, filenames in filenames_group.items():
        if JUMP_ZOOM and z == JUMP_ZOOM:
            continue
        print(f"Converting tiles at level {z} to TIFF")
        output_filename = os.path.join(OUPUT_DIR, f'{z}_merge_{len(filenames)}.png')
        stitch_tiles(filenames, TILE_DIR, output_filename)

        if RES_DIR:
            for filename in filenames:
                filename = os.path.basename(filename)
                shutil.copy(os.path.join(TILE_DIR, filename), os.path.join(RES_DIR, filename))

def combine_all_tiff(TILE_DIR, OUPUT_DIR, JUMP_ZOOM=None):
    print("Combining all TIFF files")
    filenames_dir = os.listdir(TILE_DIR)
    filenames_group = get_filename_levels_group(filenames_dir)

    for z, filenames in filenames_group.items():
        if JUMP_ZOOM and z == JUMP_ZOOM:
            continue
        print(f"Converting tiles at level {z} to TIFF")
        output_filename = os.path.join(OUPUT_DIR, f'{z}_merge_{len(filenames)}.png')
        stitch_tiles(filenames, TILE_DIR, output_filename)

if __name__ == '__main__':
    # TILENAME = "14_8189_5447"
    # TILENAME = "15_16378_10894"
    # TILENAME = "15_9374_12537"
    # FILENAME = TILENAME + ".png"
    # TARGET_ZOOM = 18
    # JUMP_ZOOM = 15

    # DATA_DIR = r"data/test/refmap-level-test-range-oc-oz"
    # DATA_DIR = r"data/test/refmap-level-test-mlme-un-4c-oz"
    # SPLIT_NAME = "samples"

    # TILE_DIR = os.path.join(DATA_DIR, SPLIT_NAME)
    # OUPUT_DIR = os.path.join(DATA_DIR, "all", SPLIT_NAME)
    # OUPUT_DIR = os.path.join(DATA_DIR, TILENAME, "output", SPLIT_NAME)
    # RES_DIR = os.path.join(DATA_DIR, TILENAME, "recursion", SPLIT_NAME)

    # if not os.path.exists(OUPUT_DIR):
    #     os.makedirs(OUPUT_DIR)

    # if not os.path.exists(RES_DIR):
        # os.makedirs(RES_DIR)

    # parts = FILENAME.split('_')
    # z = int(parts[0])

    # print(f"Converting tile {FILENAME} to TIFF")
    # tiles_name = get_filename_sub_target_tiles(FILENAME, TARGET_ZOOM)
    # print(f"Tiles name: {tiles_name}")
    # tiles_box = find_min_max_sub_tiles(FILENAME, TARGET_ZOOM)
    # print(f"Tiles box: {tiles_box}")
    # coord_box = get_coord_box(tiles_box, TARGET_ZOOM)
    # print(f"Coord box min: {coord_box}")

    # output_filename = os.path.join(OUPUT_DIR, f'{TARGET_ZOOM}_merge_{len(tiles_name)}.png')
    # stitch_tiles(tiles_name, TILE_DIR, output_filename)

    DATA_DIR = r"data/test/refmap-level-test-range-oc-oz"
    SPLIT_NAME = "samples"
    TILE_DIR = os.path.join(DATA_DIR, SPLIT_NAME)
    OUPUT_DIR = os.path.join(DATA_DIR, "outputs", "all", SPLIT_NAME)

    if not os.path.exists(OUPUT_DIR):
        os.makedirs(OUPUT_DIR)

    combine_all_tiff(TILE_DIR, OUPUT_DIR)
