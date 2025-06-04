import argparse
import glob
import os
import shutil
import urllib.request

from osgeo import gdal
from tile_convert import bbox_to_xyz, tile_edges

temp_dir = os.path.join(os.path.dirname(__file__), 'temp')

def fetch_tile(x, y, z, tile_source):
    url = tile_source.replace(
        "{x}", str(x)).replace(
        "{y}", str(y)).replace(
        "{z}", str(z))

    if not tile_source.startswith("http"):
        return url.replace("file:///", "")

    path = f'{temp_dir}/{x}_{y}_{z}.png'
    req = urllib.request.Request(
        url,
        data=None,
        headers={
            'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) tiles-to-tiff/1.0 (+https://github.com/jimutt/tiles-to-tiff)'
        }
    )
    g = urllib.request.urlopen(req)
    with open(path, 'b+w') as f:
        f.write(g.read())
    return path


def merge_tiles(input_pattern, output_path):
    vrt_path = temp_dir + "/tiles.vrt"
    gdal.BuildVRT(vrt_path, glob.glob(input_pattern))
    gdal.Translate(output_path, vrt_path)


def georeference_raster_tile(x, y, z, temp_dir, tile_path, tms):
    bounds = tile_edges(x, y, z, tms)
    filename = os.path.basename(tile_path)
    tilename = os.path.splitext(filename)[0]
    print(f"Georeferencing {filename}")
    gdal.Translate(os.path.join(temp_dir, f'{tilename}.tif'),
                   tile_path,
                   outputSRS='EPSG:4326',
                   outputBounds=bounds,
                   rgbExpand='rgb')
    return os.path.join(temp_dir, f'{tilename}.tif')

def convert(tile_source, output_dir, bounding_box, zoom):
    lon_min, lat_min, lon_max, lat_max = bounding_box

    # Script start:

    # extact {y} from tile_source, if -y then tms is True
    parts = tile_source.split("/")
    tms = parts[-1].split(".")[0].strip('{}')

    if tms.startswith("-"):
        #print("The string starts with '-'")
        tile_source = tile_source.replace("-", "")
        tms = True
        #print(f"tile_source is {tile_source}")
    else:
        #print("The string does not start with '-'")
        tms = False

    #print(f"tms is {tms}")

    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    x_min, x_max, y_min, y_max = bbox_to_xyz(
        lon_min, lon_max, lat_min, lat_max, zoom, tms)

    print(f"Fetching & georeferencing {(x_max - x_min + 1) * (y_max - y_min + 1)} tiles")

    for x in range(x_min, x_max + 1):
        for y in range(y_min, y_max + 1):
            try:
                png_path = fetch_tile(x, y, zoom, tile_source)
                print(f"{x},{y} fetched")
                georeference_raster_tile(x, y, zoom, png_path, tms)
            except OSError:
                print(f"Error, failed to get {x},{y}")
                pass

    print("Resolving and georeferencing of raster tiles complete")

    print("Merging tiles")
    merge_tiles(temp_dir + '/*.tif', output_dir + '/merged.tif')
    print("Merge complete")

    shutil.rmtree(temp_dir)

def convert_tile(tile_source, output_dir, bounding_tilename_box, zoom):
    x_min, x_max, y_min, y_max = bounding_tilename_box

    # Script start:

    # extact {y} from tile_source, if -y then tms is True
    parts = tile_source.split("/")
    tms = parts[-1].split(".")[0].strip('{}')

    if tms.startswith("-"):
        #print("The string starts with '-'")
        tile_source = tile_source.replace("-", "")
        tms = True
        #print(f"tile_source is {tile_source}")
    else:
        #print("The string does not start with '-'")
        tms = False

    #print(f"tms is {tms}")

    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)


    print(f"Fetching & georeferencing {(x_max - x_min + 1) * (y_max - y_min + 1)} tiles")

    for x in range(x_min, x_max + 1):
        for y in range(y_min, y_max + 1):
            try:
                png_path = fetch_tile(x, y, zoom, tile_source)
                print(f"{x},{y} fetched")
                georeference_raster_tile(x, y, zoom, png_path, tms)
            except OSError:
                print(f"Error, failed to get {x},{y}")
                pass

    print("Resolving and georeferencing of raster tiles complete")

    print("Merging tiles")
    merge_tiles(temp_dir + '/*.tif', output_dir + '/merged.tif')
    print("Merge complete")

    shutil.rmtree(temp_dir)

def convert_tile_local(tile_source, output_dir, bounding_tilename_box, zoom, use_tms):
    x_min, x_max, y_min, y_max = bounding_tilename_box
    tms = use_tms

    # Script start:

    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)


    print(f"Fetching & georeferencing {(x_max - x_min + 1) * (y_max - y_min + 1)} tiles")

    for x in range(x_min, x_max + 1):
        for y in range(y_min, y_max + 1):
            png_path = os.path.join(tile_source, f'{zoom}_{x}_{y}.png')
            try:
                print(f"{png_path} fetched : {os.path.exists(png_path)}")
                georeference_raster_tile(x, y, zoom, temp_dir, png_path, tms)
            except OSError:
                print(f"Error, failed to get {png_path}")
                pass

    print("Resolving and georeferencing of raster tiles complete")

    print("Merging tiles")
    merge_tiles(temp_dir + '/*.tif', output_dir + f'/{zoom}_merged.tif')
    print("Merge complete")

    shutil.rmtree(temp_dir)

if __name__ == "__main__":
    # parser = argparse.ArgumentParser("tiles_to_tiff", "python tiles_to_tiff https://tileserver-url.com/{z}/{x}/{y}.png 21.49147 65.31016 21.5 65.31688 -o output -z 17")
    # parser.add_argument("tile_source", type=str, help="Local directory pattern or URL pattern to a slippy maps tile source. Ability to use {-y} in the URL to specify a TMS service", )
    # parser.add_argument("lng_min", type=float, help="Min longitude of bounding box")
    # parser.add_argument("lat_min", type=float, help="Min latitude of bounding box")
    # parser.add_argument("lng_max", type=float, help="Max longitude of bounding box")
    # parser.add_argument("lat_max", type=float, help="Max latitude of bounding box")
    # parser.add_argument("-z", "--zoom", type=int, help="Tilesource zoom level", default=14)
    # parser.add_argument("-o", "--output", type=str, help="Output directory", required=True)

    # args = parser.parse_args()

    # tile_source = args.tile_source if args.tile_source.startswith("http") else "file:///" + args.tile_source

    # convert(tile_source, args.output, [args.lng_min, args.lat_min, args.lng_max, args.lat_max], args.zoom)

    from Tile import find_min_max_sub_tiles

    FILENAME = "14_8189_5447.png"
    TARGET_ZOOM = 18

    DATA_DIR = r"D:\Research\GEN\Experiment\tile\14_8189_5447"
    TILE_DIR = os.path.join(DATA_DIR, "samples")
    OUTPUT_DIR = os.path.join(DATA_DIR, "output")

    parts = FILENAME.split('_')
    z = int(parts[0])

    # print(f"Converting tile {FILENAME} to TIFF")
    # tiles_box = find_min_max_sub_tiles(FILENAME, TARGET_ZOOM)
    # print(f"Tiles box: {tiles_box}")

    for level in range(z+1, TARGET_ZOOM):
        tiles_box = find_min_max_sub_tiles(FILENAME, level)
        print(f"Converting tiles at level {level} to TIFF, box: {tiles_box}")
        convert_tile_local(TILE_DIR, OUTPUT_DIR, tiles_box, level, False)