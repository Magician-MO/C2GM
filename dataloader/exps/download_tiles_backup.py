import argparse
import math
import os
import random
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import requests
from PIL import Image


def parse_args():
    # Initialize the argument parser
    parser = argparse.ArgumentParser(description='Command line arguments.')

    # Add arguments
    parser.add_argument('--pfile',
                        type=str,
                        default='central-belt150000',
                        help='Points file, the id of the file. Example: central-belt50.')

    parser.add_argument('--levels',
                        type=int,
                        nargs='+',
                        default=[14, 15, 16, 17],
                        help='Zoom level. single or multiple zoom levels. Example: 18 or 18 19 20.')
    parser.add_argument('--threads',
                        type=int,
                        default=os.cpu_count(),
                        help='Number of threads to use.')
    parser.add_argument('--apis',
                        type=str,
                        nargs='+',
                        default=['google-map-nolabel'],
                        help='APIs to use. Example: maptile-map, worldimagery-clarity, openstreetmap.')
    parser.add_argument('--root',
                        type=str,
                        default=os.getcwd(),
                        help='Root folder.')
    parser.add_argument('--save_root',
                        type=str,
                        default='dataset',
                        help='Root folder to save the tiles.')

    # Parse the arguments
    args = parser.parse_args()

    # HACK: Manually set the arguments
    args.coords_path = f'{args.root}/{args.pfile}'
    args.tiles_path = f'{args.save_root}/tiles_{args.pfile}'

    return args


# API urls
maptiler_key = "54hn6djz8nl68nANbOc7",
APIS = {
    'worldimagery': 'https://services.arcgisonline.com/arcgis/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
    'worldimagery-clarity': 'https://clarity.maptiles.arcgis.com/arcgis/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
    # 'openstreetmap' : 'https://tile.openstreetmap.org/{z}/{x}/{y}.png',
    # 'openstreetmap': 'https://wprd01.is.autonavi.com/appmaptile?lang=zh_cn&size=1&style=7&x={x}&y={y}&z={z}&scl=1&ltype=3',
    'openstreetmap': 'https://tile-a.openstreetmap.fr/hot/{z}/{x}/{y}.png',
    # 'openstreetmap' : 'https://tile.thunderforest.com/atlas/{z}/{x}/{y}.png?apikey=86628529556241f09b9e0b940c34046a',
    # 'openstreetmap' : 'https://t0.tianditu.gov.cn/vec_w/wmts?SERVICE=WMTS&REQUEST=GetTile&VERSION=1.0.0&LAYER=vec&STYLE=default&TILEMATRIXSET=w&FORMAT=tiles&TILEMATRIX={z}&TILEROW={y}&TILECOL={x}&tk=93f2bd0fa790aeb6b5b8cf50f938af43',
    'ukosgb1888': 'https://api.maptiler.com/tiles/uk-osgb10k1888/{z}/{x}/{y}.jpg?key=MXVhdLdJmHeZ0z5DwjBI',

    'google-map': 'http://mts0.googleapis.com/vt?lyrs=m&x={x}&y={y}&z={z}',
    'google-sat': 'http://mts0.googleapis.com/vt?lyrs=s&x={x}&y={y}&z={z}',
    'google-map-nolabel': 'https://maps.googleapis.com/maps/vt?pb=!1m5!1m4!1i{z}!2i{x}!3i{y}!4i256!2m3!1e0!2sm!3i653393117!3m17!2szh-CN!3sUS!5e18!12m4!1e68!2m2!1sset!2sRoadmap!12m3!1e37!2m1!1ssmartmaps!12m4!1e26!2m2!1sstyles!2zcy5lOmx8cC52Om9mZixzLnQ6ODE4fHAudjpvbg!4e0!5m1!5f2!23i1379903!23i1376099&key=AIzaSyCeCzXVJo5AS-j4-K_tL-FPkMwjruQkzP4&token=69144',

    'maptile-map': 'https://api.maptiler.com/maps/fff24394-dd93-48cb-901d-024c82c510cd/256/{z}/{x}/{y}.png?key={}'.format(maptiler_key),
    'maptile-sat': 'https://api.maptiler.com/tiles/satellite-v2/{z}/{x}/{y}.jpg?key=k1SfrIBkAFG29OagoRUo'
}

def deg2num(lat_deg, lon_deg, zoom):
    lat_rad = math.radians(lat_deg)
    n = 2.0 ** zoom
    xtile = int((lon_deg + 180.0) / 360.0 * n)
    ytile = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
    return xtile, ytile


def format_string(url, x, y, zoom):
    substituted_string = url.replace('{x}', str(x))
    substituted_string = substituted_string.replace('{y}', str(y))
    substituted_string = substituted_string.replace('{z}', str(zoom))
    return substituted_string


def download_tile(url, tile_path, if_check, max_retries=5, retry_delay=2):
    retries = 0
    while retries < max_retries:
        try:
            # subprocess.run(["curl", url, '--output', tile_path], check=True)
            response = requests.get(url)
            if response.status_code == 200:
                with open(tile_path, "wb")as f:
                    f.write(response.content)
                    print("downloaded one tile !!")
                if if_check:
                    if_save = check(tile_path)
                    if(not if_save):
                        os.remove(tile_path)
                        print("deleted one tile !!")
            else:
                print("*** downloading failed !! ***")
            return  # Download successful, exit the function
        except subprocess.CalledProcessError as e:
            print(f"Download failed: {e}")
            retries += 1
            if retries < max_retries:
                print(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                print("Max retries reached, giving up.")

# check whether to save one tile (to prevent from woo much useless region)
def check(img_path):
    img = np.asarray(Image.open(img_path).convert("L"))
    if(np.var(img) > 30):
        return True
    elif(random.random() > 0.95):
        return True
    else:
        return False

def cleanUp(dir_path1, dir_path2):
    target_set = set(os.listdir(dir_path1))
    for img_name in os.listdir(dir_path2):
        if(img_name in target_set):
            continue
        else:
            os.remove(os.path.join(dir_path2, img_name))


def main():

    args = parse_args()

    if not os.path.exists(args.tiles_path):
        os.makedirs(args.tiles_path)

    # Create folders for the different image types
    for url in args.apis:
        if not os.path.exists(f'{args.tiles_path}/{url}'):
            os.makedirs(f'{args.tiles_path}/{url}')

    # Load the coordinates
    coordinates = np.load(f'{args.coords_path}/{args.pfile}.npy')

    with ThreadPoolExecutor(max_workers=args.threads) as executor:
        futures = []

        for z in args.levels:
            for coord in coordinates:
                lon, lat = coord[0], coord[1]
                x, y = deg2num(lat, lon, z)

                for url in args.apis:
                    tile_url = format_string(APIS[url], x, y, z)
                    tile_path = f'{args.tiles_path}/{url}/{z}_{x}_{y}.png'
                    if_check = (url == "maptile-map")
                    future = executor.submit(
                        download_tile, tile_url, tile_path, if_check)
                    futures.append(future)

        # Wait for all the downloads to complete
        for future in futures:
            future.result()

    tile_paths = []
    for url in args.apis:
        tile_paths.append(os.path.join(args.tiles_path, url))
    cleanUp(tile_paths[0], tile_paths[1])
    cleanUp(tile_paths[1], tile_paths[0])


if __name__ == '__main__':
    main()
