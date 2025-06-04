import math
import os

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

from dataloader.utils import Tile, Util


class TMRefDataset(Dataset):
    def __init__(
        self,
        dataroot,
        scale=2,
        size=256,
        split="train",
        data_len=-1,
        use_txt=True,
        txt_level=True,
        txt_feature=False,
        style_path=None
    ):
        self.scale = scale
        self.size = size
        self.data_len = data_len
        self.split = split
        self.use_txt = use_txt
        self.txt_level = txt_level
        self.txt_feature = txt_feature
        self.style_path = style_path

        self.rs_path = Util.get_paths_from_images(
            "{}/rs_{}".format(dataroot, size)
        )
        self.map_path = Util.get_paths_from_images(
            "{}/map_{}".format(dataroot, size)
        )
        self.ref_scale_path = Util.get_paths_from_images(
            "{}/ref_scale_{}_{}".format(dataroot, scale, size)
        )

        if self.style_path:
            # Style image directory
            if os.path.isdir(self.style_path):
                self.style_path = Util.get_paths_from_images(self.style_path)
            # Single image for style guidance
            else:
                self.style_path = [self.style_path]

        self.dataset_len = len(self.map_path)
        if self.data_len <= 0:
            self.data_len = self.dataset_len
        else:
            self.data_len = min(self.data_len, self.dataset_len)

    def __len__(self):
        return self.data_len

    def __getitem__(self, index):
        img_map = None
        img_rs = None
        img_ref_scale = None
        img_style = None

        img_map = (
            np.asarray(Image.open(self.map_path[index]).convert("RGB")) / 255.0
        ).astype(np.float32)
        img_rs = (
            np.asarray(Image.open(self.rs_path[index]).convert("RGB")) / 255.0
        ).astype(np.float32)
        img_ref_scale = (
            np.asarray(Image.open(self.ref_scale_path[index]).convert("RGB")) / 255.0
        ).astype(np.float32)

        map_ditails = {
            0: {"resolution": 156543, "scale": "1:500000000", "example": "whole world"},
            1: {"resolution": 78272, "scale": "1:250000000", "example": "whole world"},
            2: {"resolution": 39136, "scale": "1:150000000", "example": "subcontinental area"},
            3: {"resolution": 19568, "scale": "1:70000000", "example": "largest country"},
            4: {"resolution": 9784, "scale": "1:35000000", "example": "whole world"},
            5: {"resolution": 4892, "scale": "1:15000000", "example": "large African country"},
            6: {"resolution": 2446, "scale": "1:10000000", "example": "large European country"},
            7: {"resolution": 1223, "scale": "1:4000000", "example": "small country, US state"},
            8: {"resolution": 611.496, "scale": "1:2000000", "example": "small country, US state"},
            9: {"resolution": 305.748, "scale": "1:1000000", "example": "wide area, large metropolitan area"},
            10: {"resolution": 152.874, "scale": "1:500000", "example": "metropolitan area"},
            11: {"resolution": 76.437, "scale": "1:250000", "example": "city"},
            12: {"resolution": 38.219, "scale": "1:150000", "example": "town, or city district"},
            13: {"resolution": 19.109, "scale": "1:70000", "example": "village, or suburb"},
            14: {"resolution": 9.555, "scale": "1:35000", "example": "village, or suburb"},
            15: {"resolution": 4.777, "scale": "1:15000", "example": "small road"},
            16: {"resolution": 2.389, "scale": "1:8000", "example": "street"},
            17: {"resolution": 1.194, "scale": "1:4000", "example": "block, park, addresses"},
            18: {"resolution": 0.597, "scale": "1:2000", "example": "some buildings, trees"},
            19: {"resolution": 0.299, "scale": "1:1000", "example": "local highway and crossing details"},
            20: {"resolution": 0.149, "scale": "1:500", "example": "mid-sized building"}
        }

        tile_level = 16
        level = self.rs_path[index].split("/")[-1].split("_")[1]
        if isinstance(level, (int, float)) and 0 <= level <= 20:
            tile_level = level
        map_resolution = map_ditails[tile_level]["resolution"]
        map_scale = map_ditails[tile_level]["scale"]
        map_example = map_ditails[tile_level]["example"]

        if self.style_path:
            if len(self.style_path) > 1:
                img_style = (
                    np.asarray(Image.open(self.style_path[index]).convert("RGB")) / 255.0
                ).astype(np.float32)
            else:
                img_style = (
                    np.asarray(Image.open(self.style_path[0]).convert("RGB")) / 255.0
                ).astype(np.float32)

        if self.use_txt:
            txt = "geographical map tile"

            if self.txt_level:
                txt += f", tile zoom level {tile_level}"
                txt += f", map resolution {map_resolution}m/pixel"
                txt += f", map scale {map_scale}"
                txt += f", examples of areas to represent {map_example}"

            if self.txt_feature:
                txt_book = np.asarray(Image.open(self.ref_scale_path[index]).convert("RGB"))
                building_vec = np.array([226, 224, 210])
                road_vec1 = np.array([254, 254, 254])
                road_vec2 = np.array([255, 204, 133])
                vegetation_vec1 = np.array([217, 233, 198])
                vegetation_vec2 = np.array([228, 236, 213])
                vegetation_vec3 = np.array([208, 230, 190])
                water_vec = np.array([133, 203, 249])
                building_exist = np.all((txt_book == building_vec), axis=-1).any()
                road_exist = (
                    np.all((txt_book == road_vec1), axis=-1).any()
                    or np.all((txt_book == road_vec2), axis=-1).any()
                )
                vegetation_exist = (
                    np.all((txt_book == vegetation_vec1), axis=-1).any()
                    or np.all((txt_book == vegetation_vec2), axis=-1).any()
                    or np.all((txt_book == vegetation_vec3), axis=-1).any()
                )
                water_exist = np.all((txt_book == water_vec), axis=-1).any()

                txt += "; has features include land,"
                if building_exist:
                    txt += "building of regular shape, "
                if road_exist:
                    txt += "road, "
                if vegetation_exist:
                    txt += "vegetation, "
                if water_exist:
                    txt += "water area, "
        else:
            txt = ""

        if self.style_path:
            [img_style, img_ref_scale, img_rs, img_map] = Util.transform_augment(
                [img_style, img_ref_scale, img_rs, img_map], split=self.split
            )
            img_map = img_map * 2 - 1
            return {
                "MAP": img_map,
                "RS": img_rs,
                "REF": img_ref_scale,
                "STYLE": img_style,
                "zoom": tile_level,
                "txt": txt,
                "path": self.map_path[index],
            }

        else:
            [img_ref_scale, img_rs, img_map] = Util.transform_augment(
                [img_ref_scale, img_rs, img_map], split=self.split
            )
            img_map = img_map * 2 - 1
            return {
                "MAP": img_map,
                "RS": img_rs,
                "REF": img_ref_scale,
                "zoom": tile_level,
                "txt": txt,
                "path": self.map_path[index],
            }

class TMRefCascadeDataset(Dataset):
    def __init__(
        self,
        dataroot,
        datalist="tilelist_18_15.csv",
        split="train",
        size=256,
        scale=2,
        data_len=-1,
        use_cascade=True,
        use_txt=True,
        txt_level=True,
        txt_feature=False,
        cascade_path=None,
        style_path=None
    ):
        self.split = split
        self.size = size
        self.scale = scale
        self.data_len = data_len
        self.use_cascade = use_cascade
        self.use_txt = use_txt
        self.txt_level = txt_level
        self.txt_feature = txt_feature
        self.cascade_path = cascade_path
        self.style_path = style_path

        self.rs_dir = "{}/rs_{}".format(dataroot, size)
        self.map_dir = "{}/map_{}".format(dataroot, size)
        self.ref_dir = "{}/ref_scale_{}_{}".format(dataroot, scale, size)

        tile_list = os.path.join(dataroot, datalist)
        self.tile_names = pd.read_csv(tile_list)["tilename"].to_list()
        self.zoom_scale = int(math.sqrt(scale))

        level_parts = datalist.split("_")
        self.max_level = int(level_parts[1])
        self.min_level = int(level_parts[2].split(".")[0])

        if self.style_path:
            # Style image directory
            if os.path.isdir(self.style_path):
                self.style_path = Util.get_paths_from_images(self.style_path)
            # Single image for style guidance
            else:
                self.style_path = [self.style_path]

        self.dataset_len = len(self.tile_names)
        if self.data_len <= 0:
            self.data_len = self.dataset_len
        else:
            self.data_len = min(self.data_len, self.dataset_len)

    def __len__(self):
        return self.data_len

    def __getitem__(self, index):
        img_map = None
        img_rs = None
        img_ref_scale = None
        img_style = None

        tile_name = self.tile_names[index]
        tile_level = int(tile_name.split("_")[0])
        assert (self.min_level <= tile_level <= self.max_level), "Invalid tile level: {}".format(tile_level)
        parent_filename, x, y, size = Tile.get_filename_parent_tile_scale(f"{tile_name}.png", zoom_scale=self.zoom_scale)

        img_map_path = "{}/{}.png".format(self.map_dir, tile_name)
        img_rs_path = "{}/{}.png".format(self.rs_dir, tile_name)
        assert os.path.exists(img_map_path), "MAP Image not found: {}".format(img_map_path)
        assert os.path.exists(img_rs_path), "RS Image not found: {}".format(img_rs_path)

        img_map = (np.asarray(Image.open(img_map_path).convert("RGB")) / 255.0).astype(np.float32)
        img_rs = (np.asarray(Image.open(img_rs_path).convert("RGB")) / 255.0).astype(np.float32)

        # Reference Scale Tile image loading
        if self.use_cascade:
            img_ref_scale_path = None

            if self.cascade_path:
                if tile_level - self.min_level < self.zoom_scale:
                    ref_path = self.map_dir
                    ref_type = "Init"
                    ref_func = lambda img_path: Tile.get_image_parent_tile_scale(img_path, x, y, size)
                    img_ref_scale_path = f"{ref_path}/{parent_filename}"
                else:
                    ref_path = self.cascade_path
                    ref_type = "Cascade"
                    ref_func = lambda img_path: Tile.get_image_parent_tile_scale(img_path, x, y, size)
                    img_ref_scale_path = f"{ref_path}/{parent_filename}"

                if not os.path.exists(img_ref_scale_path):
                    print(f"REF Image not found: {ref_type} - {img_ref_scale_path}")
                    img_ref_scale_path = None

            if img_ref_scale_path is None:
                ref_path = self.ref_dir
                ref_type = "Source"
                ref_func = lambda img_path: (np.asarray(Image.open(img_path).convert("RGB")) / 255.0).astype(np.float32)
                img_ref_scale_path = f"{ref_path}/{tile_name}.png"

            assert os.path.exists(img_ref_scale_path), f"REF Image not found: {ref_type} - {img_ref_scale_path}"
            img_ref_scale = ref_func(img_ref_scale_path)

        if self.style_path:
            if len(self.style_path) > 1:
                img_style = (
                    np.asarray(Image.open(self.style_path[index]).convert("RGB")) / 255.0
                ).astype(np.float32)
            else:
                img_style = (
                    np.asarray(Image.open(self.style_path[0]).convert("RGB")) / 255.0
                ).astype(np.float32)

        if self.use_txt:
            txt = "geographical map tile"

            if self.txt_level:
                map_ditails = {
                    0: {"resolution": 156543, "scale": "1:500000000", "example": "whole world"},
                    1: {"resolution": 78272, "scale": "1:250000000", "example": "whole world"},
                    2: {"resolution": 39136, "scale": "1:150000000", "example": "subcontinental area"},
                    3: {"resolution": 19568, "scale": "1:70000000", "example": "largest country"},
                    4: {"resolution": 9784, "scale": "1:35000000", "example": "whole world"},
                    5: {"resolution": 4892, "scale": "1:15000000", "example": "large African country"},
                    6: {"resolution": 2446, "scale": "1:10000000", "example": "large European country"},
                    7: {"resolution": 1223, "scale": "1:4000000", "example": "small country, US state"},
                    8: {"resolution": 611.496, "scale": "1:2000000", "example": "small country, US state"},
                    9: {"resolution": 305.748, "scale": "1:1000000", "example": "wide area, large metropolitan area"},
                    10: {"resolution": 152.874, "scale": "1:500000", "example": "metropolitan area"},
                    11: {"resolution": 76.437, "scale": "1:250000", "example": "city"},
                    12: {"resolution": 38.219, "scale": "1:150000", "example": "town, or city district"},
                    13: {"resolution": 19.109, "scale": "1:70000", "example": "village, or suburb"},
                    14: {"resolution": 9.555, "scale": "1:35000", "example": "village, or suburb"},
                    15: {"resolution": 4.777, "scale": "1:15000", "example": "small road"},
                    16: {"resolution": 2.389, "scale": "1:8000", "example": "street"},
                    17: {"resolution": 1.194, "scale": "1:4000", "example": "block, park, addresses"},
                    18: {"resolution": 0.597, "scale": "1:2000", "example": "some buildings, trees"},
                    19: {"resolution": 0.299, "scale": "1:1000", "example": "local highway and crossing details"},
                    20: {"resolution": 0.149, "scale": "1:500", "example": "mid-sized building"}
                }

                map_resolution = map_ditails[tile_level]["resolution"]
                map_scale = map_ditails[tile_level]["scale"]
                map_example = map_ditails[tile_level]["example"]

                txt += f", tile zoom level {tile_level}"
                txt += f", reference zoom level {tile_level - self.zoom_scale}"
                txt += f", reference zoom scale {self.scale}X"
                txt += f", map resolution {map_resolution}m/pixel"
                txt += f", map scale {map_scale}"
                txt += f", examples of areas to represent: {map_example}"

            if self.txt_feature:
                txt_book = np.asarray(Image.open(self.ref_scale_path[index]).convert("RGB"))
                building_vec = np.array([226, 224, 210])
                road_vec1 = np.array([254, 254, 254])
                road_vec2 = np.array([255, 204, 133])
                vegetation_vec1 = np.array([217, 233, 198])
                vegetation_vec2 = np.array([228, 236, 213])
                vegetation_vec3 = np.array([208, 230, 190])
                water_vec = np.array([133, 203, 249])
                building_exist = np.all((txt_book == building_vec), axis=-1).any()
                road_exist = (
                    np.all((txt_book == road_vec1), axis=-1).any()
                    or np.all((txt_book == road_vec2), axis=-1).any()
                )
                vegetation_exist = (
                    np.all((txt_book == vegetation_vec1), axis=-1).any()
                    or np.all((txt_book == vegetation_vec2), axis=-1).any()
                    or np.all((txt_book == vegetation_vec3), axis=-1).any()
                )
                water_exist = np.all((txt_book == water_vec), axis=-1).any()

                txt += "; has features include land,"
                if building_exist:
                    txt += "building of regular shape, "
                if road_exist:
                    txt += "road, "
                if vegetation_exist:
                    txt += "vegetation, "
                if water_exist:
                    txt += "water area, "
        else:
            txt = ""

        if self.use_cascade and not (self.style_path):
            [img_ref_scale, img_rs, img_map] = Util.transform_augment(
                [img_ref_scale, img_rs, img_map], split=self.split
            )
        elif not self.use_cascade and self.style_path:
            [img_style, img_rs, img_map] = Util.transform_augment(
                [img_style, img_rs, img_map], split=self.split
            )
        elif self.use_cascade and self.style_path:
            [img_style, img_ref_scale, img_rs, img_map] = Util.transform_augment(
                [img_style, img_ref_scale, img_rs, img_map], split=self.split
            )
        else:
            [img_rs, img_map] = Util.transform_augment(
                [img_rs, img_map], split=self.split
            )

        img_map = img_map * 2 - 1

        base_return = {
            "MAP": img_map,
            "RS": img_rs,
            "scale": self.zoom_scale,
            "level": tile_level,
            "txt": txt,
            "name": tile_name
        }


        if self.use_cascade:
            base_return["REF"] = img_ref_scale
            base_return["ref_type"] = ref_type

        if self.style_path:
            base_return["STYLE"] = img_style

        return base_return