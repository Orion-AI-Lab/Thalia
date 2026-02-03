"""
The annotated Thalia dataset 
"""

import json
import os
import random
import itertools
from datetime import datetime
import matplotlib.pyplot as plt
import einops
import seaborn as sns
import numpy as np
import torch
import xarray as xr
from tqdm import tqdm
from utilities import augmentations


class InSarDataset(torch.utils.data.Dataset):
    """Dataset returning the full InSAR frame timeseries
    of length config['timseries_length'] for each primary date.
    Returns Image (InSAR, Coherence) , labels.
    """
    def __init__(
        self, config, mode="train", verbose=False, webdataset_write=False, eval=False
    ):
        # Parsing arguments
        self.config = config
        self.verbose = verbose
        self.mode = mode
        self.webdataset_write = webdataset_write
        self.eval = eval

        # Parsing configuration
        self.statistics_path = self.config["statistics"]
        self.image_size = self.config["image_size"]
        self.seed = self.config["seed"]
        self.timeseries_length = self.config["timeseries_length"]
        self.allowed_geomorphology_channels = ["insar_difference", "insar_coherence", "dem"]
        self.allowed_atmospheric_channels = ["total_column_water_vapour", "surface_pressure", "vertical_integral_of_temperature"]
        self.atmospheric_channels = self.config["atmospheric_channels"]
        self.geomorphology_channels = self.config["geomorphology_channels"]
        self.annotation_dir = self.config["annotation_path"]
        self.zarr_path = self.config["zarr_path"]

        # Load timeseries data
        self.timeseries_dict = json.load(open(self.config["timeseries_data"], "r"))
        # Load annotations
        self.complete_annotation_dict = json.load(open(self.config["annotation_path"], "r"))

        # Process mode end and start dates for temporal split
        self.mode_start = datetime.strptime(str(self.config[f"{self.mode}_years"][0]), "%Y%m%d")
        self.mode_end = datetime.strptime(str(self.config[f"{self.mode}_years"][1]), "%Y%m%d")

        # Initializing dataset variables
        self.info = {key: [] for key in ["test", "val", "train"]}
        self.contain_artifacts = {}
        self.corrupted_flag = {}
        self.no_info = {}
        self.interferograms_timeseries = []
        self.lows, self.mediums, self.highs = [], [], []
        self.positives, self.negatives = {}, {}
        self.pos_count, self.neg_count = {}, {}
        self.accepted_count, self.rejected_count = 0, 0
        self.dup_positives, self.dup_negatives = 0, 0
        self.duplicates_per_position = [0 for i in range(self.timeseries_length)]
        self.no_coherence_count = 0
        self.frame_dict = self._get_unique_frames()
    

        # Setting random seed
        self._set_random_seed(self.seed)
        # Setting up augmentations
        self.augmentations = (
            augmentations.get_augmentations(config, config["image_size"])
            if config["augment"]
            else None
        )

        # Initialization checks
        assert self.timeseries_length > 0, "Timeseries length must be greater than 0"
        assert set(self.geomorphology_channels).issubset(set(self.allowed_geomorphology_channels)), f"Invalid InSAR channels: {self.geomorphology_channels}. Allowed channels: {self.allowed_geomorphology_channels}"
        assert set(self.atmospheric_channels).issubset(set(self.allowed_atmospheric_channels)), f"Invalid atmospheric channels: {self.atmospheric_channels}. Allowed channels: {self.allowed_atmospheric_channels}"
        self.unwrapped_atmospheric_channels = []
        for var in self.atmospheric_channels:
            self.unwrapped_atmospheric_channels.append(f"primary_date_{var}")
            self.unwrapped_atmospheric_channels.append(f"secondary_date_{var}")
        
        self.channels = self.geomorphology_channels + self.unwrapped_atmospheric_channels
        self.num_channels = len(self.channels)

        assert self.timeseries_dict, "Timeseries data file is empty"
        assert self.frame_dict, "Data path is empty"

        ###########################################
        ### Start of the dataset initialization ###
        ###########################################
        self.mode_timeseries = self._time_slice_timeseries(self.timeseries_dict)
                
        # Parsing the dataset frames
        for frame_id, prim_date_series in tqdm(self.mode_timeseries.items()):
            # Parsing the timeseries of each frame based on primary dates
            for timeseries in prim_date_series.values():

                annotation_dict = {}
                for unique_id in timeseries["uniqueID"]:
                    full_annotation_path = (
                                self.annotation_dir + str(unique_id) + ".json"
                            )
                    annotation_dict[unique_id] = self.complete_annotation_dict[str(unique_id)]
                
                # Check with Non_Deformation as positives have Deformation or Earthquake
                timeseries["label"] = ['Non_Deformation' not in annotation_dict[key]['label'] for key in annotation_dict]
                timeseries_label = any(timeseries["label"])

                # Random duplication if len < timeseries_length in order to reach the desired length
                if timeseries["length"] < self.timeseries_length:
                    # Keep track of the duplicated positives and negatives
                    dups = self.timeseries_length - timeseries["length"]
                    if timeseries_label == 0:
                        self.dup_negatives += dups
                    else:
                        self.dup_positives += dups

                    timeseries = self.duplicate_entries(
                        timeseries, self.timeseries_length
                    )
                    timeseries = [timeseries]

                elif timeseries["length"] > self.timeseries_length:
                    # Determine min_distinct
                    if self.mode=="train":
                        min_distinct = self.timeseries_length-2 if timeseries_label == 0 else self.timeseries_length-1
                    else:
                        min_distinct = 1

                    timeseries = self.get_possible_subsets(
                        sample=timeseries, set_length=self.timeseries_length, label=timeseries_label, mode=self.mode, min_distinct=min_distinct
                    )
                    
                else:
                    timeseries = [timeseries]

                for sample in timeseries:
                    lows_list, mediums_list, highs_list = [], [], []
                    label_list = []
                    annotation_path_list, annotation_list = [], []
                    assert sample['length'] == self.timeseries_length
                    assert all(len(sample[key]) == self.timeseries_length for key in sample.keys() if key not in ['length', 'frame_id']), f"Timeseries length mismatch for {sample}"
                    for unique_id in sample["uniqueID"]:
                        annotation = annotation_dict[unique_id]

                        label, highs, lows, mediums = self._get_labels(annotation)
                        lows_list.append(lows)
                        mediums_list.append(mediums)
                        highs_list.append(highs)
                        label_list.append(label)

                        annotation_path_list.append(full_annotation_path)
                        annotation_list.append(annotation)

                    self.lows.append(lows_list)
                    self.mediums.append(mediums_list)
                    self.highs.append(highs_list)

                    timeseries_label = int(
                        np.sum(label_list) > 0
                    )  # Returns 0 for all 0 labels of the timeseries and 1 for at least one 1 label
                    
                    self.timeseries_dict = {
                        "frame_id": frame_id,
                        "annotation": annotation_list,
                        "label": label_list,
                        "annotation_path": annotation_path_list,
                    }
                    if timeseries_label == 0:
                        if frame_id not in self.negatives.keys():
                            self.negatives[frame_id] = []
                        self.negatives[frame_id].append(self.timeseries_dict)
                    elif timeseries_label == 1:
                        if frame_id not in self.positives.keys():
                            self.positives[frame_id] = []
                        self.positives[frame_id].append(self.timeseries_dict)

        list_positives = list(itertools.chain.from_iterable(self.positives.values()))
        list_negatives = list(itertools.chain.from_iterable(self.negatives.values()))

        #################################################################
        ###             OVERSAMPLING REQUIRES WEBDATASET              ###
        #################################################################
        # We choose to undersample the negative class and give the webdataset 3*len(positives) samples. 
        # Out of these negatives each epoch will contain a random subsample so that pos~neg
        if not config["webdataset"]:
            print("*********************** WARNING: Oversampling requires webdataset... ***********************")

        #if self.mode == "train":
        #    filtered_negatives = list(np.random.choice(list_negatives, min(10*len(list_positives), len(list_negatives)), replace=False))
        #    self.interferograms_timeseries = list_positives + filtered_negatives
        #else:
        #    self.interferograms_timeseries = list_positives + list_negatives

        self.interferograms_timeseries = list_positives + list_negatives
        print(f"Mode: {self.mode} - Number of positives: {len(list_positives)}, Number of negatives: {len(list_negatives)}")
        random.shuffle(self.interferograms_timeseries)

        self.num_examples = len(self.interferograms_timeseries)
        self.num_positives = sum(len(p) for p in list(self.positives.values()))
        self.num_negatives = sum(len(n) for n in list(self.negatives.values()))

        print("Mode: ", self.mode, " Number of examples: ", self.num_examples)
        print("Number of positives: ", self.num_positives)
        print("Number of negatives: ", self.num_negatives)
        print(f"Unique Frames: {np.unique(self.info[mode])}")
        print(
            f"Duplicated entries: {np.sum(self.duplicates_per_position)} -> {self.duplicates_per_position}"
        )
        print(f"Duplicated positives: {self.dup_positives}")
        print(f"Duplicated negatives: {self.dup_negatives}")

        if self.contain_artifacts:
            print(f"Frames with artifacts: {self.contain_artifacts}")
        if self.corrupted_flag:
            print(f"Frames with corrupted flags: {self.corrupted_flag}")
        if self.no_info:
            print(f"Frames with no info: {self.no_info}")
        print("=" * 20)


    def __len__(self):
        return self.num_examples

    @staticmethod
    def _set_random_seed(seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True

    def _spatial_slice_timeseries(self, timeseries):
        # keep only timeseries.keys() that are in self.{mode}_frames
        mode_frames = getattr(self, f"{self.mode}_frames")
        mode_timeseries = {}
        for frame_id in mode_frames:
            if frame_id in timeseries.keys():
                mode_timeseries[frame_id] = timeseries[frame_id]
        return mode_timeseries
    
    def _time_slice_timeseries(self, timeseries):
        # keep only the primary_dates that are in the {mode} years
        mode_timeseries = {}
        for frame_id, prim_date_series in timeseries.items():
            for prim_date in prim_date_series.keys():
                prim_datetime = datetime.strptime(prim_date, "%Y%m%d")
                if self.mode_start <= prim_datetime <= self.mode_end:
                    if frame_id not in mode_timeseries.keys():
                        mode_timeseries[frame_id] = {}
                    mode_timeseries[frame_id][prim_date] = prim_date_series[prim_date]
        return mode_timeseries

    def duplicate_entries(self, dictionary, target_length):
        init_len = dictionary["length"]
        if dictionary["length"] < target_length:
            remaining_length = target_length - init_len
            for i in range(remaining_length):
                random_choice = random.randint(0, init_len - 1 + i)
                for key, value in dictionary.items():
                    if key != "length":
                        # Duplicate the last entry and append it to the list
                        value.insert(random_choice, value[random_choice])
                        self.duplicates_per_position[random_choice] += 1
                dictionary["length"] += 1
        
        assert dictionary["length"] == target_length
        return dictionary

    def _get_unique_frames(self):
        """Get the unique dataset frames as listed in the specified directory.

        Returns:
            dict: A dictionary with the unique frames as keys and their corresponding index as values.
        """
        frames = os.listdir(self.zarr_path)
        frames = [frame.split(".")[0] for frame in frames]
        unique_frames = np.unique(frames)
        return {frame: idx for idx, frame in enumerate(unique_frames)}

    def _get_labels(self, annotation):
        """Get the labels of the event from the annotations.

        Args:
            annotation (dict): A dictionary containing the annotation data.

        Returns:
            label: 1 for 'Deformation' and 0 for 'Non_Deformation'
            highs: 1 if 'High' in the intensity level, 0 otherwise
            lows: 1 if 'Low' in the intensity level, 0 otherwise
            mediums: 1 if 'Medium' in the intensity level, 0 otherwise
        """
        label = 0 if "Non_Deformation" in annotation["label"] else 1
        lows = 1 if "Low" in annotation["intensity_level"] else 0
        mediums = 1 if "Medium" in annotation["intensity_level"] else 0
        highs = 1 if "High" in annotation["intensity_level"] else 0

        return label, highs, lows, mediums

    def get_possible_subsets(self, sample, set_length, label, mode, min_distinct=1):
        """Get all possible order-preserving subsets of the timeseries of length set_length.

        Args:
            sample (dict): A dictionary containing a timeseries of frames.
            set_length (int, optional): The length of the target timeseries.
            label (int): The label of the timeseries.

        Returns:
            list: A list of all possible order-preserving subsets of the timeseries of length set_length.
        """
        if set_length == 1:
            possible_subsets = []
            for i in range(len(sample["uniqueID"])):
                subset = {}
                for key, values in sample.items():
                    if key == 'length':
                        subset[key] = 1  # Since the length of the subset is 1
                    elif key == "frame_id":
                        subset[key] = [values[i]]  # Keep frame_id the same for each frame
                    else:
                        subset[key] = [values[i]]  # Keep the frame's data as a single-element list
                possible_subsets.append(subset)
            return possible_subsets
        else:
            total_length = len(sample["uniqueID"])
            all_combos = list(itertools.combinations(range(total_length), set_length))
            unique_combos = []
            
            while all_combos:
                current = all_combos.pop(0)
                unique_combos.append(current)
                
                # Filter out combinations that share at least 'm' elements with 'current'
                all_combos = [combo for combo in all_combos if len(set(current) & set(combo)) < min_distinct]
            
            possible_subsets = []
            for comb in unique_combos:
                subset = {}
                for key, values in sample.items():
                    if key == 'length':
                        subset[key] = set_length
                    elif key == "frame_id":
                        subset[key] = (
                            values  # Keep frame_id the same across all subsets
                        )
                    else:
                        subset[key] = [
                            values[i] for i in comb
                        ]
                possible_subsets.append(subset)

            return possible_subsets

    def augment(self, insar_timeseries):
        """Augment the image with the specified augmentations."""
        # Bands shape: (num_channels, len(latitude), len(longitude), timeseries_length)
        bands = np.stack([insar_timeseries[c] for c in self.channels], axis=0).squeeze()
        # Masks shape: (len(latitude), len(longitude), timeseries_length)
        masks = insar_timeseries['deformation_mask'].values.squeeze()

        # Rearrange bands and masks to match the expected format
        # Timestep is given as different channels
        bands = einops.rearrange(bands, "c t h w -> h w (c t)")
        masks = einops.rearrange(masks, "t h w -> h w t")

        transform = self.augmentations(image=bands, mask=masks)
        augmented_bands = transform["image"]
        augmented_masks = transform["deformation_mask"]
    
        # Split time (T) back from channel (C*T)
        augmented_bands = einops.rearrange(augmented_bands, "h w (c t) -> t c h w", t=self.timeseries_length)
        augmented_masks = einops.rearrange(augmented_masks, "h w t -> t h w")

        augmented_insar = torch.cat([torch.from_numpy(augmented_bands), torch.from_numpy(augmented_masks).unsqueeze(1)], dim=1)
        return augmented_insar

    @staticmethod
    def print_raster(raster):
        print(
            f"shape: {raster.rio.shape}\n"
            f"resolution: {raster.rio.resolution()}\n"
            f"bounds: {raster.rio.bounds()}\n"
            f"sum: {raster.sum().item()}\n"
            f"CRS: {raster.rio.crs}\n"
        )

    def crop_around_deformation(
        self,
        insar,
        target_mask_method="union",
        crop_size=512,
        max_retries=10,
        annotation=None
    ):
        """
        Crop around deformation ensuring the target mask is fully contained,
        avoiding areas with no data (zeros) and performing a somewhat random crop.

        Args:
            insar (torch.Tensor): Input tensor of shape (T, num_channels, H, W) with channels as difference, coherence, mask.
            crop_size (int): Desired crop size.
            target_mask_method (str): Method to select the target mask. Options: 'last', 'peak', 'union'.

        Returns:
            torch.Tensor: Cropped tensor of shape (T, num_channes, crop_size, crop_size).
        """
        T, C, H, W = insar.shape
        assert C == self.num_channels + 1, f"Expected {self.num_channels + 1} channels, got {C}."
        # Ensure crop size is not larger than the image dimensions
        crop_size = min(crop_size, H, W)

        # The -1 refers to the last channel, which is the mask
        if target_mask_method == "last":
            target_mask = insar[-1, -1, :, :]
        elif target_mask_method == "peak":
            target_mask = insar[:, -1, :, :].sum(dim=0)  # Sum across time to get the peak mask
        elif target_mask_method == "union":
            target_mask = (insar[:, -1, :, :] > 0).any(dim=0).float()  # Union of all masks
        else:
            raise ValueError(f"Invalid target_mask_method: {target_mask_method}")

        # Ensure target_mask is binary (0 or 1)
        target_mask = (target_mask > 0).float()

        # Step 2: Find bounding box around the target mask
        if target_mask.max() > 0:  # If deformation exists
            coords = torch.nonzero(target_mask, as_tuple=False)  # Get coordinates of mask=1 regions
            min_y, min_x = coords.min(dim=0).values
            max_y, max_x = coords.max(dim=0).values
        else:  # No deformation, center the crop randomly around the center
            min_y, min_x, max_y, max_x = 0, 0, H - 1, W - 1

        # Step 3: Compute crop center with randomness
        center_y = (min_y + max_y) // 2
        center_x = (min_x + max_x) // 2

        for _ in range(max_retries):
            # Add small randomness to the center, ensuring we stay within bounds
            rand_y = torch.randint(-crop_size // 4, crop_size // 4 + 1, (1,)).item()
            rand_x = torch.randint(-crop_size // 4, crop_size // 4 + 1, (1,)).item()
            center_y = max(crop_size // 2, min(H - crop_size // 2, center_y + rand_y))
            center_x = max(crop_size // 2, min(W - crop_size // 2, center_x + rand_x))
    
            # Step 4: Perform cropping
            start_y = center_y - crop_size // 2
            start_x = center_x - crop_size // 2
            end_y = start_y + crop_size
            end_x = start_x + crop_size
    
            cropped_insar = insar[:, :, start_y:end_y, start_x:end_x]
            if cropped_insar[:, -1, :, :].max() == target_mask.max():
                break

        # Step 5: Validate crop contains valuable information
        assert cropped_insar.shape[-2:] == (crop_size, crop_size), "Crop dimensions are incorrect."
        assert cropped_insar[:, -1, :, :].max() == target_mask.max(), f"Target mask has {target_mask.max()} while crop has {cropped_insar[:, 2, :, :].max()}. {annotation['frame_id']} {annotation['annotation'][0]['primary_date']} {[a['secondary_date'] for a in annotation['annotation']]}"

        return cropped_insar

    def center_crop(self, image_tensor, crop_size=512):
        """
        Perform a center crop of the given size on the input tensor.

        Args:
            image_tensor (torch.Tensor): Input tensor of shape (Time, Mask, Channels, Height, Width).
            crop_size (int): Desired crop size.

        Returns:
            torch.Tensor: Center cropped tensor.
        """
        _, _, H, W = image_tensor.shape  # Time, Mask, Channels, Height, Width

        # Ensure crop size is not larger than the image dimensions
        crop_size = min(crop_size, H, W)

        # Calculate the center coordinates
        center_y, center_x = H // 2, W // 2

        # Calculate the crop boundaries
        y1 = max(0, center_y - crop_size // 2)
        x1 = max(0, center_x - crop_size // 2)
        y2 = y1 + crop_size
        x2 = x1 + crop_size

        # Perform the center crop
        cropped_image_tensor = image_tensor[:, :, y1:y2, x1:x2]

        return cropped_image_tensor

    def random_crop_with_checks(
        self,
        image_tensor,
        crop_size=512,
        min_variance=1e-3,
        min_valuable_ratio=0.25,
        max_retries=10,
    ):
        """
        Perform a random crop with variability and valuable info checks.

        Args:
            image_tensor (torch.Tensor): Input tensor of shape (Time, Mask, Channels, Height, Width).
            crop_size (int): Desired crop size.
            min_variance (float): Minimum variance threshold for random crops.
            min_valuable_ratio (float): Minimum fraction of non-background pixels in the crop.
            max_retries (int): Maximum retries for random crops with sufficient variability.

        Returns:
            torch.Tensor: Randomly cropped tensor.
        """
        _, _, H, W = image_tensor.shape  # Time, Channels, Height, Width

        # Perform a random crop with variability and valuable info checks
        max_valuable_entry = {"valuable_ratio": 0.0, "cropped_image_tensor": None}
        for _ in range(max_retries):
            y1 = torch.randint(0, H - crop_size + 1, (1,)).item()
            x1 = torch.randint(0, W - crop_size + 1, (1,)).item()
            cropped_image_tensor = image_tensor[
                :, :, y1 : y1 + crop_size, x1 : x1 + crop_size
            ]

            # Check variance of the cropped region across channels
            crop_variance = cropped_image_tensor.var(dim=(-1, -2, -3)).mean()

            # Check percentage of valuable (non-background) pixels
            background_pixel = cropped_image_tensor[:, :2, 0, 0].mean()
            valuable_pixels = (cropped_image_tensor != background_pixel).float()
            valuable_ratio = valuable_pixels.mean().item()

            if valuable_ratio >= min_valuable_ratio and crop_variance >= min_variance:
                break
            elif valuable_ratio > max_valuable_entry["valuable_ratio"]:
                max_valuable_entry["valuable_ratio"] = valuable_ratio
                max_valuable_entry["cropped_image_tensor"] = cropped_image_tensor
        else:
            # If no crop with sufficient variability is found, return the last one
            cropped_image_tensor = max_valuable_entry["cropped_image_tensor"]

        return cropped_image_tensor
    
    
    def parse_timeseries(self, sample):
        """Parse the timeseries data.

        Args:
            sample (dict): A dictionary containing the timeseries data.

        Returns:
            torch.tensor: The image of shape timeseries_length, num_channels, image_size, image_size.
        """
        frame_id = sample["frame_id"]

        frame_path = os.path.join(self.zarr_path, f"{frame_id}.zarr")
        frame_insars = xr.open_zarr(frame_path)
        
        primary_date = np.datetime64(datetime.strptime(sample["annotation"][0]['primary_date'], "%Y%m%d"))
        secondary_dates = [np.datetime64(datetime.strptime(annot["secondary_date"], "%Y%m%d")) for annot in sample["annotation"]]
        try:
            insar_timeseries = frame_insars.sel(
                primary_date=primary_date, 
                secondary_date=secondary_dates
            )[self.channels + ["deformation_mask"]]

        except KeyError as e:
            print(f"Could not find one of: \n {primary_date} \n {secondary_dates} \n for {frame_id}.zarr")
            print(f"KeyError: {e}")
            exit()
            return None
        
        # Include Earthquake as a 1 label
        deformation_mask = insar_timeseries["deformation_mask"]
        deformation_mask.values = np.where(deformation_mask.values == 2, 1, deformation_mask.values)

        if np.unique(insar_timeseries["deformation_mask"].values).all() not in [0, 1]:
            print(sample)

        assert torch.all(torch.isin(torch.unique(torch.from_numpy(insar_timeseries["deformation_mask"].values)), torch.tensor([0, 1]))), \
        f"Deformation mask contains unexpected values: {torch.unique(torch.from_numpy(insar_timeseries['deformation_mask'].values))}. Sample: \n {sample}"

        # Repeat the DEM layer for each time step
        T = insar_timeseries.sizes["secondary_date"]  # or len(secondary_dates)

        tensors = []
        for var in self.channels + ["deformation_mask"]:
            data = insar_timeseries[var].values  # could be [T, H, W] or [H, W] (dem)
            
            if data.ndim == 2:  # Static layer like DEM
                data = np.repeat(data[np.newaxis, ...], T, axis=0)  # Shape: [T, H, W]
            
            tensors.append(torch.from_numpy(data))
        
        insar_tensor = torch.stack(tensors, dim=1)  # [T, C, H, W]

        assert np.unique(insar_tensor[:, 2, :, :].numpy().all()) in [0, 1], f"Deformation mask values are not binary: {np.unique(insar_tensor[:, 2, :, :].numpy())}"

        # Cropping is performed in order to ensure that all the frames have the same size
        # TODO: Implement a "crop_at_maximum_boundaries" where the crop is just the maximum box of all frames
        insar_tensor = self.crop_around_deformation(insar=insar_tensor, crop_size=self.image_size, target_mask_method=self.config['mask_target'], annotation=sample)
        
        #if self.verbose:
        #self.produce_variable_figures_from_tensor(
        #    var_tensor=insar_tensor,
        #    var_names=self.channels + ["deformation_mask"],
        #    info=sample,
        #    root_path="DEBUG",
        #)

        assert (
            insar_tensor.shape
            == torch.empty(
                self.timeseries_length,
                self.num_channels + 1, # +1 for the mask
                self.image_size,
                self.image_size,
            ).shape
        ), f"Image tensor with shape {insar_tensor.shape} is not {torch.empty(self.timeseries_length, self.num_channels+1, self.image_size, self.image_size).shape}"

        return insar_tensor

    def _prepare_model_input(self, insar_tensor):
        if "unet" or "deeplab" in self.config["architecture"].lower():
            image_tensor = insar_tensor[:, :self.num_channels, :, :]
            mask_tensor = insar_tensor[:, self.num_channels, :, :]

            prep_image = image_tensor.reshape(
                self.timeseries_length*self.num_channels,
                self.config["image_size"],
                self.config["image_size"],
            )
            prep_mask = mask_tensor.reshape(
                self.timeseries_length,
                self.config["image_size"],
                self.config["image_size"],
            )

            if self.webdataset_write == False:
                if self.config["timeseries_length"] != 1:
                    if self.config["mask_target"] == "peak":
                        counts = torch.sum(prep_mask, dim=(1, 2))
                        prep_mask = prep_mask[torch.argmax(counts), :, :, :]
                    elif self.config["mask_target"] == "union":
                        prep_mask = torch.sum(prep_mask, dim=0)
                        prep_mask = torch.where(prep_mask > 0, 1, 0)
                    elif self.config["mask_target"] == "last":
                        prep_mask = prep_mask[-1, :, :]
                else:
                    prep_mask = prep_mask[-1, :, :]  # Last channel

            del image_tensor, mask_tensor
            return prep_image, prep_mask
        else:
            raise NotImplementedError(
                "Model {} not implemented for dataloader output".format(self.config["model"])
            )

    def __getitem__(self, index):
        sample = self.interferograms_timeseries[index]
        if sample['frame_id'] not in self.pos_count.keys():
            self.pos_count[sample['frame_id']] = 0
            self.neg_count[sample['frame_id']] = 0

        self.neg_count[sample['frame_id']] += int(np.any(
            [int("Non_Deformation" in annot["label"]) for annot in sample["annotation"]]
        ))
        self.pos_count[sample['frame_id']] += int(np.any(
                [int("Non_Deformation" not in annot["label"]) for annot in sample["annotation"]]
            ))
        
        image_with_mask = self.parse_timeseries(sample=sample)  # (T, C+1, H, W)

        image_tensor, mask_tensor = self._prepare_model_input(image_with_mask)

        if (
            self.eval is True
            or self.webdataset_write is True
            or self.config["webdataset"] is True
        ):
            return image_tensor.float(), mask_tensor.clone().detach().long(), sample
        else:
           return image_tensor.float(), mask_tensor.clone().detach().long()