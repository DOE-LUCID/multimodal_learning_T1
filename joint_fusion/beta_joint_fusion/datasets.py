import os
import gc
import psutil
import pickle
from pathlib import Path
import numpy as np
from pdb import set_trace
import time
from PIL import Image
import torch
from torch.utils.data.dataset import Dataset  # For custom datasets
from torchvision import transforms
import cv2
import h5py
import ast
import logging
from collections import OrderedDict
from threading import Lock


# Set up logging for debugging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# using opencv with cached images
# caching significantly accelerates data loading after the first epoch
class CustomDataset(Dataset):
    def __init__(self, opt, mapping_df, split=None, mode="wsi", train_val_test="train"):
        self.opt = opt
        self.train_val_test = train_val_test
        self.mapping_df = mapping_df
        self.cache_dir = "./image_cache"
        os.makedirs(self.cache_dir, exist_ok=True)

        # transformations/augmentations for WSI data
        # not required: already available in the functions for loading the models in generate_wsi_embeddings.py
        self.transforms = transforms.ToTensor()
        # log transform for rnaseq data
        # not required for SPECTRA processed data

    def __getitem__(self, index):
        start_time = time.time()

        if torch.is_tensor(index):
            index = index.tolist()

        sample = self.mapping_df.iloc[index]
        tcga_id = self.mapping_df.index[index]
        # days_to_death = sample['days_to_death']
        # days_to_last_followup = sample['days_to_last_followup']
        days_to_event = sample["time"]
        event_occurred = 1 if sample["event_occurred"] == "Dead" else 0
        tiles = sample["tiles"]

        step1_time = time.time()

        # Load preprocessed images if they exist, else preprocess and cache
        cached_images = []
        for tile in tiles:
            cached_image_path = os.path.join(self.cache_dir, f"{tile}.pt")
            try:
                if os.path.exists(cached_image_path):
                    cached_image = torch.load(cached_image_path)
                    # set_trace()
                    # cached_image = self.transforms(cached_image)
                    if not isinstance(cached_image, torch.Tensor):
                        # print("Skipping ToTensor(), already a tensor [for uni]")
                        cached_image = self.transforms(cached_image)
                else:
                    raise FileNotFoundError
            except (FileNotFoundError, RuntimeError):
                image_path = os.path.join(self.opt.input_wsi_path, tile)
                image = cv2.imread(image_path)
                if image is None:
                    raise FileNotFoundError(f"Image {tile} not found at {image_path}")

                # check if the tiles are in RGB and BGR format and convert to RGB if required

                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(image)
                cached_image = self.transforms(image)
                # cached_image = transforms.ToTensor()(image).to(device).requires_grad_()
                torch.save(cached_image, cached_image_path)
            cached_images.append(cached_image)

        step2_time = time.time()
        cached_images = torch.stack(cached_images)

        # rnaseq_data = sample['rnaseq_data']
        # set_trace()
        rnaseq_data = ast.literal_eval(sample["rnaseq_data"])
        # set_trace()
        rnaseq_values = np.array(list(rnaseq_data.values()), dtype=np.float32)
        # convert to PyTorch tensor and enable gradient flow
        x_omic = torch.from_numpy(rnaseq_values).requires_grad_()
        # x_omic = torch.tensor(rnaseq_data, dtype=torch.float32)

        step3_time = time.time()

        # Timing print statements
        print(
            f"Index: {index}, Step 1: {step1_time - start_time:.4f}s, Step 2: {step2_time - step1_time:.4f}s, Step 3: {step3_time - step2_time:.4f}s"
        )

        return tcga_id, days_to_event, event_occurred, cached_images, x_omic

    def __len__(self):
        return len(self.mapping_df)


class HDF5Dataset(Dataset):
    def __init__(self, opt, h5_file, split, mode="wsi", train_val_test="train"):
        self.opt = opt
        self.train_val_test = train_val_test
        self.split = split
        self.mode = mode

        self.h5_file_path = h5_file
        self.h5_file = h5py.File(h5_file, "r")

        if split == "all":
            self.dataset = self.h5_file
        else:
            self.dataset = self.h5_file[split]

        # Pre-compute patient IDs
        self.patient_ids = list(self.dataset.keys())
        logger.info(f"Loaded {len(self.patient_ids)} patients for split: {split}")

        # Memory management
        self._setup_memory_management()

        # Persistent cache directory
        self.cache_dir = Path("./caches") / split
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Pre-computed normalization constants (moved to device for efficiency)
        self.register_normalization_constants()

        # Setup transforms once
        self.setup_transforms()

        # Pre-process RNA-seq data (lightweight, one-time computation)
        self.precomputed_omic = {}
        self._precompute_omic_data()

        # Multi-caching system
        self.setup_caching()

        # Thread safety for caching
        self._cache_lock = Lock()

        logger.info(
            f"Dataset initialized with {self.available_memory_gb:.1f}GB available memory"
        )

    def _setup_memory_management(self):
        """Dynamic memory management based on system resources"""

        memory = psutil.virtual_memory()
        self.total_memory_gb = memory.total / (1024**3)
        self.available_memory_gb = memory.available / (1024**3)

        # Use only 30% of available memory
        self.max_cache_memory_gb = self.available_memory_gb * 0.3

        # Estimate memory per image (256x256x3 RGB + tensor overhead)
        self.estimated_image_memory_mb = (256 * 256 * 3 * 4) / (
            1024**2
        )  # 4 bytes per float32

        # Calculate max cached patients
        images_per_patient = 200  # NEED_ADD: Set right now but adjust based on params

        memory_per_patient_mb = self.estimated_image_memory_mb * images_per_patient
        self.max_cached_patients = max(
            1, int((self.max_cache_memory_gb * 1024) / memory_per_patient_mb)
        )

        logger.info(
            f"Memory management: {self.max_cache_memory_gb:.1f}GB allocated, "
            f"max {self.max_cached_patients} patients in cache"
        )

    def register_normalization_constants(self):
        """Precompute and register normalization constants"""
        # Pretrained constants
        mean = torch.tensor([0.70322989, 0.53606487, 0.66096631])
        std = torch.tensor([0.21716536, 0.26081574, 0.20723464])

        # Register as buffers so they move with the dataset if needed
        self.register_buffer("norm_mean", mean.view(3, 1, 1))
        self.register_buffer("norm_std", std.view(3, 1, 1))

    def register_buffer(self, name, tensor):

        setattr(self, name, tensor)

    def setup_transforms(self):
        if self.train_val_test == "train":
            # Training transforms with augmentation
            self.transform_list = [
                transforms.RandomHorizontalFlip(0.5),
                transforms.RandomVerticalFlip(0.5),
                transforms.ColorJitter(
                    brightness=0.1, contrast=0.1, saturation=0.05, hue=0.01
                ),
            ]
            self.use_augmentation = True
        else:
            # No augmentation for validation/test
            self.transform_list = []
            self.use_augmentation = False

        # Base transforms (always applied)
        self.base_transforms = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.70322989, 0.53606487, 0.66096631],
                    std=[0.21716536, 0.26081574, 0.20723464],
                ),
            ]
        )

    def setup_caching(self):
        """Setup multi-level caching system"""
        # Level 1: In-memory LRU cache for frequently accessed patients
        self.memory_cache = OrderedDict()

        # Level 2: Disk cache for preprocessed data
        self.disk_cache_enabled = True

        # Level 3: Lazy loading flags
        self.lazy_load = True

        # Cache statistics
        self.cache_hits = 0
        self.cache_misses = 0

    def _precompute_omic_data(self):
        """
        Pre-compute all RNA-Seq transforms once during initialization.
        """

        logger.info("Pre-computing RNA-Seq data transformations...")

        cache_file = self.cache_dir / "precomputed_omic.pkl"

        if cache_file.exists():
            logger.info("Loading precomputed omic data from cache...")
            with open(cache_file, "rb") as f:
                self.precomputed_omic = pickle.load(f)
        else:
            for i, patient_id in enumerate(self.patient_ids):
                if i % 100 == 0:
                    logger.info(f"Processing omic data: {i}/{len(self.patient_ids)}")

                patient_data = self.dataset[patient_id]
                rnaseq_data = patient_data["rnaseq_data"][()]

                # Apply log1p transform and convert to tensor
                rnaseq_transformed = np.log1p(rnaseq_data).astype(np.float32)
                self.precomputed_omic[patient_id] = torch.from_numpy(rnaseq_transformed)

            # Save to cache
            with open(cache_file, "wb") as f:
                pickle.dump(self.precomputed_omic, f)

        logger.info(
            f"RNA-seq data precomputed for {len(self.precomputed_omic)} patients"
        )

    def _get_cache_key(self, patient_id):
        """Generate cache key including augmentation state"""

        return f"{patient_id}_{self.train_val_test}"

    def _load_from_disk_cache(self, patient_id):
        """Load preprocessed images from disk cache"""
        cache_key = self._get_cache_key(patient_id)
        cache_file = self.cache_dir / f"{cache_key}_images.pt"

        if cache_file.exists():
            try:
                return torch.load(cache_file, map_location="cpu")
            except Exception as e:
                logger.warning(f"Failed to load cache for {patient_id}: {e}")
                # Remove corrupted cache file
                cache_file.unlink(missing_ok=True)
        return None

    def _save_to_disk_cache(self, patient_id, images):
        """Save preprocessed images to disk cache"""
        if not self.disk_cache_enabled:
            return

        cache_key = self._get_cache_key(patient_id)
        cache_file = self.cache_dir / f"{cache_key}_images.pt"

        try:
            torch.save(images, cache_file)
        except Exception as e:
            logger.warning(f"Failed to save cache for {patient_id}: {e}")

    def _manage_memory_cache(self):
        """Manage memory cache size using LRU eviction"""
        while len(self.memory_cache) >= self.max_cached_patients:
            # Remove least recently used item
            oldest_key = next(iter(self.memory_cache))
            removed_item = self.memory_cache.pop(oldest_key)
            del removed_item  # Explicit deletion

        # Trigger garbage collection if memory cache is getting full
        if len(self.memory_cache) % 10 == 0:
            gc.collect()

    def _vectorized_image_loading(self, patient_data):
        """Optimized vectorized image loading"""
        images_group = patient_data["images"]
        image_keys = list(images_group.keys())

        # Pre-allocate arrays for better memory efficiency
        n_images = len(image_keys)
        images_np = np.empty((n_images, 256, 256, 3), dtype=np.uint8)

        # Batch load all images
        for i, key in enumerate(image_keys):
            images_np[i] = images_group[key][()]

        # Convert to tensor in one operation
        images_tensor = torch.from_numpy(images_np).float()

        # Normalize to [0,1]
        images_tensor /= 255.0

        # Convert from NHWC to NCHW
        images_tensor = images_tensor.permute(0, 3, 1, 2)

        # Apply normalization
        images_tensor = (images_tensor - self.norm_mean) / self.norm_std

        return images_tensor

    def _apply_augmentations_batch(self, images_tensor):
        """Apply augmentations to batch of images efficiently"""
        if not self.use_augmentation:
            return images_tensor

        # Apply transforms that can be batched
        if torch.rand(1).item() < 0.5:  # Random horizontal flip
            images_tensor = torch.flip(images_tensor, [3])

        if torch.rand(1).item() < 0.5:  # Random vertical flip
            images_tensor = torch.flip(images_tensor, [2])

        # Color jitter (simplified version for batch processing)
        if torch.rand(1).item() < 0.5:
            # Random brightness
            brightness_factor = 1.0 + (torch.rand(1).item() - 0.5) * 0.2
            images_tensor = torch.clamp(images_tensor * brightness_factor, 0, 1)

        return images_tensor

    def _load_and_process_images(self, patient_id, patient_data):
        """Optimized image loading with multi-level caching"""
        cache_key = self._get_cache_key(patient_id)

        # Level 1: Check memory cache
        with self._cache_lock:
            if cache_key in self.memory_cache:
                # Move to end (mark as recently used)
                images = self.memory_cache.pop(cache_key)
                self.memory_cache[cache_key] = images
                self.cache_hits += 1
                return images

        # Level 2: Check disk cache
        images = self._load_from_disk_cache(patient_id)
        if images is not None:
            # Add to memory cache
            with self._cache_lock:
                self._manage_memory_cache()
                self.memory_cache[cache_key] = images
            self.cache_hits += 1
            return images

        # Level 3: Load and process from HDF5
        self.cache_misses += 1

        try:
            # Use vectorized loading for better performance
            images = self._vectorized_image_loading(patient_data)

            # Apply augmentations if needed
            if self.use_augmentation:
                images = self._apply_augmentations_batch(images)

            # Save to disk cache for future use
            self._save_to_disk_cache(patient_id, images)

            # Add to memory cache
            with self._cache_lock:
                self._manage_memory_cache()
                self.memory_cache[cache_key] = images

        except Exception as e:
            logger.error(f"Error loading images for patient {patient_id}: {e}")
            # Return dummy data to prevent crashes
            images = torch.zeros((200, 3, 256, 256))

        return images

    def get_cache_stats(self):
        """Get caching statistics"""
        total_requests = self.cache_hits + self.cache_misses
        hit_rate = self.cache_hits / max(total_requests, 1) * 100

        return {
            "hit_rate": hit_rate,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "memory_cache_size": len(self.memory_cache),
            "max_cache_size": self.max_cached_patients,
        }

    def __len__(self):
        return len(self.patient_ids)

    def __getitem__(self, index):
        """Optimized __getitem__ with comprehensive caching"""
        start_time = time.time()

        if torch.is_tensor(index):
            index = index.tolist()

        patient_id = self.patient_ids[index]
        patient_data = self.dataset[patient_id]

        # Load clinical data (fast)
        days_to_event = patient_data["days_to_event"][()]
        event_occurred = patient_data["event_occurred"][()]

        step1_time = time.time()

        # Load precomputed omic data (very fast)
        x_omic = self.precomputed_omic[
            patient_id
        ].clone()  # Clone to avoid modifying cached data

        step2_time = time.time()

        # Load images with caching (optimized)
        images = self._load_and_process_images(patient_id, patient_data)

        # Convert to list for compatibility with existing code
        images_list = [img for img in images]

        step3_time = time.time()

        # Periodic cache statistics logging
        if index % 100 == 0:
            stats = self.get_cache_stats()
            logger.info(
                f"Cache stats: {stats['hit_rate']:.1f}% hit rate, "
                f"{stats['memory_cache_size']}/{stats['max_cache_size']} cached"
            )

        # Optional: Enable gradients for test phase (for saliency maps)
        if self.train_val_test == "test":
            for img in images_list:
                img.requires_grad_(True)

        return patient_id, days_to_event, event_occurred, images_list, x_omic

    def __del__(self):
        """Cleanup resources"""
        if hasattr(self, "h5_file") and self.h5_file is not None:
            self.h5_file.close()
