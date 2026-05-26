# tools/dataloader.py

from pathlib import Path
from collections import Counter, defaultdict
import csv
import json

import cv2
import yaml
import numpy as np
import torch
import torchvision
from tqdm import tqdm

from data_utils import LatexVocab


class LoadDataset(torch.utils.data.Dataset):
    """
    Dataset for image-to-LaTeX recognition.

    Each sample returns:
        image: Tensor with shape [C, H, W]
        target: ([image_name, idx], formula_seq)
    """

    def __init__(
        self,
        dataset_config,
        vocab_path=None,
        image_size=None,
        mode="train",
        cache=False,
        max_size=150,
        transforms=None,
        no_sampling=False,
        no_arrays=False,
        only_basic=False,
        dpi=600,
        retain_pct=0.001,
        seed=1407,
        debug_oov=True,
        oov_max_examples=3,
        oov_topk=200,
        oov_report_path=None,
    ):
        super().__init__()

        if image_size is None:
            raise ValueError("image_size must be provided, e.g. {'height': 256, 'width': 1024}")

        self.formulas = []
        self.styles = []
        self.drop_styles = []

        self.vocab = LatexVocab(vocab_path) if vocab_path is not None else None
        self.max_size = max_size

        self.use_cache = cache
        self.cached = None

        self.height = image_size["height"]
        self.width = image_size["width"]
        self.channels = 1

        self.dpi = dpi
        self.resize_factor = 1.0

        self.min_resize_ratio = 0.8
        self.max_resize_ratio = 2.0

        self.whitespace_samples = 10
        self.whitespace_max_size = 300

        transform_names = transforms or []

        self._load_dataset_configs(
            dataset_config=dataset_config,
            mode=mode,
            no_sampling=no_sampling,
            no_arrays=no_arrays,
            only_basic=only_basic,
        )

        self._sample_dataset(retain_pct=retain_pct, seed=seed)

        if debug_oov and self.vocab is not None:
            self._debug_oov_tokens(
                topk=oov_topk,
                max_examples=oov_max_examples,
                report_path=oov_report_path,
            )

        self.transforms = self._build_transforms(transform_names)

        if self.use_cache:
            self.cached = [None] * len(self.formulas)
            # In cache mode, randomly select one rendered style before applying image augmentations.
            self.transforms.insert(0, self.random_selection)

    # ------------------------------------------------------------------
    # Dataset loading
    # ------------------------------------------------------------------
    def _load_dataset_configs(
        self,
        dataset_config,
        mode,
        no_sampling,
        no_arrays,
        only_basic,
    ):
        """
        Load CSV files according to the merged dataset configuration.
    
        Expected config format:
            dataset:
              path: data/im2latex-kaggle
              encoding: ISO-8859-1
              dpi: 100
              target_dpi: 600
    
              train / val / test:
                files:
                  - image_dir: formula_images_crop
                    csv: im2latexv2_train_cut.csv
        """
        dataset_root = Path(dataset_config["path"])
        encoding = dataset_config.get("encoding", "utf-8")
    
        source_dpi = dataset_config.get("dpi", 100)
        target_dpi = dataset_config.get("target_dpi", self.dpi)
        self.resize_factor = target_dpi / source_dpi
    
        self.drop_styles = dataset_config.get("drop_styles", [])
    
        split_config = dataset_config[mode]
    
        for file_config in split_config["files"]:
            image_root = dataset_root / file_config["image_dir"]
            csv_path = dataset_root / file_config["csv"]
    
            print(f"[DATA] reading csv: {csv_path}")
    
            with open(csv_path, "r", encoding=encoding) as f:
                reader = csv.reader(f)
    
                for row_idx, row in tqdm(enumerate(reader), postfix="load dataset"):
                    if not row:
                        continue
    
                    # Skip CSV header.
                    if row_idx == 0 and row[0].replace(" ", "") == "formula":
                        continue
    
                    formula = row[0]
                    image_paths = row[1:]
    
                    if no_arrays and "\\begin{array}" in formula:
                        continue
    
                    image_paths = self._filter_images_by_style(
                        image_paths=image_paths,
                        only_basic=only_basic,
                    )
    
                    if len(image_paths) == 0:
                        continue
    
                    full_image_paths = [
                        str(image_root / img_path) for img_path in image_paths
                    ]
    
                    if no_sampling:
                        # Treat each rendered style as an independent sample.
                        for img_path in full_image_paths:
                            self.formulas.append([formula, [img_path]])
                    else:
                        # Keep all rendered styles under one formula and sample one during training.
                        self.formulas.append([formula, full_image_paths])

    def _filter_images_by_style(self, image_paths, only_basic=False):
        """Filter image paths according to style settings."""
        filtered = []

        for img_path in image_paths:
            style = img_path.split("/")[0]

            if only_basic and style != "basic":
                continue

            if style in self.drop_styles:
                continue

            filtered.append(img_path)

            if style not in self.styles:
                self.styles.append(style)

        return filtered

    def _sample_dataset(self, retain_pct, seed):
        """Randomly keep a subset of samples for quick experiments."""
        if retain_pct >= 1.0 or len(self.formulas) <= 1:
            return

        original_count = len(self.formulas)
        sample_count = max(1, int(original_count * retain_pct))

        generator = torch.Generator()
        generator.manual_seed(seed)

        indices = torch.randperm(original_count, generator=generator)[:sample_count]
        self.formulas = [self.formulas[int(i)] for i in indices]

        print(
            f"[SAMPLE] total={original_count}, "
            f"kept={sample_count}, "
            f"retain_pct={retain_pct}, seed={seed}"
        )

    # ------------------------------------------------------------------
    # OOV debug
    # ------------------------------------------------------------------

    def _tokenize_for_vocab(self, formula):
        """
        Try to reuse the tokenizer from LatexVocab.
        Fall back to whitespace splitting if no tokenizer is available.
        """
        for name in ("tokenize", "text2tokens", "_tokenize"):
            fn = getattr(self.vocab, name, None)
            if callable(fn):
                try:
                    return fn(formula)
                except TypeError:
                    continue
                except Exception:
                    break

        return [token for token in formula.strip().split() if token]

    def _get_token2id(self):
        """Get the token-to-id dictionary from common LatexVocab implementations."""
        for name in ("token2id", "stoi", "word2idx"):
            token2id = getattr(self.vocab, name, None)
            if isinstance(token2id, dict):
                return token2id

        return None

    def _debug_oov_tokens(self, topk=200, max_examples=3, report_path=None):
        """Scan formulas once and report tokens that are not included in the vocabulary."""
        token2id = self._get_token2id()

        if token2id is None:
            print("[OOV] skipped: token2id dictionary was not found in vocab")
            return

        special_tokens = {"<unk>", "<pad>", "<sos>", "<eos>"}
        counter = Counter()
        examples = defaultdict(list)
        oov_sample_count = 0

        for idx, (formula, image_names) in enumerate(self.formulas):
            tokens = self._tokenize_for_vocab(formula)
            first_image = image_names[0] if image_names else ""
            has_oov = False

            for token in tokens:
                if token in special_tokens:
                    continue

                if token not in token2id:
                    counter[token] += 1
                    has_oov = True

                    if len(examples[token]) < max_examples:
                        examples[token].append(
                            {
                                "idx": idx,
                                "formula": formula,
                                "image": first_image,
                            }
                        )

            if has_oov:
                oov_sample_count += 1

        total_oov = sum(counter.values())
        unique_oov = len(counter)

        print(
            f"[OOV] unique={unique_oov}, "
            f"occurrences={total_oov}, "
            f"samples={oov_sample_count}/{len(self.formulas)}"
        )

        if unique_oov == 0:
            return

        print(f"[OOV] top {min(topk, unique_oov)} tokens:")
        for token, count in counter.most_common(topk):
            example = examples[token][0] if examples[token] else None
            if example:
                print(
                    f"  {token!r}: {count} | "
                    f"img: {example['image']} | "
                    f"formula: {example['formula']}"
                )
            else:
                print(f"  {token!r}: {count}")

        if report_path is not None:
            report = {
                "unique_oov": unique_oov,
                "total_oov_occurrences": total_oov,
                "oov_samples": oov_sample_count,
                "num_samples": len(self.formulas),
                "top": [
                    {
                        "token": token,
                        "count": count,
                        "examples": examples.get(token, []),
                    }
                    for token, count in counter.most_common(topk)
                ],
            }

            with open(report_path, "w", encoding="utf-8") as f:
                json.dump(report, f, ensure_ascii=False, indent=2)

            print(f"[OOV] report saved to: {report_path}")

    # ------------------------------------------------------------------
    # Image loading
    # ------------------------------------------------------------------

    def _formula_to_seq(self, formula):
        """Convert a LaTeX formula string to a token-id sequence."""
        if self.vocab is None:
            return np.array([], dtype=np.int64)

        return np.asarray(self.vocab.text2seq(formula, self.max_size))

    @staticmethod
    def _read_image(image_path):
        """Read an image as a grayscale float tensor in [0, 1] with shape [1, H, W]."""
        image = cv2.imread(image_path)

        if image is None:
            raise FileNotFoundError(f"Image not found or unreadable: {image_path}")

        if image.ndim == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        image = image[None, :, :]
        image = image.astype(np.float32) / 255.0
        image = np.ascontiguousarray(image)

        return torch.from_numpy(image)

    @staticmethod
    def _get_display_image_name(image_path):
        """
        Keep the original display-name behavior:
            xxx_yyy.png -> xxx.png
        """
        stem = Path(image_path).name.split("_")[0]
        return f"{stem}.png"

    def _load_all_images(self, idx):
        """Load all rendered style images for one formula. Used when cache=True."""
        formula, image_names = self.formulas[idx]

        formula_seq = self._formula_to_seq(formula)
        images = [self._read_image(image_name) for image_name in image_names]
        image_name = self._get_display_image_name(image_names[0])
        styles = [Path(image_name).name for image_name in image_names]

        return formula_seq, images, image_name, styles

    def _load_one_image(self, idx):
        """Randomly select and load one rendered style image for one formula."""
        formula, image_names = self.formulas[idx]

        formula_seq = self._formula_to_seq(formula)
        image_path = np.random.choice(image_names)

        image = self._read_image(image_path)
        image_name = self._get_display_image_name(image_names[0])

        return formula_seq, image, image_name

    def cache_all_files(self):
        """Preload all images into memory. This is useful when the dataset is small enough."""
        if not self.use_cache:
            raise RuntimeError("cache_all_files() requires cache=True")

        for idx in tqdm(range(len(self.formulas)), postfix="cache dataset"):
            if self.cached[idx] is None:
                self.cached[idx] = self._load_all_images(idx)

    # ------------------------------------------------------------------
    # PyTorch dataset API
    # ------------------------------------------------------------------

    def __len__(self):
        return len(self.formulas)

    def __getitem__(self, idx):
        if self.use_cache:
            if self.cached[idx] is None:
                self.cached[idx] = self._load_all_images(idx)

            formula_seq, image, image_name, _styles = self.cached[idx]
        else:
            formula_seq, image, image_name = self._load_one_image(idx)

        for transform in self.transforms:
            image = transform(image)

        formula_seq = torch.from_numpy(formula_seq)

        return image, ([image_name, idx], formula_seq)

    # ------------------------------------------------------------------
    # Transform building
    # ------------------------------------------------------------------

    def _build_transforms(self, transform_names):
        """Build the image transform pipeline from transform names."""
        transforms = [self.resize]

        transform_map = {
            "AddWhitespace": self.add_whitespace,
            "RandomResize": self.random_resize,
            "WhiteBorder": self.white_border,
            "DownUpResize": self.random_down_up_resize,
            "AdjustSharpness": self.adjust_sharpness,
            "White2Black": self.white2black,
        }

        for name in transform_names:
            if name in transform_map:
                transforms.append(transform_map[name])
            elif name == "GaussianBlur":
                transforms.append(
                    torchvision.transforms.GaussianBlur(
                        kernel_size=(5, 9),
                        sigma=(0.1, 5),
                    )
                )
            elif name == "ColorJitter":
                transforms.append(
                    torchvision.transforms.ColorJitter(
                        brightness=0.5,
                        contrast=0.5,
                        saturation=0.5,
                        hue=0,
                    )
                )
            else:
                raise ValueError(f"Unknown transform: {name}")

        return transforms

    # ------------------------------------------------------------------
    # Image transforms
    # ------------------------------------------------------------------

    def random_selection(self, images):
        """Randomly select one image from cached rendered styles."""
        index = int(torch.randint(0, len(images), (1,)).item())
        return images[index]

    def resize(self, image):
        """Resize the image according to the target dpi ratio."""
        _, height, width = image.shape

        new_height = max(1, int(self.resize_factor * height))
        new_width = max(1, int(self.resize_factor * width))

        return torchvision.transforms.functional.resize(
            image,
            (new_height, new_width),
            interpolation=torchvision.transforms.functional.InterpolationMode.BICUBIC,
            antialias=False,
        )

    def random_resize(self, image):
        """Randomly resize the image while keeping it inside the target canvas size."""
        _, height, width = image.shape

        if height > self.height or width > self.width:
            ratio = min(self.height / height, self.width / width)
        else:
            max_ratio = min(self.max_resize_ratio, self.height / height, self.width / width)
            ratio = self.min_resize_ratio + torch.rand(1).item() * (
                max_ratio - self.min_resize_ratio
            )

        return torchvision.transforms.functional.resize(
            image,
            (max(1, int(ratio * height)), max(1, int(ratio * width))),
            interpolation=torchvision.transforms.functional.InterpolationMode.BICUBIC,
            antialias=False,
        )

    def white_border(self, image):
        """
        Place the image on a fixed-size white canvas.
        If the image is too large, it is resized first.
        """
        canvas = torch.ones(
            (self.channels, self.height, self.width),
            dtype=image.dtype,
            device=image.device,
        )

        _, height, width = image.shape

        if width > self.width or height > self.height:
            ratio = min(self.height / height, self.width / width)

            image = torchvision.transforms.functional.resize(
                image,
                (max(1, int(ratio * height)), max(1, int(ratio * width))),
                interpolation=torchvision.transforms.functional.InterpolationMode.BICUBIC,
                antialias=False,
            )

            _, height, width = image.shape

        x_offset = 0
        y_offset = 0

        if width < self.width:
            x_offset = int(torch.randint(0, self.width - width + 1, (1,)).item())

        if height < self.height:
            y_offset = int(torch.randint(0, self.height - height + 1, (1,)).item())

        canvas[:, y_offset:y_offset + height, x_offset:x_offset + width] = image

        return canvas

    def add_whitespace(self, image):
        """
        Randomly insert white vertical strips into blank columns.

        The column-wise mean is computed over the height dimension, so each
        candidate position corresponds to one image column.
        """
        _, height, width = image.shape
        column_mean = image.mean(dim=1)

        for _ in range(self.whitespace_samples):
            col = int(torch.randint(0, width, (1,)).item())

            if torch.allclose(column_mean[:, col], torch.ones_like(column_mean[:, col])):
                insert_width = int(
                    torch.randint(1, self.whitespace_max_size + 1, (1,)).item()
                )

                white_strip = torch.ones(
                    (self.channels, height, insert_width),
                    dtype=image.dtype,
                    device=image.device,
                )

                image = torch.cat(
                    [image[:, :, :col], white_strip, image[:, :, col:]],
                    dim=-1,
                )

                width += insert_width
                column_mean = image.mean(dim=1)

        return image

    @staticmethod
    def white2black(image):
        """Invert a white-background image to a black-background image."""
        return 1.0 - image

    @staticmethod
    def adjust_sharpness(image):
        """Increase image sharpness."""
        return torchvision.transforms.functional.adjust_sharpness(image, 2)

    def random_down_up_resize(self, image):
        """
        Randomly downsample and then upsample the image.
        This simulates low-resolution degradation.
        """
        if self.dpi > 100:
            resize_factor = torch.randint(100, self.dpi, (1,)).item() / self.dpi
        else:
            resize_factor = 1.0

        _, height, width = image.shape

        downsampled = torchvision.transforms.functional.resize(
            image,
            (max(1, int(resize_factor * height)), max(1, int(resize_factor * width))),
            interpolation=torchvision.transforms.functional.InterpolationMode.BICUBIC,
            antialias=False,
        )

        return torchvision.transforms.functional.resize(
            downsampled,
            (height, width),
            interpolation=torchvision.transforms.functional.InterpolationMode.BICUBIC,
            antialias=False,
        )