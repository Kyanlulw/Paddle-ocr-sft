"""
Dataset classes for PaddleOCR-VL training.

Supports:
- Manga109s OCR training with optional synthetic data mixing
- mixed30k VQA-style training from two image sources
"""

import json
import os
from pathlib import Path
from typing import Optional

import pandas as pd
from dotenv import load_dotenv
from PIL import Image
from torch.utils.data import Dataset


load_dotenv()

MANGA109_ROOT = Path(os.getenv("MANGA109_ROOT", "")).expanduser()
DATA_SYNTHETIC_ROOT = Path(os.getenv("DATA_SYNTHETIC_ROOT", "")).expanduser()

OCR_PROMPT = "OCR:"
MIXED30K_SOURCE_SPLIT_LINE = 15000


def _build_messages(prompt: str, answer: str, image: Image.Image) -> list[dict]:
    """Build chat-format messages for a single vision-language sample."""
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        },
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": answer},
            ],
        },
    ]


class MangaDataset(Dataset):
    """
    Dataset for Manga109s OCR training.
    
    Returns image-text pairs in the format expected by VL model processors.
    Preprocessing (tokenization, padding) is handled by the data collator.
    
    Args:
        split: Dataset split ('train' or 'test')
        limit_size: Limit dataset size (useful for evaluation)
        augment: Enable data augmentation (default False)
        skip_packages: Set of synthetic data package IDs to skip
        use_synthetic: Include synthetic data in training (default True)
    """

    def __init__(
        self,
        split: str,
        limit_size=None,
        augment: bool = False,
        skip_packages=None,
        use_synthetic: bool = True,
    ):
        data = []

        print(f"Initializing dataset {split}...")

        if skip_packages is None:
            skip_packages = set()
        else:
            skip_packages = {f"{x:04d}" for x in skip_packages}

        # Load synthetic data if available and enabled
        if use_synthetic and DATA_SYNTHETIC_ROOT.exists():
            meta_dir = DATA_SYNTHETIC_ROOT / "meta"
            if meta_dir.exists():
                for path in sorted(meta_dir.glob("*.csv")):
                    if path.stem in skip_packages:
                        continue
                    if not (DATA_SYNTHETIC_ROOT / "img" / path.stem).is_dir():
                        continue
                    df = pd.read_csv(path)
                    df = df.dropna()
                    df["path"] = df.id.apply(
                        lambda x, stem=path.stem: str(
                            DATA_SYNTHETIC_ROOT / "img" / stem / f"{x}.jpg"
                        )
                    )
                    df = df[["path", "text"]]
                    data.append(df)

        # Load Manga109 data
        data_csv = MANGA109_ROOT / "data.csv"
        if data_csv.exists():
            df = pd.read_csv(data_csv)
            df = df[df.split == split].reset_index(drop=True)
            df["path"] = df.crop_path.apply(lambda x: str(MANGA109_ROOT / x))
            df = df[["path", "text"]]
            data.append(df)
        else:
            raise FileNotFoundError(f"Dataset not found: {data_csv}")

        data = pd.concat(data, ignore_index=True)

        if limit_size:
            data = data.iloc[:limit_size]
        
        self.data = data
        self.augment = augment

        print(f"Dataset {split}: {len(self.data)} samples")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        Return image and messages for VL model training.

        Returns:
            dict with 'images' (list of PIL Images) and 'messages' (chat format)
        """
        sample = self.data.loc[idx]
        text = sample.text
        image = Image.open(sample.path).convert("RGB")

        return {
            "images": [image],
            "messages": _build_messages(OCR_PROMPT, text, image),
        }


class Mixed30kDataset(Dataset):
    """
    Dataset for mixed30k VQA fine-tuning with two image sources.

    The annotation file is expected to be JSON Lines, one object per row:
    {
        "image": "000000004169.jpg",
        "question": "...",
        "answer": "...",
        "_source": "dataset2"
    }

    Image routing is driven by `_source`:
    - dataset1 -> dataset1_image_root / image
    - dataset2 -> dataset2_image_root / image

    For dataset2, OpenViVQA metadata is optional and used only to validate or
    recover a filename from a numeric image ID representation.
    """

    def __init__(
        self,
        annotation_path: str,
        dataset1_image_root: str,
        dataset2_image_root: str,
        dataset2_metadata_path: Optional[str] = None,
        limit_size=None,
        validate_source_split: bool = True,
    ):
        self.annotation_path = Path(annotation_path).expanduser()
        if not self.annotation_path.exists():
            raise FileNotFoundError(
                f"mixed30k annotation file not found: {self.annotation_path}"
            )

        self.dataset_roots = {
            "dataset1": self._require_directory(
                dataset1_image_root, "dataset1_image_root"
            ),
            "dataset2": self._require_directory(
                dataset2_image_root, "dataset2_image_root"
            ),
        }
        self.dataset2_metadata = self._load_dataset2_metadata(dataset2_metadata_path)
        self.samples = self._load_samples(limit_size, validate_source_split)

        print(f"Dataset mixed30k: {len(self.samples)} samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        image = Image.open(sample["path"]).convert("RGB")

        return {
            "images": [image],
            "messages": _build_messages(sample["question"], sample["answer"], image),
        }

    def _load_samples(
        self, limit_size=None, validate_source_split: bool = True
    ) -> list[dict]:
        samples = []

        with self.annotation_path.open("r", encoding="utf-8") as handle:
            for line_number, raw_line in enumerate(handle, start=1):
                if limit_size and len(samples) >= limit_size:
                    break

                line = raw_line.strip()
                if not line:
                    continue

                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Malformed JSON on line {line_number} in "
                        f"{self.annotation_path}: {exc.msg}"
                    ) from exc

                samples.append(
                    self._build_sample(record, line_number, validate_source_split)
                )

        if not samples:
            raise ValueError(
                f"No usable samples found in mixed30k annotation file: "
                f"{self.annotation_path}"
            )

        return samples

    def _build_sample(
        self, record: dict, line_number: int, validate_source_split: bool
    ) -> dict:
        source = record.get("_source")
        if source not in self.dataset_roots:
            raise ValueError(
                f"Line {line_number} has invalid or missing _source: {source!r}. "
                "Expected 'dataset1' or 'dataset2'."
            )

        if validate_source_split:
            expected_source = (
                "dataset1"
                if line_number <= MIXED30K_SOURCE_SPLIT_LINE
                else "dataset2"
            )
            if source != expected_source:
                raise ValueError(
                    f"Line {line_number} has _source={source!r}, expected "
                    f"{expected_source!r} based on the mixed30k split boundary at "
                    f"line {MIXED30K_SOURCE_SPLIT_LINE}."
                )

        question = self._require_text_field(record, "question", line_number)
        answer = self._require_text_field(record, "answer", line_number)
        image_name = self._resolve_image_name(record, source, line_number)
        image_path = self.dataset_roots[source] / image_name

        if not image_path.is_file():
            raise FileNotFoundError(
                f"Image file not found for line {line_number}: {image_path}"
            )

        return {
            "path": image_path,
            "question": question,
            "answer": answer,
            "source": source,
            "line_number": line_number,
        }

    def _resolve_image_name(self, record: dict, source: str, line_number: int) -> str:
        raw_image = record.get("image")
        image_value = raw_image.strip() if isinstance(raw_image, str) else raw_image

        if source == "dataset1":
            if not isinstance(image_value, str) or not image_value:
                raise ValueError(
                    f"Line {line_number} is missing a usable image filename for "
                    "dataset1."
                )
            return image_value

        metadata_filename = self._lookup_dataset2_filename(image_value)
        if metadata_filename:
            if isinstance(image_value, str) and image_value:
                candidate_name = Path(image_value).name
                if Path(candidate_name).suffix and candidate_name != metadata_filename:
                    raise ValueError(
                        f"Line {line_number} has dataset2 image filename "
                        f"{candidate_name!r}, but metadata resolves it to "
                        f"{metadata_filename!r}."
                    )
            return metadata_filename

        if isinstance(image_value, str) and image_value:
            candidate_name = Path(image_value).name
            if Path(candidate_name).suffix:
                return candidate_name
            raise ValueError(
                f"Line {line_number} has dataset2 image value {image_value!r}, "
                "but it is not a filename and could not be resolved through "
                "dataset2 metadata."
            )

        raise ValueError(
            f"Line {line_number} is missing a usable image filename for dataset2."
        )

    def _lookup_dataset2_filename(self, image_value) -> Optional[str]:
        if not self.dataset2_metadata:
            return None

        candidate_ids = []
        if isinstance(image_value, int):
            candidate_ids.append(image_value)
        elif isinstance(image_value, str) and image_value:
            for candidate in (image_value, Path(image_value).stem):
                if candidate.isdigit():
                    candidate_ids.append(int(candidate))

        for image_id in candidate_ids:
            filename = self.dataset2_metadata.get(image_id)
            if filename:
                return filename

        return None

    def _load_dataset2_metadata(
        self, dataset2_metadata_path: Optional[str]
    ) -> dict[int, str]:
        if not dataset2_metadata_path:
            return {}

        metadata_path = Path(dataset2_metadata_path).expanduser()
        if not metadata_path.exists():
            raise FileNotFoundError(
                f"dataset2 metadata file not found: {metadata_path}"
            )

        with metadata_path.open("r", encoding="utf-8") as handle:
            metadata = json.load(handle)

        images = metadata.get("images")
        if not isinstance(images, list):
            raise ValueError(
                f"dataset2 metadata file must contain an 'images' list: "
                f"{metadata_path}"
            )

        image_map = {}
        for entry in images:
            if not isinstance(entry, dict):
                continue
            image_id = entry.get("id")
            filename = entry.get("filename")
            if isinstance(image_id, int) and isinstance(filename, str) and filename:
                image_map[image_id] = filename

        return image_map

    def _require_directory(self, path_value: str, field_name: str) -> Path:
        if not path_value:
            raise ValueError(
                f"{field_name} is required when dataset_backend='mixed30k'."
            )

        directory = Path(path_value).expanduser()
        if not directory.is_dir():
            raise FileNotFoundError(f"{field_name} is not a directory: {directory}")

        return directory

    def _require_text_field(self, record: dict, field_name: str, line_number: int) -> str:
        value = record.get(field_name)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(
                f"Line {line_number} is missing a usable {field_name!r} value."
            )
        return value.strip()


if __name__ == "__main__":
    # Quick test
    ds = MangaDataset("train", limit_size=5)
    for i in range(min(5, len(ds))):
        sample = ds[i]
        print(f"Sample {i}: {sample['images'][0].size}, text: {sample['messages'][1]['content'][0]['text'][:50]}...")
