"""Tests for the standalone hierarchical experiment."""

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from PIL import Image
from torchvision.transforms.functional import pil_to_tensor


EXPERIMENT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENT_DIR))

import cascade  # noqa: E402
from common import (  # noqa: E402
    HierarchicalImageFolder,
    RunMetadata,
    read_merged_classes,
    validate_hierarchy,
)


class PixelCoarseModel(torch.nn.Module):
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        means = images.float().mean(dim=(1, 2, 3))
        return torch.stack((255.0 - means, means), dim=1)


class FixedFineModel(torch.nn.Module):
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        logits = images.new_tensor([0.0, 0.0, 10.0], dtype=torch.float32)
        return logits.repeat(images.shape[0], 1)


class StandaloneExperimentTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def make_image_folder(self) -> Path:
        image_root = self.root / "images"
        for class_name in ("class_a", "class_b", "class_c", "class_d"):
            class_dir = image_root / class_name
            class_dir.mkdir(parents=True)
            Image.new("RGB", (4, 4), color="white").save(class_dir / "sample.png")
        return image_root

    def test_txt_controls_coarse_and_fine_views(self) -> None:
        txt_path = self.root / "merged.txt"
        txt_path.write_text(
            "\ufeffclass_b\nclass_c\nclass_d\n",
            encoding="utf-8",
        )
        selected = read_merged_classes(txt_path)
        image_root = self.make_image_folder()
        coarse = HierarchicalImageFolder(
            image_root,
            "coarse",
            selected,
            "special_group",
        )
        fine = HierarchicalImageFolder(
            image_root,
            "fine",
            selected,
            "special_group",
        )
        self.assertEqual(coarse.classes, ["class_a", "special_group"])
        self.assertEqual(coarse.targets, [0, 1, 1, 1])
        self.assertEqual(fine.classes, ["class_b", "class_c", "class_d"])
        self.assertEqual(fine.targets, [0, 1, 2])

    def test_only_merged_coarse_predictions_are_routed(self) -> None:
        black_path = self.root / "black.png"
        white_path = self.root / "white.png"
        Image.new("RGB", (4, 4), color="black").save(black_path)
        Image.new("RGB", (4, 4), color="white").save(white_path)
        coarse_run = RunMetadata(
            self.root,
            "unused",
            ["normal", "special_group"],
            None,
            "coarse",
            ["class_b", "class_c", "class_d"],
            "special_group",
        )
        fine_run = RunMetadata(
            self.root,
            "unused",
            ["class_b", "class_c", "class_d"],
            None,
            "fine",
            ["class_b", "class_c", "class_d"],
            "special_group",
        )
        validate_hierarchy(coarse_run, fine_run)
        with (
            patch.object(
                cascade,
                "load_trained_model",
                side_effect=[PixelCoarseModel(), FixedFineModel()],
            ),
            patch.object(
                cascade,
                "build_eval_transform",
                side_effect=[pil_to_tensor, pil_to_tensor],
            ),
        ):
            records = cascade.run_cascade(
                coarse_run,
                fine_run,
                [black_path, white_path],
                None,
                batch_size=2,
                num_workers=0,
                device=torch.device("cpu"),
                use_amp=False,
            )
        self.assertEqual(records[0]["final_class"], "normal")
        self.assertFalse(records[0]["routed_to_fine"])
        self.assertEqual(records[1]["final_class"], "class_d")
        self.assertTrue(records[1]["routed_to_fine"])


if __name__ == "__main__":
    unittest.main()
