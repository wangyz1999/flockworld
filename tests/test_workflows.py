"""CPU regression checks for command entry points and shared evaluation helpers.

Run from the repository root: python -m unittest discover -s tests -v
No recorded dataset or pretrained checkpoint is required.
"""

from __future__ import annotations

import contextlib
import csv
import importlib
import io
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch
from omegaconf import OmegaConf

from flockworld.utils import load_config
from modeling.configs import load_cfg
from modeling.eval.common import build_model, find_best_checkpoint
from modeling.eval.multi import _rollout_model, _rollout_single, write_metrics_csv
from modeling.models.decoding import build_decode_fn


ROOT = Path(__file__).resolve().parents[1]


class CommandTests(unittest.TestCase):
    def test_command_help_from_repository_root(self):
        modules = [
            "modeling.cli.train_vae",
            "modeling.cli.train_flock_dit",
            "modeling.cli.precompute_latents",
            "modeling.cli.eval_flock_dit",
            "modeling.cli.eval_flock_multi",
            "modeling.cli.eval_heading_identity",
            "modeling.cli.compare_experiments",
            "modeling.legacy.train_world_model",
            "scripts.benchmarks.benchmark_render_speed",
            "scripts.figures.plot_eval_sweep",
            "scripts.figures.gen_attention_heatmap",
            "scripts.diagnostics.vae_roundtrip",
            "scripts.diagnostics.overfit_rollout",
            "scripts.diagnostics.diagnose_identity",
            "scripts.diagnostics.diagnose_identity_breakdown",
            "scripts.analysis.probe_action_influence",
            "scripts.analysis.probe_action_sensitivity",
            "scripts.analysis.probe_command_coherence",
        ]
        for module in modules:
            with self.subTest(module=module):
                result = subprocess.run(
                    [sys.executable, "-m", module, "--help"],
                    cwd=ROOT, capture_output=True, text=True, timeout=60,
                    env={**os.environ, "PYTHONIOENCODING": "utf-8"},
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("usage:", result.stdout.lower())

    def test_recording_import_has_no_side_effects(self):
        module = importlib.import_module("flockworld.cli.data_recording")
        self.assertTrue(callable(module.main))

    def test_dit_requires_an_explicit_experiment(self):
        from modeling.cli.train_flock_dit import parse_args

        with patch.object(sys, "argv", ["train_flock_dit"]):
            with contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as error:
                    parse_args()
        self.assertEqual(error.exception.code, 2)


class ConfigTests(unittest.TestCase):
    def test_default_configs_load_outside_repository(self):
        from modeling.cli.train_vae import load_cfg as load_vae_cfg

        with tempfile.TemporaryDirectory() as directory:
            with contextlib.chdir(directory):
                self.assertEqual(load_cfg(overrides=["seed=17"]).seed, 17)
                self.assertEqual(load_vae_cfg(None, ["seed=18"]).seed, 18)
                self.assertEqual(load_config(["seed=19"]).seed, 19)

    def test_published_experiment_configs_exist(self):
        from modeling.eval.experiments import EXPERIMENTS, FLOOR_CONFIG

        for config in [FLOOR_CONFIG, *(entry[0] for entry in EXPERIMENTS.values())]:
            with self.subTest(config=config):
                cfg = load_cfg(ROOT / config)
                self.assertTrue(cfg.data.streaming.enabled)
                self.assertEqual(cfg.model.in_channels, 8)


class CheckpointTests(unittest.TestCase):
    def test_best_validation_checkpoint_takes_priority_over_step_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            torch.save({"val_loss": 0.4}, root / "epoch_001.pt")
            torch.save({"val_loss": 0.2}, root / "epoch_002.pt")
            torch.save({"val_loss": 0.3}, root / "epoch_003.pt")
            torch.save({"model": {}}, root / "step_999999.pt")
            checkpoint, loss = find_best_checkpoint(root)
            self.assertEqual(checkpoint.name, "epoch_002.pt")
            self.assertEqual(loss, 0.2)

    def test_unscored_runs_use_latest_step(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            torch.save({}, root / "epoch_001.pt")
            for step in (1, 2, 10):
                torch.save({}, root / f"step_{step:06d}.pt")
            checkpoint, loss = find_best_checkpoint(root)
            self.assertEqual(checkpoint.name, "step_000010.pt")
            self.assertTrue(math.isnan(loss))

    def test_missing_checkpoints_report_the_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FileNotFoundError) as error:
                find_best_checkpoint(Path(directory))
            self.assertIn(directory, str(error.exception))


class EvaluationTests(unittest.TestCase):
    def test_multi_agent_rollouts_preserve_context_and_produce_finite_frames(self):
        torch.manual_seed(42)
        cfg = OmegaConf.create({
            "model": {"in_channels": 2, "out_channels": 2, "dim": 32,
                      "depth": 1, "heads": 2, "ffn_dim": 64, "patch": 1},
            "data": {"num_agents": 2, "action_features": ["ax", "ay"]},
        })
        frames = torch.randn(1, 2, 3, 2, 2, 2)
        actions = torch.randn(1, 2, 3, 2)
        for tiled in (False, True):
            with self.subTest(tiled=tiled), torch.no_grad():
                cfg.model.tiled_rope = tiled
                cfg.model.broadcast_actions = tiled
                cfg.model.use_agent_embed = not tiled
                cfg.model.tile_grid = [1, 2] if tiled else None
                model = build_model(cfg).eval()
                prediction = _rollout_model(model, frames, actions, 1, 3, 1, 2)
                self.assertEqual(prediction.shape, (2, 3, 2, 2, 2))
                self.assertTrue(torch.isfinite(prediction).all())
                torch.testing.assert_close(prediction[:, :1], frames[0, :, :1])

        cfg.data.num_agents = 1
        cfg.model.tiled_rope = cfg.model.broadcast_actions = False
        cfg.model.use_agent_embed = True
        cfg.model.tile_grid = None
        with torch.no_grad():
            prediction = _rollout_single(build_model(cfg).eval(), frames, actions, 1, 3, 1, 2)
        self.assertEqual(prediction.shape, (2, 3, 2, 2, 2))
        torch.testing.assert_close(prediction[:, :1], frames[0, :, :1])

    def test_cached_latent_decoder_restores_channel_statistics_and_axes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cache = root / "latent_cache"
            cache.mkdir()
            torch.save({"mean": torch.tensor([2.0, 3.0]),
                        "std": torch.tensor([4.0, 5.0])}, cache / "stats.pt")
            cfg = OmegaConf.create({"device": "cpu", "data": {"root": directory},
                                    "vae": {"checkpoint_path": "synthetic.ckpt"}})
            with patch("modeling.models.frozen_vae.FrozenVAE") as vae:
                vae.return_value.decode.side_effect = lambda value: value
                decode = build_decode_fn(cfg)
                actual = decode(torch.ones(3, 2, 2, 2))
                expected = torch.tensor([6.0, 8.0]).view(1, 2, 1, 1).expand(3, 2, 2, 2)
                torch.testing.assert_close(actual, expected)
                self.assertEqual(vae.return_value.decode.call_args.args[0].shape, (1, 2, 3, 2, 2))

    def test_metrics_csv_preserves_episode_identity(self):
        results = {"episode_ids": ["ep-a", "ep-b"], "per_episode": {
            "model": {"tier_a": [{"detection_rate": 0.5}, {"detection_rate": 0.75}]},
        }}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "nested" / "metrics.csv"
            with contextlib.redirect_stdout(io.StringIO()):
                write_metrics_csv(path, "baseline", results)
            with path.open(newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual([row["episode_id"] for row in rows], ["ep-a", "ep-b"])
            self.assertEqual([float(row["value"]) for row in rows], [0.5, 0.75])
            self.assertTrue(all(row["experiment"] == "baseline" for row in rows))


if __name__ == "__main__":
    unittest.main()
