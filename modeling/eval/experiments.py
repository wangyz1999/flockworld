"""Published streaming experiment configurations and checkpoint locations."""

RUN_ROOT = "flockdit_streaming_20260715"

# run-key -> (config, output_dir); output_dir holds checkpoints/epoch_*.pt.
EXPERIMENTS = {
    "exp01_baseline": (
        "config/train_flockdit_latent_multi_stream.yaml",
        f"{RUN_ROOT}/exp01_baseline-10290187_0",
    ),
    "exp02_tiled": (
        "config/train_flockdit_latent_multi_stream_tiled.yaml",
        f"{RUN_ROOT}/exp02_tiled-10290187_1",
    ),
    "exp03_diffusion_forcing": (
        "config/train_flockdit_latent_multi_stream_df.yaml",
        f"{RUN_ROOT}/exp03_diffusion_forcing-10290187_2",
    ),
    "exp06_tiled_df": (
        "config/train_flockdit_latent_multi_stream_tiled_df.yaml",
        f"{RUN_ROOT}/exp06_tiled_df-10290187_4",
    ),
    # Two-stage runs: single-agent pretrain -> multi-agent fine-tune. The output_dir
    # holds the STAGE-2 checkpoints; stage1/ alongside it holds the pretrain (which is
    # also what FLOOR_OUTPUT_DIR below points at).
    "exp04_two_stage": (
        "config/train_flockdit_latent_multi_stream_twostage.yaml",
        f"{RUN_ROOT}/exp04_two_stage-10621962_3",
    ),
    "exp07_tiled_df_two_stage": (
        "config/train_flockdit_latent_multi_stream_triple.yaml",
        f"{RUN_ROOT}/exp07_tiled_df_two_stage-10621962_5",
    ),
    "exp08_tiled_two_stage": (
        "config/train_flockdit_latent_multi_stream_twostage_tiled.yaml",
        f"{RUN_ROOT}/exp08_tiled_two_stage-10621962_6",
    ),
}

# The floor: an independent single-agent model, rolled out per-camera-agent on the
# SAME episodes (in-distribution -- the "why not just run 10 single-agent models" bar).
# exp04's stage-1 pretrain is the plain single-agent config (exp07's stage 1 adds
# diffusion forcing, so it is NOT interchangeable here). Same config and same epoch 84
# as the pre-re-run floor, so this column stays comparable to the 2026-07-29 table.
FLOOR_CONFIG = "config/train_flockdit_latent_single_stream.yaml"
FLOOR_OUTPUT_DIR = f"{RUN_ROOT}/exp04_two_stage-10621962_3/stage1"
