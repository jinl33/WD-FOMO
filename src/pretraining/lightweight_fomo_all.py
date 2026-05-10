import numpy as np
import torch
import os
import gc
import argparse
import pickle
from copy import deepcopy
from collections import Counter

import resource

rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import monai
from monai.utils import set_determinism
from monai.engines import SupervisedTrainer
from monai.handlers import MeanSquaredError, from_engine

from generative.networks.nets import DiffusionModelUNet
from generative.networks.schedulers import DDPMScheduler, DDIMScheduler

import ignite
from ignite.contrib.handlers import ProgressBar

from torch.utils.tensorboard import SummaryWriter

import utils
import random
from tqdm import tqdm
import pywt
import glob

torch.multiprocessing.set_sharing_strategy("file_system")

torch.manual_seed(42)
set_determinism(42)
random.seed(42)
np.random.seed(42)
torch.set_float32_matmul_precision("high")


parser = argparse.ArgumentParser(description="ALL-Data FOMO Diffusion Model Training")
parser.add_argument(
    "--data_path", type=str, required=True, help="Path to the preprocessed dataset"
)
parser.add_argument(
    "--split_json",
    type=str,
    default=None,
    help="Path to canonical split JSON (fomo60k_split.json). When provided, uses pretrain list directly to avoid distillation data leakage.",
)
parser.add_argument("--batch_size", type=int, default=2, help="Batch size for training")
parser.add_argument(
    "--num_workers", type=int, default=6, help="Number of data loader workers"
)
parser.add_argument(
    "--train_epochs", type=int, default=100, help="Number of training epochs"
)
parser.add_argument(
    "--train_batches_per_epoch",
    type=int,
    default=None,
    help="Cap the number of training batches per epoch. Uses the full loader when unset.",
)
parser.add_argument(
    "--model_prefix", type=str, default="fomo_all", help="Prefix for model files"
)
parser.add_argument(
    "--checkpoint_dir",
    type=str,
    default="./checkpoints",
    help="Directory to save checkpoints",
)
parser.add_argument(
    "--logs_dir", type=str, default="./logs", help="Directory to save logs"
)
parser.add_argument(
    "--resume_epoch", type=int, default=None, help="Specific epoch to resume from"
)
parser.add_argument(
    "--ema_decay", type=float, default=0.0, help="EMA decay (0 to disable, e.g., 0.999)"
)
parser.add_argument(
    "--ema_start_epoch", type=int, default=0, help="Epoch to start EMA (e.g., 50)"
)
parser.add_argument(
    "--use_gpu_wavelet",
    action="store_true",
    help="Use GPU-accelerated wavelet transform",
)
parser.add_argument(
    "--model_scale",
    type=str,
    default="paper_232m",
    choices=["paper_232m", "s58", "s33", "s23", "s15", "custom"],
    help="Architecture scale profile. s33/s23/s15 are efficient reductions for A40 sweeps.",
)
parser.add_argument(
    "--custom_num_channels",
    type=int,
    nargs="+",
    default=None,
    help="Custom num_channels for model_scale=custom (e.g., --custom_num_channels 48 48 96 96 192).",
)
parser.add_argument(
    "--num_res_blocks", type=int, default=2, help="Number of residual blocks per stage"
)
parser.add_argument(
    "--attention_mode",
    type=str,
    default="paper",
    choices=["paper", "last2", "last1", "none"],
    help="Attention placement strategy. paper keeps original 5-level setting.",
)
parser.add_argument(
    "--attn_head_channels",
    type=int,
    default=16,
    help="Head channel size for attention-enabled stages in reduced-scale profiles.",
)
parser.add_argument(
    "--skip_preprocessing_check",
    action="store_true",
    help="Skip .pkl metadata validation (default validates 1mm + crop_to_nonzero consistency).",
)
parser.add_argument(
    "--preprocessing_check_samples",
    type=int,
    default=2048,
    help="Number of .pkl files to sample for preprocessing validation (<=0 checks all).",
)
args = parser.parse_args()

lr = 1e-5
channels = 64
levels = 2
num_train_timesteps = 1000
num_workers = min(args.num_workers, os.cpu_count() // 2)
pin_memory = torch.cuda.is_available() if num_workers > 0 else False
batch_size_train = args.batch_size
batch_size_test = batch_size_train
train_epochs = args.train_epochs
train_batches_per_epoch = args.train_batches_per_epoch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

initial_size = [192, 256, 192]
target_size = [d // (2**levels) for d in initial_size]
print(f"Wavelet transform: {initial_size} -> {target_size} ({channels} channels)")
print(f"Using device: {device}")
print(f"Data Policy: LOAD EVERYTHING (*.npy)")

gpu_dwt = None
GPUWaveletPrepareBatch = None
if args.use_gpu_wavelet:
    if not torch.cuda.is_available():
        print(
            "WARNING: --use_gpu_wavelet specified but CUDA not available. Falling back to CPU pywt."
        )
    else:
        try:
            from gpu_wavelet import HaarDWT3D_2Level, GPUWaveletPrepareBatch

            gpu_dwt = HaarDWT3D_2Level(normalize_factor=8.0).to(device)
            print("✓ GPU wavelet transform initialized")
            print("✓ GPUWaveletPrepareBatch loaded")
        except ImportError as e:
            print(f"WARNING: gpu_wavelet.py import failed: {e}")
            print("  Falling back to CPU pywt.")

model_prefix = args.model_prefix
checkpoint_dir = os.path.join(args.checkpoint_dir, model_prefix)
logs_dir = os.path.join(args.logs_dir, model_prefix)
tensorboard_dir = os.path.join(logs_dir, "tensorboard")
os.makedirs(checkpoint_dir, exist_ok=True)
os.makedirs(logs_dir, exist_ok=True)
os.makedirs(tensorboard_dir, exist_ok=True)


def pywt_decompose_cpu(x):
    packet = pywt.WaveletPacketND(
        x.get_array(), "haar", axes=(-3, -2, -1), maxlevel=levels
    )
    x.set_array(
        torch.tensor(
            np.concatenate([y.data.view() for y in packet.get_level(levels)], axis=0)
            / 8.0
        )
    )
    gc.collect()
    return x


def identity_transform(x):
    img_np = x.get_array()
    x.set_array(torch.from_numpy(img_np).float())
    return x


def robust_scale_intensity(x):
    """
    Robust percentile-based scaling that handles flat/corrupted images.
    Safely handles edge cases without losing training data.
    CRITICAL: Must never produce NaN or Inf values.
    """
    img = x.get_array().astype(np.float32)

    if not np.isfinite(img).all():
        print(f"WARNING: Input has NaN/Inf values. Replacing with small random noise.")
        img = np.random.normal(0, 0.01, img.shape).astype(np.float32)
    else:
        p1 = np.percentile(img, 1)
        p99 = np.percentile(img, 99)

        if np.abs(p99 - p1) < 1e-6:
            img = np.random.normal(0, 0.01, img.shape).astype(np.float32)
        else:
            denominator = p99 - p1
            if np.abs(denominator) < 1e-10:
                img = np.random.normal(0, 0.01, img.shape).astype(np.float32)
            else:
                img = (img - p1) / denominator  # [0, 1]
                img = np.clip(img, 0, 1)
                img = img * 2.0 - 1.0  # [-1, 1]

    if not np.isfinite(img).all():
        print(
            f"FATAL: robust_scale_intensity produced NaN/Inf! Replacing entire array."
        )
        img = np.random.normal(0, 0.01, img.shape).astype(np.float32)

    x.set_array(torch.from_numpy(img).float())
    return x


lambd_pywt = monai.transforms.Lambdad(keys=["image"], func=pywt_decompose_cpu)
lambd_identity = monai.transforms.Lambdad(keys=["image"], func=identity_transform)
lambd_robust_scale = monai.transforms.Lambdad(
    keys=["image"], func=robust_scale_intensity
)


class NumpyLoader(monai.data.ImageReader):
    """Load .npy files with proper error handling"""

    def read(self, data, **kwargs):
        img = np.load(data).astype(np.float32)
        return img, {"spatial_shape": img.shape}

    def get_data(self, img):
        return img[0], img[1]

    def verify_suffix(self, filename):
        return filename.endswith(".npy")


wavelet_transform = lambd_identity if gpu_dwt is not None else lambd_pywt

train_transforms = monai.transforms.Compose(
    [
        monai.transforms.LoadImaged(
            keys=["image"], reader=NumpyLoader(), ensure_channel_first=True
        ),
        monai.transforms.ResizeWithPadOrCropd(
            keys=["image"], spatial_size=initial_size
        ),
        lambd_robust_scale,
        wavelet_transform,
        monai.transforms.ToTensord(keys=["image"], track_meta=False),
    ],
    lazy=False,
)

test_transforms = monai.transforms.Compose(
    [
        monai.transforms.LoadImaged(
            keys=["image"], reader=NumpyLoader(), ensure_channel_first=True
        ),
        monai.transforms.ResizeWithPadOrCropd(
            keys=["image"], spatial_size=initial_size
        ),
        lambd_robust_scale,
        wavelet_transform,
        monai.transforms.ToTensord(keys=["image"], track_meta=False),
    ]
)


def _is_spacing_1mm(spacing, tol=1e-4):
    if spacing is None:
        return False
    try:
        spacing_arr = np.asarray(spacing, dtype=float).reshape(-1)
    except Exception:
        return False
    if spacing_arr.size < 3:
        return False
    return bool(np.all(np.abs(spacing_arr[:3] - 1.0) <= tol))


def verify_preprocessed_metadata(data_path, sample_count=2048, seed=42):
    """Validate that input files are Yucca-preprocessed with 1mm spacing + crop metadata."""
    pkl_files = glob.glob(os.path.join(data_path, "*.pkl"))
    if not pkl_files:
        raise RuntimeError(
            f"No .pkl files found in {data_path}. Expected Yucca-preprocessed .npy/.pkl pairs."
        )

    if sample_count is not None and sample_count > 0 and len(pkl_files) > sample_count:
        rng = random.Random(seed)
        pkl_files = rng.sample(pkl_files, sample_count)

    missing_pairs = []
    invalid_meta = []
    non_iso = []
    missing_crop = []

    for pkl_path in pkl_files:
        npy_path = pkl_path[:-4] + ".npy"
        if not os.path.exists(npy_path):
            missing_pairs.append(os.path.basename(pkl_path))
            continue

        try:
            with open(pkl_path, "rb") as f:
                meta = pickle.load(f)
        except Exception:
            invalid_meta.append(os.path.basename(pkl_path))
            continue

        if not _is_spacing_1mm(meta.get("new_spacing")):
            non_iso.append(os.path.basename(pkl_path))
        if "crop_to_nonzero" not in meta:
            missing_crop.append(os.path.basename(pkl_path))

    if missing_pairs or invalid_meta or non_iso or missing_crop:
        raise RuntimeError(
            "Preprocessing validation failed. "
            f"checked={len(pkl_files)} "
            f"missing_pairs={len(missing_pairs)} "
            f"invalid_meta={len(invalid_meta)} "
            f"non_1mm={len(non_iso)} "
            f"missing_crop_to_nonzero={len(missing_crop)}\n"
            f"examples_missing_pairs={missing_pairs[:3]}\n"
            f"examples_invalid_meta={invalid_meta[:3]}\n"
            f"examples_non_1mm={non_iso[:3]}\n"
            f"examples_missing_crop={missing_crop[:3]}"
        )

    print(
        f"Preprocessing check passed: checked {len(pkl_files)} .pkl files, "
        "all have matching .npy, new_spacing≈[1,1,1], crop_to_nonzero metadata present."
    )


def get_everything_data(data_path, split_ratios=(0.8, 0.2), split_json=None):
    import json as _json

    if split_json is not None:
        print(f"Loading canonical split from: {split_json}")
        with open(split_json) as f:
            split = _json.load(f)
        pretrain_files = split["pretrain"]  # 41,483 files
        distil_files = set(split["distillation"])  # 5,000 files (excluded)
        print(
            f"  Canonical pretrain: {len(pretrain_files)}, distillation (excluded): {len(distil_files)}"
        )
        verified = []
        missing = 0
        for f in pretrain_files:
            candidate = os.path.join(data_path, os.path.basename(f))
            if os.path.exists(candidate):
                verified.append(candidate)
            elif os.path.exists(f):
                verified.append(f)
            else:
                missing += 1
        if missing:
            print(f"  WARNING: {missing} pretrain files not found on disk (skipped)")
        pretrain_files = verified
        random.Random(42).shuffle(pretrain_files)
        n_train = int(len(pretrain_files) * split_ratios[0])
        train_files = pretrain_files[:n_train]
        val_files = pretrain_files[n_train:]
        print(f"  Training: {len(train_files)}, Validation: {len(val_files)}")
        print(f"  Distillation set EXCLUDED from training (no leakage)")
        return train_files, val_files

    print(f"WARNING: No --split_json provided. Using random 80/20 split of all files.")
    print(f"  This may leak distillation files into training!")
    print(f"Scanning data_path: {data_path}")
    pattern = os.path.join(data_path, "*.npy")
    all_files = glob.glob(pattern)
    total_found = len(all_files)

    print(f"Total files found (*.npy): {total_found}")
    if total_found == 0:
        raise ValueError("No data found! Check path or filename pattern.")

    print("\n" + "=" * 40)
    print("DATA FREQUENCY TABLE")
    print("=" * 40)
    type_counts = Counter()
    for f in all_files:
        filename = os.path.basename(f)
        try:
            name_no_suffix = filename.replace(".npy", "")
            parts = name_no_suffix.split("_")
            if "skull_stripped" in name_no_suffix:
                if "skull" in parts:
                    idx = parts.index("skull")
                    mod_type = parts[idx - 1] if idx > 0 else "unknown"
                else:
                    mod_type = "unknown"
            else:
                mod_type = parts[-1] if len(parts) > 1 else "unknown"
            type_counts[mod_type] += 1
        except Exception:
            type_counts["error_parsing"] += 1

    print(f"{'TYPE':<20} | {'COUNT':<10} | {'%':<10}")
    print("-" * 46)
    for mod_type, count in sorted(type_counts.items()):
        percent = (count / total_found) * 100
        print(f"{mod_type:<20} | {count:<10} | {percent:<6.1f}%")
    print("-" * 46)
    print(f"{'TOTAL':<20} | {total_found:<10} | 100.0%")
    print("=" * 40 + "\n")

    print("Skipping file verification (assuming files are valid)...")
    valid_files = all_files
    random.Random(42).shuffle(valid_files)

    n_train = int(len(valid_files) * split_ratios[0])
    train_files = valid_files[:n_train]
    val_files = valid_files[n_train:]

    print(f"Training: {len(train_files)}, Validation: {len(val_files)}")
    return train_files, val_files


def _build_attention_levels(num_levels, mode):
    if mode == "none":
        return [False] * num_levels
    if mode == "last1":
        return [False] * (num_levels - 1) + [True]
    if mode == "last2":
        if num_levels < 2:
            return [True]
        return [False] * (num_levels - 2) + [True, True]
    if num_levels == 5:
        return [False, False, False, True, True]
    if num_levels < 2:
        return [True]
    return [False] * (num_levels - 2) + [True, True]


def _build_head_channels(attention_levels, head_ch):
    return [head_ch if use_attn else 0 for use_attn in attention_levels]


def _resolve_model_channels():
    scale_map = {
        "paper_232m": [128, 128, 256, 256, 512],
        "s58": [64, 64, 128, 128, 256],
        "s33": [48, 48, 96, 96, 192],
        "s23": [40, 40, 80, 80, 160],
        "s15": [32, 32, 64, 64, 128],
    }

    if args.model_scale == "custom":
        if not args.custom_num_channels:
            raise ValueError("model_scale=custom requires --custom_num_channels")
        return args.custom_num_channels
    return scale_map[args.model_scale]


def _resolve_norm_num_groups(num_channels_cfg):
    for groups in (32, 16, 8, 4, 2, 1):
        if all(ch % groups == 0 for ch in num_channels_cfg):
            return groups
    raise ValueError(f"No valid norm_num_groups found for channels: {num_channels_cfg}")


def _build_model_config():
    num_channels_cfg = _resolve_model_channels()
    attn_levels = _build_attention_levels(len(num_channels_cfg), args.attention_mode)
    norm_num_groups = _resolve_norm_num_groups(num_channels_cfg)

    if args.model_scale == "paper_232m" and args.attention_mode == "paper":
        head_channels = [0, 0, 0, 32, 32]
    else:
        head_channels = _build_head_channels(attn_levels, args.attn_head_channels)

    return {
        "spatial_dims": 3,
        "in_channels": channels,
        "out_channels": channels,
        "num_channels": num_channels_cfg,
        "norm_num_groups": norm_num_groups,
        "attention_levels": attn_levels,
        "num_head_channels": head_channels,
        "num_res_blocks": args.num_res_blocks,
        "use_flash_attention": False,
        "with_conditioning": False,
    }


MODEL_CONFIG = _build_model_config()


def report_model_efficiency(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    fp32_mib = total_params * 4 / (1024**2)
    print("Model efficiency summary:")
    print(f"  Scale profile: {args.model_scale}")
    print(f"  Num channels: {MODEL_CONFIG['num_channels']}")
    print(f"  Norm groups: {MODEL_CONFIG['norm_num_groups']}")
    print(f"  Attention levels: {MODEL_CONFIG['attention_levels']}")
    print(f"  Head channels: {MODEL_CONFIG['num_head_channels']}")
    print(f"  Residual blocks/stage: {MODEL_CONFIG['num_res_blocks']}")
    print(f"  Parameters: {total_params:,} ({total_params / 1e6:.2f}M)")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Parameter memory (fp32): {fp32_mib:.2f} MiB")


def get_model():
    return DiffusionModelUNet(**MODEL_CONFIG)


def main():
    print(f"Foundational Model Training - ALL AVAILABLE DATA")

    if not args.skip_preprocessing_check:
        verify_preprocessed_metadata(
            args.data_path,
            sample_count=args.preprocessing_check_samples,
        )

    train_files, val_files = get_everything_data(
        args.data_path, split_json=args.split_json
    )

    train_ds = monai.data.Dataset(
        data=[{"image": f} for f in train_files], transform=train_transforms
    )
    val_ds = monai.data.Dataset(
        data=[{"image": f} for f in val_files], transform=test_transforms
    )

    train_loader = monai.data.DataLoader(
        train_ds,
        batch_size=batch_size_train,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=True,
    )
    val_loader = monai.data.DataLoader(
        val_ds,
        batch_size=batch_size_test,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=True,
    )

    model = get_model().to(device)
    report_model_efficiency(model)
    opt = torch.optim.AdamW(params=model.parameters(), lr=lr)

    epoch_length = (
        len(train_loader)
        if train_batches_per_epoch is None
        else min(len(train_loader), train_batches_per_epoch)
    )
    if train_batches_per_epoch is None:
        print(f"Using full epoch length: {epoch_length} batches per epoch")
    else:
        print(
            f"Using capped epoch length: {epoch_length} batches per epoch (requested cap: {train_batches_per_epoch})"
        )

    resumed_epoch = 0
    if args.resume_epoch is not None:
        ckpt_path = os.path.join(
            checkpoint_dir, f"{model_prefix}_epoch_{args.resume_epoch}.pt"
        )
        if os.path.exists(ckpt_path):
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            model.load_state_dict(ckpt["model_state_dict"])
            opt.load_state_dict(ckpt["optimizer_state_dict"])
            resumed_epoch = ckpt["epoch"]
            print(f"Resumed from epoch {resumed_epoch}")

    scheduler = DDPMScheduler(
        num_train_timesteps=num_train_timesteps,
        schedule="linear_beta",
        prediction_type="sample",
    )
    inferer = utils.UnconditionalDiffusionInferer(scheduler)

    if gpu_dwt is not None and GPUWaveletPrepareBatch is not None:
        prepare_batch_fn = GPUWaveletPrepareBatch(
            num_train_timesteps, gpu_wavelet_fn=gpu_dwt
        )
    else:
        prepare_batch_fn = utils.SamplePredictionPrepareBatch(num_train_timesteps)

    trainer = SupervisedTrainer(
        device=device,
        max_epochs=train_epochs,
        epoch_length=epoch_length,
        train_data_loader=train_loader,
        network=model,
        optimizer=opt,
        loss_function=torch.nn.MSELoss(),
        inferer=inferer,
        prepare_batch=prepare_batch_fn,
        key_train_metric={
            "train_mse": MeanSquaredError(
                output_transform=from_engine(["pred", "label"])
            )
        },
        amp=True,
    )

    writer = SummaryWriter(log_dir=tensorboard_dir)

    if args.resume_epoch is not None:
        trainer.state.epoch = args.resume_epoch

    @trainer.on(ignite.engine.Events.STARTED)
    def set_resumed_epoch(engine):
        if resumed_epoch > 0:
            engine.state.epoch = resumed_epoch

    ignite.metrics.RunningAverage(
        output_transform=from_engine(["loss"], first=True)
    ).attach(trainer, "avg_loss")
    ProgressBar(persist=True).attach(trainer, ["avg_loss"])

    @trainer.on(ignite.engine.Events.ITERATION_STARTED)
    def on_iteration_start(engine):
        if engine.state.iteration == 1:
            print(f"Training started: {epoch_length} batches per epoch")

    @trainer.on(ignite.engine.Events.ITERATION_COMPLETED(every=100))
    def log_iteration_detail(engine):
        avg_loss = engine.state.metrics.get("avg_loss", None)
        iteration = engine.state.iteration
        if avg_loss is not None:
            print(f"  Iteration {iteration}: avg_loss={avg_loss:.6f}")
        else:
            print(f"  Iteration {iteration}: avg_loss=None")

    @trainer.on(ignite.engine.Events.ITERATION_COMPLETED)
    def check_nan_during_training(engine):
        """Detect NaN/Inf in loss immediately to fail fast"""
        avg_loss = engine.state.metrics.get("avg_loss", None)
        if avg_loss is not None and not torch.isfinite(
            torch.tensor(avg_loss, dtype=torch.float32)
        ):
            print(
                f"FATAL: NaN/Inf detected at iteration {engine.state.iteration}, avg_loss={avg_loss}"
            )
            engine.terminate()

    @trainer.on(ignite.engine.Events.EPOCH_COMPLETED(every=5))
    def save_checkpoint(engine):
        ep = engine.state.epoch
        avg_loss = engine.state.metrics.get("avg_loss", None)
        torch.save(
            {
                "epoch": ep,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": opt.state_dict(),
                "loss": avg_loss,
            },
            os.path.join(checkpoint_dir, f"{model_prefix}_epoch_{ep}.pt"),
        )
        if avg_loss is not None:
            print(f"Saved epoch {ep} - Avg Loss: {avg_loss:.6f}")
        else:
            print(f"Saved epoch {ep}")

    @trainer.on(ignite.engine.Events.ITERATION_COMPLETED(every=10))
    def log_training_loss(engine):
        avg_loss = engine.state.metrics.get("avg_loss", None)
        if avg_loss is not None:
            if torch.isfinite(torch.tensor(avg_loss, dtype=torch.float32)):
                writer.add_scalar(
                    "Training/Loss_Iter", avg_loss, engine.state.iteration
                )
            else:
                print(
                    f"WARNING: Iteration {engine.state.iteration} - Loss is {avg_loss}"
                )

    @trainer.on(ignite.engine.Events.EPOCH_COMPLETED)
    def log_epoch_metrics(engine):
        avg_loss = engine.state.metrics.get("avg_loss", None)
        if avg_loss is not None:
            if torch.isfinite(torch.tensor(avg_loss, dtype=torch.float32)):
                writer.add_scalar("Training/Avg_Loss", avg_loss, engine.state.epoch)
                print(f"Epoch {engine.state.epoch} - Avg Train Loss: {avg_loss:.6f}")
            else:
                print(
                    f"ERROR: Epoch {engine.state.epoch} - Loss is {avg_loss}. Stopping training."
                )
                engine.terminate()
        else:
            print(f"Epoch {engine.state.epoch} - Loss tracking unavailable")

    try:
        print(f"Starting training for {train_epochs} epochs...")
        trainer.run()
    finally:
        writer.close()
        print("Training finished.")


if __name__ == "__main__":
    main()
