import pathlib
import datetime
import shutil
import re
import traceback
from pathlib import Path

import numpy as np
import torch
import yaml
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm

import model
from Tools import score_files
from dataloader import LoadDataset


class Trainer:
    STRUCT_TOKENS = {
        r"\frac", r"\sqrt", r"\sum", r"\lim", r"\int",
        "^", "_",
        "{", "}", "(", ")", "[", "]",
    }

    TOKEN_BLACKLIST = {
        "_START_", "_END_", "_PAD_", "_UNK_",
        "{", "}", "(", ")", "[", "]",
        ",", ".", ":", ";",
    }

    def __init__(self, config):
        # Save config.
        self.config = config

        # Prepare save folders.
        now = datetime.datetime.now()
        self.save_path = (
            pathlib.Path(self.config["model"]["model_save_path"])
            / f"{now.year}-{now.month}-{now.day}_{now.hour}-{now.minute}"
        )
        self.config["model"]["model_save_path"] = str(self.save_path)

        print(f"save path: {self.save_path}")
        pathlib.Path(self.save_path).mkdir(exist_ok=True, parents=True)

        self.save_path_results = self.save_path / "results"
        self.save_path_results.mkdir(exist_ok=True, parents=True)

        self.save_path_model = self.save_path / "model"
        self.save_path_model.mkdir(exist_ok=True, parents=True)

        # Prepare dataloaders.
        self.train_dataloader = None
        self.val_dataloader = None
        self.test_dataloader = {}
        self.lazy_cache_train = False

        # Load test dataset.
        # The merged dataset config uses:
        # dataset:
        #   test:
        #     files:
        #       - image_dir: ...
        #         csv: ...
        print("load test dataset")
        test_name = self.config["dataset"]["test"].get("name", "im2latexv2")
        self.test_dataloader = {
            test_name: self._load_dataset(mode="test")
        }

        for name, dataloader in self.test_dataloader.items():
            self._save_gt(f"test_{name}", dataloader)

        # Load validation and training datasets only when training is enabled.
        if "train" in self.config["arguments"]["task"]:
            print("load val dataset")
            self.val_dataloader = self._load_dataset(mode="val")
            self._save_gt("val", self.val_dataloader)

            print("load train dataset")
            self.train_dataloader = self._load_dataset(mode="train")

            self.lazy_cache_train = self.config["dataset"]["train"]["cache"]

        # Use vocabulary from the first test dataloader.
        first_test_name = list(self.test_dataloader.keys())[0]
        self.vocab = self.test_dataloader[first_test_name].dataset.vocab

        print("Create model")
        self.model = model.model(
            self.config,
            self.vocab,
            self.train_dataloader.dataset.styles
            if self.train_dataloader is not None
            else None,
        )

        # Optimizer is only required for training.
        self.optimizer = None
        if "train" in self.config["arguments"]["task"]:
            self.optimizer = torch.optim.Adam(
                self.model.parameters(),
                self.config["train"]["init_lr"],
            )

        self.ckpt = {
            "epoch": 0,
            "best_fitness": 0,
            "encoder_model": None,
            "decoder_model": None,
            "optimizer": None,
            "train_loss": [],
            "train_accuracy": [],
            "val_loss": [],
            "val_accuracy": [],
        }

        if self.config["arguments"].get("resume_from"):
            print("loading model")
            self._load_model()

        self.epoch = self.ckpt["epoch"]

        # Count model parameters.
        params = 0
        for p in self.model.decoder.parameters():
            params += np.prod(p.size())
        for p in self.model.encoder.parameters():
            params += np.prod(p.size())
        print("total parameters = ", params)

        # Save vocabulary and config.
        shutil.copyfile(
            self.config["dataset"]["vocab_file"],
            self.save_path / "latex.vocab",
        )

        with open(self.save_path / "config.yml", "w") as outfile:
            yaml.dump(self.config, outfile, default_flow_style=False)

    def _save_gt(self, mode, dataloader):
        split_name = mode.split("_")[0]
        lazy_cache = self.config["dataset"][split_name]["cache"]
        gt_path = self.save_path / f"gt_{mode}.txt"

        with open(gt_path, "w") as file:
            for images, labels in tqdm(dataloader, postfix=f"save {mode} dataset"):
                formulas = labels[1].cpu().numpy()

                for label_i, label in enumerate(labels[0][0]):
                    raw = dataloader.dataset.vocab.seq2text(formulas[label_i])
                    clean = self._sanitize_text(raw)
                    file.write(label + ": " + clean + "\n")

            if lazy_cache:
                dataloader.dataset.cache_all_files()
                dataloader.num_workers = self.config["dataset"][split_name]["num_workers"]

    def _load_dataset(self, mode):
        dataset_cfg = self.config["dataset"]
        split_cfg = dataset_cfg[mode]

        dataset = LoadDataset(
            dataset_config=dataset_cfg,
            vocab_path=dataset_cfg["vocab_file"],
            image_size=self.config["model"]["image"],
            mode=mode,
            cache=split_cfg["cache"],
            max_size=self.config["model"]["max_len"],
            transforms=split_cfg["transforms"],
            no_sampling=split_cfg["no_sampling"],
            no_arrays=split_cfg["no_arrays"],
            only_basic=split_cfg["only_basic"],
            dpi=dataset_cfg.get("target_dpi", dataset_cfg.get("dpi", 600)),
        )

        ds_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=split_cfg["batch_size"],
            shuffle=split_cfg["shuffle"],
            drop_last=split_cfg["drop_last"],
            num_workers=0 if split_cfg["cache"] else split_cfg["num_workers"],
        )

        if dataset.vocab is not None:
            self.config["model"]["vocab_size"] = len(dataset.vocab.id2token)

        return ds_loader

    def _load_model(self):
        try:
            self.ckpt = torch.load(self.config["arguments"]["resume_from"])

            self.model.decoder.load_state_dict(
                self.ckpt["decoder_model"],
                strict=True,
            )
            self.model.encoder.load_state_dict(
                self.ckpt["encoder_model"],
                strict=True,
            )

            if self.optimizer is not None and self.ckpt.get("optimizer") is not None:
                self.optimizer.load_state_dict(self.ckpt["optimizer"])

            print("-------------------load model from checkpoint-------------------")
            print(
                f"Loading checkpoint file: "
                f"{Path(self.config['arguments']['resume_from']).name}"
            )

        except Exception as e:
            print(
                "Error occurred during loading",
                self.config["arguments"]["resume_from"],
                "(",
                e,
                ")",
            )

    def _train_batch(self, labels, images):
        labels = labels[1]

        self.model.encoder.zero_grad()
        self.model.decoder.zero_grad()

        images = images.to(torch.float).to(self.config["device"])
        labels = labels.to(self.config["device"])

        if self.config["train"]["fp16"]:
            with torch.amp.autocast(
                device_type="cuda" if "cuda" in self.config["device"] else "cpu"
            ):
                outputs, loss, acc = self.model(images, labels, self.epoch)
        else:
            outputs, loss, acc = self.model(images, labels, self.epoch)

        word_loss = loss.item()
        self.optimizer.zero_grad()

        if self.config["train"]["fp16"]:
            self.scaler.scale(loss).backward()

            if self.config["train"]["grad_clip_value"]:
                torch.nn.utils.clip_grad_value_(
                    self.model.encoder.parameters(),
                    self.config["train"]["grad_clip_value"],
                )
                torch.nn.utils.clip_grad_value_(
                    self.model.decoder.parameters(),
                    self.config["train"]["grad_clip_value"],
                )

            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()

            if self.config["train"]["grad_clip_value"]:
                torch.nn.utils.clip_grad_value_(
                    self.model.encoder.parameters(),
                    self.config["train"]["grad_clip_value"],
                )
                torch.nn.utils.clip_grad_value_(
                    self.model.decoder.parameters(),
                    self.config["train"]["grad_clip_value"],
                )

            self.optimizer.step()

        loss_value = loss.item()
        return loss_value, acc, word_loss

    def train(self):
        pationt = 0
        best_fitness = 0

        if self.config["train"]["fp16"]:
            self.scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.ckpt["epoch"], self.config["train"]["epochs"]):
            self.epoch = epoch

            # Train.
            self.model.encoder.train()
            self.model.decoder.train()

            mean_loss = 0
            mean_accuracy = 0

            with tqdm(self.train_dataloader) as pbar:
                for itr, (images, labels) in enumerate(pbar):
                    loss_value, acc, word_loss = self._train_batch(labels, images)

                    mean_loss += 1 / (itr + 1) * (loss_value - mean_loss)
                    mean_accuracy += 1 / (itr + 1) * (acc.item() - mean_accuracy)

                    s = "EPOCH[%d/%d] loss=%2.4f - accuracy=%2.4f" % (
                        self.epoch,
                        self.config["train"]["epochs"],
                        mean_loss,
                        mean_accuracy,
                    )
                    pbar.set_description(s)

                if self.lazy_cache_train:
                    self.train_dataloader.dataset.cache_all_files()
                    self.train_dataloader.num_workers = self.config["dataset"]["train"]["num_workers"]
                    self.lazy_cache_train = False

            self.ckpt["train_loss"].append(mean_loss)
            self.ckpt["train_accuracy"].append(mean_accuracy)

            if (self.epoch + 1) % self.config["train"]["lr_descent"][0] == 0:
                for g in self.optimizer.param_groups:
                    g["lr"] = self.config["train"]["lr_descent"][1] * g["lr"]

            if self.epoch % self.config["train"]["save_each"] == 0:
                self._save_model(f"epoch_{self.epoch}.pt")

            # Validation.
            if (
                self.epoch % self.config["val"]["each"] == 0
                and self.epoch > self.config["train"]["wait_n_epochs"]
            ):
                metrics = self.validate()

                if pationt > self.config["train"]["early_stop"]:
                    print("early stopping ... ")
                    break

                if metrics[self.config["val"]["metric"]] > best_fitness:
                    pationt = 0
                    best_fitness = metrics[self.config["val"]["metric"]]
                    self.ckpt["best_fitness"] = best_fitness
                    print("save best at accuracy = ", best_fitness)
                    self._save_model("best.pt")
                else:
                    pationt += 1

            # Testing.
            if (
                epoch % self.config["test"]["each"] == 0
                and self.epoch > self.config["train"]["wait_n_epochs"]
            ):
                self.test()

    def _sanitize_text(self, s: str) -> str:
        if s is None:
            return ""

        s = s.replace("_UNK_", "")
        s = re.sub(r"\s+", " ", s).strip()
        return s

    def _pred_ids_to_tokens(self, pred_ids):
        """Convert predicted token ids to token strings."""
        toks = []

        for tid in pred_ids:
            try:
                toks.append(self.vocab.id2token[int(tid)])
            except Exception:
                toks.append(str(int(tid)))

        return toks

    def _img_to_uint8_gray(self, img_tensor) -> np.ndarray:
        """
        Convert a model-input tensor to a viewable uint8 grayscale image.

        This function does not apply per-image min-max normalization.
        It assumes the input is approximately in [0, 1].
        """
        x = img_tensor.detach().cpu()

        if x.ndim == 3:
            x = x[0]

        x = x.to(torch.float32).clamp(0.0, 1.0).numpy()
        x = (x * 255.0).round().clip(0, 255).astype(np.uint8)

        return x

    def _save_overlay(
        self,
        base_u8: np.ndarray,
        heat: np.ndarray,
        out_path: Path,
        alpha: float = 0.45,
    ):
        """
        Save an attention overlay image.

        Args:
            base_u8: Grayscale base image with shape [H, W].
            heat: Attention map to be resized to [H, W].
            out_path: Output image path.
            alpha: Overlay opacity.
        """
        height, width = base_u8.shape
        h = heat.astype(np.float32)

        lo = float(np.quantile(h, 0.005))
        hi = float(np.quantile(h, 0.995))

        if hi <= lo + 1e-8:
            h = h - h.min()
            h = h / (h.max() + 1e-8)
        else:
            h = np.clip(h, lo, hi)
            h = (h - lo) / (hi - lo + 1e-6)

        heat_img = Image.fromarray((h * 255).astype(np.uint8)).resize(
            (width, height),
            resample=Image.BILINEAR,
        )
        heat_up = np.array(heat_img).astype(np.float32) / 255.0

        plt.figure()
        plt.imshow(base_u8, cmap="gray")
        plt.imshow(heat_up, alpha=alpha)
        plt.axis("off")
        plt.savefig(out_path, bbox_inches="tight", pad_inches=0)
        plt.close()

    def _save_heatmap_img(self, arr2d: np.ndarray, out_path: Path):
        """Save a normalized heatmap image."""
        lo = float(np.quantile(arr2d, 0.005))
        hi = float(np.quantile(arr2d, 0.995))

        if hi <= lo + 1e-8:
            arr2d = arr2d - arr2d.min()
            arr2d = arr2d / (arr2d.max() + 1e-8)
        else:
            arr2d = np.clip(arr2d, lo, hi)
            arr2d = (arr2d - lo) / (hi - lo + 1e-6)

        plt.figure()
        plt.imshow(arr2d, cmap="jet")
        plt.axis("off")
        plt.savefig(out_path, bbox_inches="tight", pad_inches=0)
        plt.close()

    def _evaluate(self, pred_path, dataloader, mode, more_information=False):
        self.more_information = more_information

        self.model.encoder.eval()
        self.model.decoder.eval()

        prediction_file = open(pred_path, "w")

        # Output directories.
        attn_dir = self.save_path_results / f"{mode}_attn"
        attn_dir.mkdir(exist_ok=True, parents=True)

        inp_dir = self.save_path_results / f"{mode}_inputs"
        inp_dir.mkdir(exist_ok=True, parents=True)

        ovl_dir = self.save_path_results / f"{mode}_overlay"
        ovl_dir.mkdir(exist_ok=True, parents=True)

        def _reshape_s(s1d: np.ndarray) -> np.ndarray:
            seq_len = int(s1d.shape[0])

            if seq_len == 867:
                return s1d.reshape(17, 51)

            side = int(np.sqrt(seq_len))
            if side * side == seq_len:
                return s1d.reshape(side, side)

            return s1d.reshape(1, -1)

        with torch.no_grad():
            with tqdm(dataloader, desc=f"{mode} ({self.epoch})") as data_loader:
                for batch_i, [images, labels] in enumerate(data_loader):
                    image_names = labels[0][0]

                    if len(labels[0]) > 2:
                        style = labels[0][2]
                    else:
                        style = False

                    labels = labels[1]

                    images = images.to(torch.float).to(self.config["device"])
                    labels = labels.to(self.config["device"])

                    outputs, loss, acc = self.model(images, labels, self.epoch)

                    # Save predictions and optional attention visualizations.
                    selection = range(len(images))

                    for i in range(len(outputs)):
                        seq = list(outputs[i].cpu().numpy())
                        pred_text_raw = self.vocab.seq2text(seq)
                        pred_text = self._sanitize_text(pred_text_raw)

                        prediction_file.write(f"{image_names[i]}: " + pred_text)

                        if style:
                            prediction_file.write(f" style: {style[i]}")

                        prediction_file.write("\n")

                        if i in selection:
                            # Save model input image.
                            base_u8 = self._img_to_uint8_gray(images[i])
                            Image.fromarray(base_u8).save(
                                inp_dir / f"{image_names[i]}.input.png"
                            )

                            # Save attention maps.
                            try:
                                img1 = images[i:i + 1]
                                lbl1 = labels[i:i + 1]

                                # This requires the model to support return_attn=True.
                                preds1, loss1, acc1, attn_info = self.model(
                                    img1,
                                    lbl1,
                                    self.epoch,
                                    return_attn=True,
                                )

                                attn_last = attn_info["cross_attn_last"][0]

                                if attn_last is None:
                                    continue

                                # [nhead, T, S] -> [T, S]
                                attn_ts_max = (
                                    attn_last.max(dim=0)
                                    .values
                                    .detach()
                                    .cpu()
                                    .numpy()
                                    .astype(np.float32)
                                )
                                attn_ts_mean = (
                                    attn_last.mean(dim=0)
                                    .detach()
                                    .cpu()
                                    .numpy()
                                    .astype(np.float32)
                                )

                                # Save full token-attention matrices.
                                np.save(
                                    attn_dir / f"{image_names[i]}.attn_headMax_TxS.npy",
                                    attn_ts_max,
                                )
                                self._save_heatmap_img(
                                    attn_ts_max,
                                    attn_dir / f"{image_names[i]}.attn_headMax_TxS.png",
                                )

                                np.save(
                                    attn_dir / f"{image_names[i]}.attn_headMean_TxS.npy",
                                    attn_ts_mean,
                                )
                                self._save_heatmap_img(
                                    attn_ts_mean,
                                    attn_dir / f"{image_names[i]}.attn_headMean_TxS.png",
                                )

                                pred_ids = list(preds1[0].detach().cpu().numpy())
                                tokens = self._pred_ids_to_tokens(pred_ids)
                                valid_idx = [
                                    j for j, tk in enumerate(tokens)
                                    if tk not in self.TOKEN_BLACKLIST
                                ]

                                def _token_mean(attn_ts_2d: np.ndarray) -> np.ndarray:
                                    if len(valid_idx) > 0:
                                        return attn_ts_2d[valid_idx].mean(axis=0)

                                    if attn_ts_2d.shape[0] > 1:
                                        return attn_ts_2d[1:].mean(axis=0)

                                    return attn_ts_2d.mean(axis=0)

                                mean_s_max = _token_mean(attn_ts_max)
                                mean_s_mean = _token_mean(attn_ts_mean)

                                np.save(
                                    attn_dir / f"{image_names[i]}.attn_headMax_blacklistMeanS.npy",
                                    mean_s_max,
                                )
                                np.save(
                                    attn_dir / f"{image_names[i]}.attn_headMean_blacklistMeanS.npy",
                                    mean_s_mean,
                                )

                                # Last non-special token.
                                if len(valid_idx) > 0:
                                    last_t = valid_idx[-1]
                                else:
                                    last_t = attn_ts_max.shape[0] - 1

                                last_s_max = attn_ts_max[last_t]
                                last_s_mean = attn_ts_mean[last_t]

                                np.save(
                                    attn_dir / f"{image_names[i]}.attn_headMax_lastNonSpecialS.npy",
                                    last_s_max,
                                )
                                np.save(
                                    attn_dir / f"{image_names[i]}.attn_headMean_lastNonSpecialS.npy",
                                    last_s_mean,
                                )

                                # Reshape attention vectors for visualization.
                                mean_heat_max = _reshape_s(mean_s_max)
                                mean_heat_mean = _reshape_s(mean_s_mean)
                                last_heat_max = _reshape_s(last_s_max)
                                last_heat_mean = _reshape_s(last_s_mean)

                                # Save head-max heatmaps.
                                np.save(
                                    attn_dir / f"{image_names[i]}.headMax_struct_mean_heat.npy",
                                    mean_heat_max,
                                )
                                self._save_heatmap_img(
                                    mean_heat_max,
                                    attn_dir / f"{image_names[i]}.headMax_struct_mean_heat.png",
                                )

                                np.save(
                                    attn_dir / f"{image_names[i]}.headMax_last_heat.npy",
                                    last_heat_max,
                                )
                                self._save_heatmap_img(
                                    last_heat_max,
                                    attn_dir / f"{image_names[i]}.headMax_last_heat.png",
                                )

                                # Save head-mean heatmaps.
                                np.save(
                                    attn_dir / f"{image_names[i]}.headMean_struct_mean_heat.npy",
                                    mean_heat_mean,
                                )
                                self._save_heatmap_img(
                                    mean_heat_mean,
                                    attn_dir / f"{image_names[i]}.headMean_struct_mean_heat.png",
                                )

                                np.save(
                                    attn_dir / f"{image_names[i]}.headMean_last_heat.npy",
                                    last_heat_mean,
                                )
                                self._save_heatmap_img(
                                    last_heat_mean,
                                    attn_dir / f"{image_names[i]}.headMean_last_heat.png",
                                )

                                # Save overlays.
                                self._save_overlay(
                                    base_u8=base_u8,
                                    heat=mean_heat_max,
                                    out_path=ovl_dir / f"{image_names[i]}.overlay_headMax_blacklist_mean.png",
                                    alpha=0.45,
                                )

                                self._save_overlay(
                                    base_u8=base_u8,
                                    heat=last_heat_max,
                                    out_path=ovl_dir / f"{image_names[i]}.overlay_headMax_last.png",
                                    alpha=0.45,
                                )

                                self._save_overlay(
                                    base_u8=base_u8,
                                    heat=mean_heat_mean,
                                    out_path=ovl_dir / f"{image_names[i]}.overlay_headMean_blacklist_mean.png",
                                    alpha=0.45,
                                )

                                self._save_overlay(
                                    base_u8=base_u8,
                                    heat=last_heat_mean,
                                    out_path=ovl_dir / f"{image_names[i]}.overlay_headMean_last.png",
                                    alpha=0.45,
                                )

                            except Exception as e:
                                print(
                                    f"[WARN] attention dump failed for "
                                    f"{image_names[i]}: {repr(e)}"
                                )
                                traceback.print_exc()

        prediction_file.close()

        gt = self.save_path / f"gt_{mode}.txt"
        metrics = score_files(gt, pred_path, more_information=self.more_information)

        return metrics

    def validate(self):
        pred_path = self.save_path_results / f"val_{self.epoch}_predictions.txt"
        pred_path.parent.mkdir(exist_ok=True, parents=True)

        metrics = self._evaluate(
            pred_path,
            self.val_dataloader,
            "val",
            self.config["test"]["more_information"],
        )

        self._process_metrics(f"val_{self.epoch}_results.txt", metrics)

        return metrics

    def _save_model(self, file_name):
        self.ckpt["epoch"] = self.epoch + 1
        self.ckpt["encoder_model"] = self.model.encoder.state_dict()
        self.ckpt["decoder_model"] = self.model.decoder.state_dict()

        if self.optimizer is not None:
            self.ckpt["optimizer"] = self.optimizer.state_dict()

        torch.save(self.ckpt, self.save_path_model / file_name)

    def test(self):
        metrics = {}

        for name, dataloader in self.test_dataloader.items():
            pred_path = self.save_path_results / f"test_{self.epoch}_predictions.txt"
            pred_path.parent.mkdir(exist_ok=True, parents=True)

            temp_metrics = self._evaluate(
                pred_path,
                dataloader,
                f"test_{name}",
                self.config["test"]["more_information"],
            )

            metrics[name] = temp_metrics

            self._process_metrics(
                f"test_{name}_{self.epoch}_results.txt",
                metrics[name],
                name,
            )

        return metrics

    def _process_metrics(self, filename, metrics, name=None):
        with open(str(self.save_path_results / filename), "w") as f:
            if name:
                print(f"Dataset {name}:\n")
                f.write(f"Dataset {name}:\n")

            for key, value in metrics.items():
                if isinstance(value, dict):
                    f.write(f"\t{key}:\n")
                    for k, v in value.items():
                        f.write(f"\t\t{k}: {v}\n")
                else:
                    print(f"\t{key}: {value}")
                    f.write(f"\t{key}: {value}\n")