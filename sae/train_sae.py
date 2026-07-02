import os
import random
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from safetensors.torch import load_file
from sae.sae_model import TopKSAE
from tqdm import tqdm


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def _resample_dead(model, optimizer, activations, dead_mask, device):
    """Hard-reset dead features. For each feature dead > dead_after_steps:
      - draw a random input token, center it against b_dec, normalize;
      - copy that direction into the encoder column and decoder row;
      - zero its bias and Adam moments so it re-trains from scratch.
    Without this, aux-loss alone is too slow once dead-fraction passes ~30 %.
    """
    n_dead = int(dead_mask.sum())
    if n_dead == 0:
        return
    n = activations.shape[0]
    idx = torch.randint(0, n, (n_dead,))
    samples = activations[idx].to(device).float()
    samples = samples - model.b_dec
    samples = F.normalize(samples, dim=-1)

    model.W_enc.data[:, dead_mask] = samples.T
    model.b_enc.data[dead_mask] = 0.0
    model.W_dec.data[dead_mask] = samples

    for p in (model.W_enc, model.b_enc, model.W_dec):
        st = optimizer.state.get(p, None)
        if not st or "exp_avg" not in st:
            continue
        if p is model.W_enc:
            st["exp_avg"][:, dead_mask] = 0
            st["exp_avg_sq"][:, dead_mask] = 0
        elif p is model.b_enc:
            st["exp_avg"][dead_mask] = 0
            st["exp_avg_sq"][dead_mask] = 0
        elif p is model.W_dec:
            st["exp_avg"][dead_mask] = 0
            st["exp_avg_sq"][dead_mask] = 0


def train_sae(activations_path, metadata_path=None, split_filter="train", d_model=None, d_hidden=None, k=32, aux_k=1024, batch_size=2048, lr=5e-4, warmup_steps=100, epochs=10, dead_after_steps=50, resample_every=0, output_dir="sae/checkpoints"):
    print(f"Loading activations from {activations_path}...")
    tensors = load_file(activations_path)
    activations = tensors["encoder_embeddings"]
    print(f"Loaded activations with shape: {activations.shape}")

    # Auto-detect dims from the activation tensor so the SAE can never silently
    # mismatch the TSFM backbone (e.g. chronos-t5-small=512 vs base=768).
    detected_d_model = int(activations.shape[-1])
    if d_model is None:
        d_model = detected_d_model
        print(f"Auto-detected d_model = {d_model}")
    elif d_model != detected_d_model:
        raise ValueError(f"--d_model={d_model} but activations have last dim "
                         f"{detected_d_model}. Wrong activations file or wrong flag.")
    if d_hidden is None:
        d_hidden = 8 * d_model
        print(f"Auto-set d_hidden = {d_hidden}  (8x d_model)")

    # Train the SAE only on TRAIN-split windows. Fitting it on test-window
    # activations is an unsupervised form of leakage that an interviewer will
    # ask about; keep the SAE blind to anything the probe will be tested on.
    if metadata_path and os.path.exists(metadata_path) and activations.dim() == 3:
        meta = pd.read_parquet(metadata_path)
        if "split" in meta.columns and len(meta) == activations.shape[0]:
            keep = torch.as_tensor((meta["split"].values == split_filter))
            print(f"Filtering to split='{split_filter}': {int(keep.sum())}/{len(meta)} windows")
            activations = activations[keep]
        else:
            print("WARNING: metadata has no 'split' column or row/window count "
                  "mismatch -> training on ALL windows. Re-run extract_activations.py "
                  "to regenerate labelled metadata before trusting probe results.")
    elif metadata_path:
        print(f"WARNING: metadata '{metadata_path}' not found -> training on ALL windows.")

    if activations.dim() == 3:
        activations = activations.reshape(-1, activations.shape[-1])
    print(f"Token activations for SAE training: {tuple(activations.shape)}")

    activations = activations.to(torch.float32)
    activation_mean = activations.mean(dim=0)
    activation_variance = activations.var(dim=0).mean().item()
    print(f"Activation variance: {activation_variance:.4f}")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    dataset = TensorDataset(activations)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    model = TopKSAE(d_model=d_model, d_hidden=d_hidden, k=k, aux_k=aux_k).to(device)
    # Initialize decoder bias to activation mean
    model.b_dec.data = activation_mean.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # Warmup scheduler
    def warmup_lambda(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        return 1.0
        
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warmup_lambda)
    
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Training TopK SAE with k={k}, aux_k={aux_k}, lr={lr}")
    
    # Steps since each feature last fired. Starts at 0 (nothing dead yet, so the
    # aux-revival loss stays off until a feature has genuinely gone quiet for
    # `dead_after_steps` consecutive optimizer steps).
    steps_since_fired = torch.zeros(d_hidden, device=device)
    
    model.train()
    global_step = 0
    for epoch in range(epochs):
        total_mse = 0
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        for (batch,) in pbar:
            batch = batch.to(device)
            
            dead_mask = (steps_since_fired > dead_after_steps)

            optimizer.zero_grad()
            acts, reconstructed, aux_loss = model(batch, dead_mask=dead_mask)
            
            mse_loss = F.mse_loss(reconstructed, batch)
            # Total loss includes aux loss for dead feature revival
            loss = mse_loss
            if isinstance(aux_loss, torch.Tensor):
                 loss = loss + aux_loss
            
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            model.normalize_decoder()
            
            # Update last-fired counter: reset to 0 if the feature fired in this
            # batch, else +1. A feature is "dead" once it has not fired for
            # dead_after_steps consecutive steps.
            fired = (acts > 0).sum(dim=0) > 0
            steps_since_fired = torch.where(
                fired,
                torch.zeros_like(steps_since_fired),
                steps_since_fired + 1,
            )

            total_mse += mse_loss.item()

            dead_fraction = (steps_since_fired > dead_after_steps).float().mean().item()
            mean_l0 = (acts > 0).float().sum(dim=-1).mean().item()
            norm_mse = mse_loss.item() / (activation_variance + 1e-8)

            pbar.set_postfix({
                "nMSE": f"{norm_mse:.3f}",
                "L0": f"{mean_l0:.1f}",
                "dead": f"{dead_fraction:.1%}",
                "lr": f"{scheduler.get_last_lr()[0]:.2e}"
            })
            global_step += 1

            # Hard reset of dead features every `resample_every` steps. The
            # aux-revival loss alone is too slow once features have collapsed:
            # we copy live input directions into the dead encoder/decoder slots
            # and zero their Adam moments so they re-learn from a clean state.
            # Same recipe Anthropic / OpenAI use to keep TopK SAEs alive.
            if resample_every and global_step % resample_every == 0 and dead_mask.any():
                _resample_dead(model, optimizer, activations, dead_mask, device)
                # Give the resampled features a grace period before they're
                # eligible to be killed again.
                steps_since_fired[dead_mask] = 0
            
        avg_mse = total_mse / len(dataloader)
        avg_norm_mse = avg_mse / (activation_variance + 1e-8)
        print(f"Epoch {epoch+1} | MSE: {avg_mse:.4f} | nMSE: {avg_norm_mse:.3f} | L0: {mean_l0:.1f} | Dead: {dead_fraction:.1%}")
        
    save_path = os.path.join(output_dir, f"sae_topk_{k}.pt")
    torch.save(model.state_dict(), save_path)
    print(f"Saved SAE checkpoint to {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train TopK Sparse Autoencoder")
    parser.add_argument("--activations", type=str, default="activations/ETTh1_activations.safetensors", help="Path to cached activations")
    parser.add_argument("--d_model", type=int, default=None, help="TSFM hidden size (auto-detected from activations if omitted)")
    parser.add_argument("--d_hidden", type=int, default=None, help="SAE hidden size (defaults to 8x d_model)")
    parser.add_argument("--k", type=int, default=32, help="TopK active features")
    parser.add_argument("--aux_k", type=int, default=1024, help="Auxiliary K for dead feature revival")
    parser.add_argument("--batch_size", type=int, default=2048, help="Batch size")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
    parser.add_argument("--warmup_steps", type=int, default=100, help="LR warmup steps (smaller is better when total steps ~600)")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--metadata", type=str, default="activations/ETTh1_metadata.parquet", help="Metadata parquet (used to filter to the train split)")
    parser.add_argument("--split_filter", type=str, default="train", help="Which split to train the SAE on")
    parser.add_argument("--dead_after_steps", type=int, default=50, help="Steps without firing before a feature counts as dead")
    parser.add_argument("--resample_every", type=int, default=0, help="Hard-reset dead features every N steps (0 disables). Off by default: it can fight reconstruction. Aux loss + small dead_after_steps + larger aux_k handles dead features without spiking nMSE.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--output_dir", type=str, default="sae/checkpoints", help="Directory to save the SAE checkpoint")

    args = parser.parse_args()
    set_seed(args.seed)

    train_sae(
        activations_path=args.activations,
        metadata_path=args.metadata,
        split_filter=args.split_filter,
        d_model=args.d_model,
        d_hidden=args.d_hidden,
        k=args.k,
        aux_k=args.aux_k,
        batch_size=args.batch_size,
        lr=args.lr,
        warmup_steps=args.warmup_steps,
        epochs=args.epochs,
        dead_after_steps=args.dead_after_steps,
        resample_every=args.resample_every,
        output_dir=args.output_dir,
    )
