"""
Benchmarking script with NVTX annotations for profiling with nsys.

This script includes:
1. NVTX ranges for scaled dot-product attention components (attention scores, softmax, final matmul)
2. NVTX ranges to separate warm-up from actual benchmarking
3. NVTX ranges for forward pass, backward pass, and optimizer steps
4. Individual step annotations for detailed analysis

Usage:
python benchmarking_script.py [args]

For nsys profiling, use:
nsys profile --trace=cuda,nvtx --output=profile python benchmarking_script.py [args]

Then filter by NVTX ranges in the nsys GUI to analyze specific parts:
- Filter on "training_benchmarking" to exclude warm-up
- Filter on "training_forward_pass" or "training_backward_pass" for specific analysis
- Filter on "scaled dot product attention" for attention analysis
- Filter on individual components like "computing softmax" for detailed analysis
"""

import logging
from cs336_basics import BasicsTransformerLM, get_batch, cross_entropy, AdamW, softmax
from timeit import default_timer
import numpy as np
import torch
import torch.cuda.nvtx as nvtx
import fire
import math
from einops import einsum

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(name)s] [%(levelname)s]: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
logger = logging.getLogger(__name__)

@nvtx.range("scaled dot product attention")
def annotated_scaled_dot_product_attention(
    Q, K, V, mask=None
):
    """Annotated scaled dot-product attention with NVTX ranges."""
    d_k = K.shape[-1]
    
    with nvtx.range("computing attention scores"):
        attention_scores = einsum(Q, K, "... query d_k, ... key d_k -> ... query key") / math.sqrt(d_k)
        
        if mask is not None:
            attention_scores = torch.where(mask, attention_scores, float("-inf"))
    
    with nvtx.range("computing softmax"):
        attention_weights = softmax(attention_scores, dim=-1)
    
    with nvtx.range("final matmul"):
        output = einsum(attention_weights, V, "... query key, ... key d_v ->  ... query d_v")
    
    return output

def main(
    d_model: int = 768,
    d_ff: int = 3072,
    num_layers: int = 12,
    num_heads: int = 12,
    context_length: int = 128,
    warm_up: int = 5,
    n_steps: int = 10,
    batch_size: int = 4,
    vocab_size: int = 10000,
    rope_theta: float = 10000.0
):
    """Benchmark complete training steps of a Transformer Language Model.

    Args:
        d_model: Dimension of the model.
        d_ff: Dimension of the feed-forward layer.
        num_layers: Number of transformer layers.
        num_heads: Number of attention heads.
        context_length: Length of the input sequence.
        warm_up: Number of warm-up steps.
        n_steps: Number of steps to benchmark.
        batch_size: Batch size for the input data.
        vocab_size: Size of the vocabulary.
        rope_theta: Theta parameter for rotary position embeddings.
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {device}")

    # Initialize the transformer model
    model = BasicsTransformerLM(
        vocab_size=vocab_size,
        context_length=context_length,
        d_model=d_model,
        num_layers=num_layers,
        num_heads=num_heads,
        d_ff=d_ff,
        rope_theta=rope_theta
    ).to(device)
    optimizer = AdamW(model.parameters())

    # Replace the original scaled_dot_product_attention with annotated version
    import sys
    sys.path.append('/home/kai/projects/CS336/sp25hw/Systems-and-Parallelism/cs336-basics')
    import cs336_basics.model as basics_model
    basics_model.scaled_dot_product_attention = annotated_scaled_dot_product_attention

    # Generate a random batch of data
    dataset = np.random.randint(low=0, high=vocab_size, size=1000*context_length)
    input_ids, target_ids = get_batch(dataset, batch_size, context_length, device)

    # Warm-up steps (marked with NVTX for filtering out in profile)
    logger.info(f"Running {warm_up} warm-up steps...")
    with nvtx.range("warm-up"):
        for _ in range(warm_up):
            logits = model(input_ids)
            loss = cross_entropy(logits, target_ids)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # Benchmark complete training steps
    training_times = []
    forward_times = []
    backward_times = []
    optimizer_times = []
    logger.info(f"Benchmarking {n_steps} complete training steps...")

    with nvtx.range("training_benchmarking"):
        for step in range(n_steps):
            with nvtx.range(f"training_step_{step}"):
                t1 = default_timer()
                
                with nvtx.range("training_forward_pass"):
                    f1 = default_timer()
                    logits = model(input_ids)
                    loss = cross_entropy(logits, target_ids)
                    if device == 'cuda':
                        torch.cuda.synchronize()
                    f2 = default_timer()
                
                with nvtx.range("training_backward_pass"):
                    b1 = default_timer()
                    optimizer.zero_grad()
                    loss.backward()
                    if device == 'cuda':
                        torch.cuda.synchronize()
                    b2 = default_timer()
                
                with nvtx.range("optimizer_step"):
                    o1 = default_timer()
                    optimizer.step()
                    if device == 'cuda':
                        torch.cuda.synchronize()
                    o2 = default_timer()
                
                t2 = default_timer()

                training_times.append(t2 - t1)
                forward_times.append(f2 - f1)
                backward_times.append(b2 - b1)
                optimizer_times.append(o2 - o1)

    # Calculate and log results
    total_time = np.sum(training_times)
    time_per_step = total_time / n_steps
    std = np.std(training_times)
    
    # Forward pass statistics
    forward_total = np.sum(forward_times)
    forward_mean = forward_total / n_steps
    forward_std = np.std(forward_times)
    
    # Backward pass statistics
    backward_total = np.sum(backward_times)
    backward_mean = backward_total / n_steps
    backward_std = np.std(backward_times)
    
    # Optimizer step statistics
    optimizer_total = np.sum(optimizer_times)
    optimizer_mean = optimizer_total / n_steps
    optimizer_std = np.std(optimizer_times)
    
    logger.info(f"===== Complete training step =====")
    logger.info(f"Total time for {n_steps} steps: {total_time:.4f} seconds")
    logger.info(f"Average time per step: {time_per_step:.4f} seconds")
    logger.info(f"Standard deviation: {std:.4f}")
    
    logger.info(f"===== Forward pass =====")
    logger.info(f"Total time: {forward_total:.4f} seconds ({forward_total/total_time*100:.1f}% of total)")
    logger.info(f"Average time per step: {forward_mean:.4f} seconds")
    logger.info(f"Standard deviation: {forward_std:.4f}")
    
    logger.info(f"===== Backward pass =====")
    logger.info(f"Total time: {backward_total:.4f} seconds ({backward_total/total_time*100:.1f}% of total)")
    logger.info(f"Average time per step: {backward_mean:.4f} seconds")
    logger.info(f"Standard deviation: {backward_std:.4f}")
    
    logger.info(f"===== Optimizer step =====")
    logger.info(f"Total time: {optimizer_total:.4f} seconds ({optimizer_total/total_time*100:.1f}% of total)")
    logger.info(f"Average time per step: {optimizer_mean:.4f} seconds")
    logger.info(f"Standard deviation: {optimizer_std:.4f}")

    return {
        'total': {'mean': time_per_step, 'std': std},
        'forward': {'mean': forward_mean, 'std': forward_std},
        'backward': {'mean': backward_mean, 'std': backward_std},
        'optimizer': {'mean': optimizer_mean, 'std': optimizer_std}
    }

if __name__ == "__main__":
    fire.Fire(main)