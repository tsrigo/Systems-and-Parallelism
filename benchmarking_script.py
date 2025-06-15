import logging
from cs336_basics import BasicsTransformerLM, get_batch, cross_entropy, AdamW
from timeit import default_timer
import numpy as np
import torch
import fire

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(name)s] [%(levelname)s]: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
logger = logging.getLogger(__name__)
def main(
    d_model: int = 512,
    d_ff: int = 2048,
    num_layers: int = 6,
    num_heads: int = 8,
    context_length: int = 128,
    warm_up: int = 5,
    n_steps: int = 10,
    batch_size: int = 64,
    vocab_size: int = 10000,
    rope_theta: float = 10000.0
):
    """Benchmark a Transformer Language Model.

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

    # Generate a random batch of data
    dataset = np.random.randint(low=0, high=vocab_size, size=1000*context_length)
    input_ids, target_ids = get_batch(dataset, batch_size, context_length, device)

    # Warm-up steps
    logger.info(f"Running {warm_up} warm-up steps...")
    for _ in range(warm_up):
        model(input_ids)

    # Benchmark forward pass
    ftimes = []
    btimes = []
    logger.info(f"Benchmarking {n_steps} steps...")

    for _ in range(n_steps):
        f1 = default_timer()
        logits = model(input_ids)
        if device == 'cuda':
            torch.cuda.synchronize()
        f2 = default_timer()

        b1 = default_timer()
        loss = cross_entropy(logits, target_ids)
        optimizer.zero_grad()
        loss.backward()
        if device == 'cuda':
            torch.cuda.synchronize()
        b2 = default_timer()

        ftimes.append(f2 - f1)
        btimes.append(b2 - b1)

    # Calculate and log results
    total_time = np.sum(ftimes)
    ftime_per_step = total_time / n_steps
    fstd = np.std(ftimes)
    logger.info(f"===== Forward pass =====")
    logger.info(f"Total time for {n_steps} steps: {total_time:.4f} seconds")
    logger.info(f"Average time per step: {ftime_per_step:.4f} seconds")
    logger.info(f"Standard deviation: {fstd:.4f}")

    total_time = np.sum(btimes)
    btime_per_step = total_time / n_steps
    bstd = np.std(btimes)
    logger.info(f"===== Backward pass =====")
    logger.info(f"Total time for {n_steps} steps: {total_time:.4f} seconds")
    logger.info(f"Average time per step: {btime_per_step:.4f} seconds")
    logger.info(f"Standard deviation: {bstd:.4f}")

    return ftime_per_step, fstd, btime_per_step, bstd

if __name__ == "__main__":
    fire.Fire(main)