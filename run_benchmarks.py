import pandas as pd
from benchmarking_script import main  # Assuming your timing tool is in a file named main.py
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(name)s] [%(levelname)s]: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
logger = logging.getLogger(__name__)

# Define model configurations
configs = [
    {"size": "small", "d_model": 768, "d_ff": 3072, "num_layers": 12, "num_heads": 12},
    {"size": "medium", "d_model": 1024, "d_ff": 4096, "num_layers": 24, "num_heads": 16},
    {"size": "large", "d_model": 1280, "d_ff": 5120, "num_layers": 36, "num_heads": 20},
    {"size": "xl", "d_model": 1600, "d_ff": 6400, "num_layers": 48, "num_heads": 25},
    {"size": "2.7B", "d_model": 2560, "d_ff": 10240, "num_layers": 32, "num_heads": 32},
]

def run_benchmarks():
    results = []
    
    for config in configs:
        logger.info(f"Running benchmark for model size: {config['size']}")
        try:
            # Run the timing tool with the specified parameters
            ftime_per_step, fstd, btime_per_step, bstd = main(
                d_model=config["d_model"],
                d_ff=config["d_ff"],
                num_layers=config["num_layers"],
                num_heads=config["num_heads"],
                context_length=128,  # Default value
                warm_up=2,           # Default value
                n_steps=10,          # Default value
                batch_size=1,       # Default value
                vocab_size=10000,    # Default value
                rope_theta=10000.0   # Default value
            )
            
            # Collect results
            results.append({
                "Model Size": config["size"],
                "d_model": config["d_model"],
                "d_ff": config["d_ff"],
                "num_layers": config["num_layers"],
                "num_heads": config["num_heads"],
                "Forward Time (s)": ftime_per_step,
                "Forward Std (s)": fstd,
                "Backward Time (s)": btime_per_step,
                "Backward Std (s)": bstd
            })
        except Exception as e:
            logger.error(f"Failed to run benchmark for {config['size']}: {str(e)}")
            continue
    
    # Create DataFrame and save to markdown
    df = pd.DataFrame(results)
    markdown_table = df.to_markdown(index=False, floatfmt=".4f")
    
    # Save markdown table to file
    with open("benchmark_results3.md", "w") as f:
        f.write("# Benchmark Results\n\n")
        f.write(markdown_table)
    
    logger.info("Benchmark results saved to benchmark_results.md")
    return df

if __name__ == "__main__":
    run_benchmarks()