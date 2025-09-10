from enum import Enum
import time
import torch
import argparse
import logging
from datetime import datetime
from pathlib import Path

from transformer import TransformerModel
from dataset import collate_fn, collate_fn_ultra

# Configure logging
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)
log_file = log_dir / f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
class CollateWrapper:
    def __init__(self, max_length): 
        self.max_length = max_length
        
    def __call__(self, batch): 
        return collate_fn(batch, self.max_length)

class DataMode(Enum):
    BRAIN = 1
    BIG_BRAIN = 2
    ULTRA_BIG_BRAIN = 3
    ULTRA_DUPER_BIG_BRAIN = 4


def get_gpt2_model(vocab_size: int) -> torch.nn.Module:
    """Create a small GPT-2-like model as per task requirements."""
    model = TransformerModel(
        ntoken=vocab_size,
        d_model=1024,  # hidden size = 1024 as required
        nhead=4,       # 4 heads as per task requirement
        d_hid=4096,    # 4x hidden size is typical
        nlayers=1,     # single layer as required
        dropout=0.1
    )
    logging.info("Created model with parameters:")
    logging.info("  - Vocabulary size: {vocab_size}")
    logging.info("  - Hidden size (d_model): 1024")
    logging.info("  - Number of heads: 4")
    logging.info("  - Feed-forward size: 4096")
    logging.info("  - Number of layers: 1")
    logging.info("  - Dropout: 0.1")
    return model


def run_epoch(data_mode: DataMode) -> None:
    """Run a training epoch with specified data loading mode.
    
    Args:
        data_mode: Enum specifying which dataset/dataloader implementation to use
    """
    from dataset import (
        BrainDataset, BigBrainDataset, UltraDuperBigBrainDataset,
        UltraDuperBigBrainBatchSampler
    )
    from pathlib import Path
    from torch.utils.data import DataLoader
    import time
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")
    
    # Get dataset path
    data_path = Path("/Users/atom/dev/efficient-dl-systems/week03_fast_pipelines/homework/task2/wikitext-103/wiki.train.tokens")
    logging.info(f"Loading data from: {data_path}")
    
    # Create model and move to device
    
    # Configure dataset and dataloader based on mode
    batch_size = 4
    max_length = 180
    
    if data_mode == DataMode.BRAIN:
        dataset = BrainDataset(data_path, max_length=max_length)
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4
        )
    elif data_mode == DataMode.BIG_BRAIN:
        dataset = BigBrainDataset(data_path, max_length=max_length)
      
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            collate_fn=CollateWrapper(max_length)
        )
    else: 
        dataset = UltraDuperBigBrainDataset(data_path, max_length=max_length)
        dataloader = DataLoader(
            dataset,
            num_workers=4,
            batch_sampler=UltraDuperBigBrainBatchSampler(dataset, batch_size, k=10),  # k=10 - разница длин в батче
            collate_fn=CollateWrapper(max_length)
        )
    model = get_gpt2_model(vocab_size=dataset.vocab_size).to(device)
    model.train()
    
    # Run mock training epoch and measure statistics
    stats = mock_training_epoch(dataloader, model, device, max_length)
    
    # Log statistics
    logging.info("=" * 50)
    logging.info(f"Training Statistics for {data_mode.name}")
    logging.info("-" * 30)
    logging.info(f"Total batches processed: {stats['total_batches']}")
    logging.info(f"Total tokens processed: {stats['total_tokens']:,}")
    logging.info("\nBatch processing times (ms):")
    logging.info(f"{'min':>8}: {stats['min']:.2f}")
    logging.info(f"{'max':>8}: {stats['max']:.2f}")
    logging.info(f"{'mean':>8}: {stats['mean']:.2f}")
    logging.info(f"{'median':>8}: {stats['median']:.2f}")
    logging.info(f"Performance: {stats['tokens_per_second']:,.2f} tokens/second")


from tqdm import tqdm

def mock_training_epoch(dataloader, model, device, max_length, warmup_batches=10):
    """Mock one training epoch and measure batch processing times.
    
    This function:
    1. Warms up the GPU with a few batches
    2. Measures forward pass time for each batch
    3. Calculates statistics: min, max, mean, median batch processing times
    
    
    Args:
        dataloader: DataLoader instance for the dataset
        model: The model to train
        device: Device to run the model on
        max_length: Maximum sequence length for padding
        warmup_batches: Number of batches to use for GPU warmup
        
    Returns:
        Dictionary with statistics:
        - min, max, mean, median batch processing times
        - total_batches: number of batches processed
        - total_tokens: number of tokens processed
        - tokens_per_second: throughput
    """
    
    
    logging.info("=" * 50)
    logging.info("Starting new training epoch")
    logging.info(f"Batch size: {dataloader.batch_size if hasattr(dataloader, 'batch_size') else 'N/A'}")
    logging.info(f"Max sequence length: {max_length}")
    logging.info(f"Number of warmup batches: {warmup_batches}")
    def create_mask(size):
        """Create causal mask for transformer."""
        mask = torch.triu(torch.ones(size, size) * float('-inf'), diagonal=1)
        return mask.to(device)
    
    logging.info("Starting GPU warmup...")
    # Warmup phase
    for i, batch in enumerate(dataloader):
        if i >= warmup_batches:
            break
        logging.debug(f"Warmup batch {i+1}/{warmup_batches}")
        if isinstance(batch, (tuple, list)):
            inputs, targets = batch
        else:
            inputs = batch
        inputs = inputs.to(device)
        seq_len = inputs.size(0)  # sequence length для sequence-first формата
        mask = create_mask(seq_len)
        _ = model(inputs, mask)
        
    
    print("Running training epoch...")
    times = []
    total_tokens = 0
    
    # Measurement phase
    total_batches = len(dataloader)
    print(f"\nStarting training with {total_batches} batches...")
    
    for batch_idx, batch in enumerate(tqdm(dataloader, desc="Training", unit="batch")):
        if isinstance(batch, (tuple, list)):
            inputs, targets = batch  # for BrainDataset and collate_fn cases
        else:
            inputs = batch
            
        inputs = inputs.to(device)
        seq_len = inputs.size(0)  # sequence length для sequence-first формата
        mask = create_mask(seq_len)
        
        # Count non-padding tokens
        num_tokens = (inputs != 0).sum().item()  # assuming 0 is padding
        total_tokens += num_tokens
        
        # Measure forward pass time
        start = time.perf_counter()
        _ = model(inputs, mask)  # Only forward pass as per requirements
       
        end = time.perf_counter()
        
        times.append(end - start)
    
    # Convert to milliseconds and calculate statistics
    times = torch.tensor(times) * 1000
    
    return {
        "min": times.min().item(),
        "max": times.max().item(),
        "mean": times.mean().item(),
        "median": times.median().item(),
        "total_batches": len(times),
        "total_tokens": total_tokens,
        "tokens_per_second": total_tokens / times.sum().item() * 1000
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--mode', 
        type=str, 
        choices=['brain', 'big-brain', 'ultra-duper-big-brain'],
        default='ultra-duper-big-brain',
        help='Dataset mode to use'
    )
    args = parser.parse_args()
    
    # Map string argument to enum
    mode_map = {
        'brain': DataMode.BRAIN,
        'big-brain': DataMode.BIG_BRAIN,
        'ultra-duper-big-brain': DataMode.ULTRA_DUPER_BIG_BRAIN
    }
    
    run_epoch(mode_map[args.mode])


if __name__ == "__main__":
    main()