import sys
import argparse
import torch
from pathlib import Path
import pickle
from typing import TypeVar

import hv_functions
from utils import find_unique_path
from device import default_device

T = TypeVar('T')

def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog='GHRR with attention experiments',
        allow_abbrev=False
    )
    parser.add_argument("--test-GPU-mem", action="store_true")
    parser.add_argument("--times", type=int)
    return parser

def process_raw_file_path(raw: str | None) -> Path:
    if raw is None:
        raise ValueError("File path is None")
    
    res = find_unique_path(raw)
    return res

def record_gpu_management(fn: Callable[[], T], out: Path) -> T:
    torch.cuda.memory._record_memory_history()
    
    res = fn()
    
    snapshot = torch.cuda.memory._snapshot()
    torch.cuda.memory._record_memory_history(enabled=None)
    with out.open("wb") as f:
        pickle.dump(snapshot, f, protocol=4)
    
    return res

def main() -> None:
    args = get_parser().parse_args()
    
    if args.test_GPU_mem:
        print("Testing GPU memory management on hv_functions.fromfunction")
        
        out1_file = process_raw_file_path("out1.pickle")
        print(f"Output without management to {str(out1_file)}")
        out2_file = process_raw_file_path("out2.pickle")
        print(f"Output with management to {str(out2_file)}")
        
        times = args.times
        if times is None:
            def f1() -> None:
                ptrs = hv_functions.test__torch_fromfunction__no_management(lambda a, b, c: a + b + c, (10, 10, 10), device=default_device)
                torch.cuda.empty_cache()
                print("Pointer addresses (without management):", ptrs)
            
            def f2() -> None:
                ptrs = hv_functions.test__torch_fromfunction__with_management(lambda a, b, c: a + b + c, (10, 10, 10), device=default_device)
                torch.cuda.empty_cache()
                print("Pointer addresses (with management):", ptrs)
            
            record_gpu_management(f1, out=out1_file)
            record_gpu_management(f2, out=out2_file)
        else:
            def f1() -> None:
                for i in range(times):
                    hv_functions.test__torch_fromfunction__no_management(lambda a, b, c: a + b + c, (10, 10, 10), device=default_device)
                torch.cuda.empty_cache()
            
            def f2() -> None:
                for i in range(times):
                    hv_functions.test__torch_fromfunction__with_management(lambda a, b, c: a + b + c, (10, 10, 10), device=default_device)
                torch.cuda.empty_cache()
            
            record_gpu_management(f1, out=out1_file)
            record_gpu_management(f2, out=out2_file)
    else:
        print("Use --test-GPU-mem to test GPU memory management")

if __name__ == "__main__":
    main()
