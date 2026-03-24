from pathlib import Path
from typing import Sequence
import json
import torch

from hv_proxy import HVProxy, iter_from_fs as proxies_from_fs
from fs_organization import FsOrganizer
from utils import take_random_from_list, approximation
from hv_functions import UpperTensorFunctionsManager
from constants import F2_SAMPLE_SIZE
from fn_context import FnContext

def get_split_ids_by_label(root: FsOrganizer) -> dict[int, list[HVProxy]]:
    proxies = tuple(proxies_from_fs(root, range(188)))
    labels = tuple(set(proxy.label for proxy in proxies))
    
    res = dict( \
        (label, list(proxy for proxy in proxies if proxy.label == label)) \
        for label in labels \
    )
    
    return res

# Compare similarities of HVs in random samples of different classes
def func(ctx: FnContext) -> None:
    root = ctx.fs
    functions = ctx.functions
    root.config.result_file = "hv_comparison.json"
    
    samples0: dict[int, list[HVProxy]] = get_split_ids_by_label(root)
    samples: dict[int, tuple[HVProxy, ...]] = {k: take_random_from_list(v, F2_SAMPLE_SIZE) for k, v in samples0.items()}
    
    hv1: torch.Tensor | None = None
    hv2: torch.Tensor | None = None
    raw_data: dict[str, dict[str, float]] = dict()
    approximations: dict[str, str] = dict()
    for label1 in samples.keys():
        for label2 in samples.keys():
            if label1 > label2: continue
            mid_raw_data: dict[str, float] = dict()
            for proxy1 in samples[label1]:
                for proxy2 in samples[label2]:
                    if proxy1.id > proxy2.id: continue
                    hv1 = proxy1.get_hv(out=hv1)
                    hv2 = proxy2.get_hv(out=hv2)
                    (tmp,) = functions.tmp_gen.new_paths(1)
                    similarity = functions.normalized_similarity(hv1, hv2, out=tmp).tensor()
                    mid_raw_data[f"{proxy1.id},{proxy2.id}"] = similarity.item()
            approximations[f"{label1},{label2}"] = approximation(tuple(mid_raw_data.values()))
            raw_data[f"{label1},{label2}"] = mid_raw_data
    
    json_res = json.dumps({"approximations": approximations, "raw_data": raw_data})
    root.result_file.write_text(json_res)
 
__all__ = ["func"]

