from constants import SliceInfo

"""
def divide_shape(shape: tuple[int, ...], max_size: int) -> Iterable[SliceInfo | None]:
    if max_size <= 0: raise ValueError()
    if len(shape) == 0: return (None,)
    
    slice_size = reduce(lambda a, b: a*b, shape[1:], 1)
    size = slice_size * shape[0]
    
    
"""

