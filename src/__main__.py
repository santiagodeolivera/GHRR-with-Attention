import torch

from tests import all_tests
from constants import element_type

def main():
    if not torch.cuda.is_available():
        raise Exception("CUDA is not available")
    
    all_tests()

if __name__ == "__main__":
    main()

