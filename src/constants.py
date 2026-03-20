import os

D_str = os.environ.get("DIMENSIONS", None)
if D_str is None:
    raise Exception("Env var not found: 'DIMENSIONS'")

D = int(D_str)
if D <= 0:
    raise Exception(f"Invalid value for 'DIMENSIONS': {repr(D_str)}")

m_str = os.environ.get("MATRIX_SIZE", None)
if m_str is None:
    raise Exception("Env var not found: 'MATRIX_SIZE'")

m = int(m_str)
if m <= 0:
    raise Exception(f"Invalid value for 'MATRIX_SIZE': {repr(m_str)}")


F2_SAMPLE_SIZE = 20

__all__ = ["D", "m", "F2_SAMPLE_SIZE"]

