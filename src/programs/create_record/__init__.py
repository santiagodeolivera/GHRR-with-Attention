from pathlib import Path

from operation_manager import OperationManagerRecord

def func(path: Path) -> None:
    OperationManagerRecord(path).setup()

