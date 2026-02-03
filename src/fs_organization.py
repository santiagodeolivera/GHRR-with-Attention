from pathlib import Path

class FsOrganizer:
	__root: Path
	
	def __init__(self, root: Path):
		self.__root = root
	
	def setup(self):
		(self.__root / "encodings").mkdir(parents=True, exist_ok=True)
		(self.__root / "base").mkdir(parents=True, exist_ok=True)
		(self.__root / "tudataset").mkdir(parents=True, exist_ok=True)
		(self.__root / "model").mkdir(parents=True, exist_ok=True)
	
	@property
	def tudataset(self) -> Path:
		return self.__root / "tudataset"
		
	@property
	def hv_encodings(self) -> Path:
		return self.__root / "encodings"
	
	def hv_encoding_of(self, id: int) -> Path:
		return self.__root / f"encodings/{id}.pt"
	
	@property
	def query_encodings(self) -> Path:
		return self.__root / "base/query.pt"
	
	@property
	def key_encodings_1(self) -> Path:
		return self.__root / "base/key1.pt"
	
	@property
	def key_encodings_2(self) -> Path:
		return self.__root / "base/key2.pt"
	
	@property
	def value_encodings(self) -> Path:
		return self.__root / "base/value.pt"
	
	@property
	def ids_to_labels(self) -> Path:
		return self.__root / "ids_to_labels.json"
	
	@property
	def train_and_test_sets_distribution(self) -> Path:
		return self.__root / "sets_distribution.json"
	
	@property
	def model(self):
		return self.__root / "model"
	
	@property
	def test_results(self):
		return self.__root / "test_results.json"
