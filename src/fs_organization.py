from dataclasses import dataclass
from pathlib import Path

@dataclass
class FsConfig:
	base_dir: str
	encodings_dir: str
	tudataset_dir: str
	model_dir: str
	dist_file: str
	ids_to_labels_file: str
	result_file: str
	
	@staticmethod
	def default() -> "FsConfig":
		return FsConfig( \
			base_dir = "base", \
			encodings_dir = "encodings", \
			tudataset_dir = "tudataset", \
			model_dir = "model", \
			dist_file = "sets_distribution.json", \
			ids_to_labels_file = "ids_to_labels.json", \
			result_file = "test_results.json" \
		)

class FsOrganizer:
	__root: Path
	config: FsConfig
	
	def __init__(self, root: Path, config: FsConfig | None = None):
		self.__root = root
		self.config = FsConfig.default() if config is None else None
	
	def setup(self):
		self.base_encodings.mkdir(parents=True, exist_ok=True)
		self.hv_encodings.mkdir(parents=True, exist_ok=True)
		self.tudataset.mkdir(parents=True, exist_ok=True)
		self.model.mkdir(parents=True, exist_ok=True)
		
		self.ids_to_labels.parent.mkdir(parents=True, exist_ok=True)
		self.train_and_test_sets_distribution.parent.mkdir(parents=True, exist_ok=True)
		self.test_results.parent.mkdir(parents=True, exist_ok=True)
	
	@property
	def root(self) -> Path:
		return self.__root
	
	@property
	def tudataset(self) -> Path:
		return self.__root / self.config.tudataset_dir
		
	@property
	def hv_encodings(self) -> Path:
		return self.__root / self.config.encodings_dir
	
	def hv_encoding_of(self, id: int) -> Path:
		return self.hv_encodings / f"{id}.pt"
	
	@property
	def base_encodings(self) -> Path:
		return self.__root / self.config.base_dir
	
	@property
	def query_encodings(self) -> Path:
		return self.base_encodings / "query.pt"
	
	@property
	def key_encodings_1(self) -> Path:
		return self.base_encodings / "key1.pt"
	
	@property
	def key_encodings_2(self) -> Path:
		return self.base_encodings / "key2.pt"
	
	@property
	def value_encodings(self) -> Path:
		return self.base_encodings / "value.pt"
	
	@property
	def ids_to_labels(self) -> Path:
		return self.__root / self.config.ids_to_labels_file
	
	@property
	def train_and_test_sets_distribution(self) -> Path:
		return self.__root / self.config.dist_file
	
	@property
	def model(self):
		return self.__root / self.config.model_dir
	
	@property
	def test_results(self):
		return self.__root / self.config.result_file
