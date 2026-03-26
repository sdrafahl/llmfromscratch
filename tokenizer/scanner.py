from pydantic.dataclasses import dataclass
from pydantic import RootModel
from pydantic import BaseModel, ConfigDict, Field, model_validator
from typing import Dict, List, Optional
import re

class Vocabulary(RootModel):
    model_config = ConfigDict(frozen=True)
    root: Dict[str, int]
    def get(self, key: str) -> Optional[int]:
        return self.root.get(key)

    def add(self, key: str, v: int):
        newSelf = self.model_copy(
            update={"root": self.root | {key: v}}
        )
        return newSelf

    def length(self) -> int:
        return len(self.root)

    def __contains__(self, key: str) -> bool:
        return key in self.root

class VocabularyIDMap(RootModel):
    model_config = ConfigDict(frozen=True)
    root: Dict[int, str]
    def get(self, key: int) -> Optional[str]:
        return self.root.get(key)

    def add(self, key: int, v: str):
        newSelf = self.model_copy(
            update={"root": self.root | {key: v}}
        )
        return newSelf

    def __contains__(self, key: int) -> bool:
        return key in self.root


@dataclass(config={'frozen': True})
class ScanResult:
    tokenToId: Vocabulary
    idLookup: VocabularyIDMap
    cleanedText: List[str]

# In some cases you may want to keep the white spaces, like for python code because indentation matters
def scan(text: str) -> ScanResult:
    result = re.split(r'([,.:;?_!"()\]]|--|\s)', text)
    cleanedResult = [item for item in result if item.strip()]
    # Unknown and end of text
    vocab = Vocabulary({
        token: i 
        for i, token in enumerate(sorted(set(cleanedResult)))
    })
    withAdd = vocab.add("<|unk|>", -1)
    withEndOfText = withAdd.add("<|endoftext|>", -2)
    idLookup = VocabularyIDMap({
        i: token
        for i, token in enumerate(sorted(set(cleanedResult)))
    })
    idLookupwithAdd = idLookup.add(-1, "<|unk|>")
    idLookupWithEndOfText = idLookupwithAdd.add(-2, "<|endoftext|>")
    return ScanResult(tokenToId = withEndOfText, idLookup = idLookupWithEndOfText, cleanedText = cleanedResult)

## todo refactor special codes
def encode(scan: ScanResult) -> List[int]:
    ids = [scan.tokenToId.get(t) or -1 for t in scan.cleanedText]
    return ids

def decode(scan: ScanResult, ids: List[int]) -> str:
    tokens = [scan.idLookup.get(i) or "<|unk|>" for i in ids]
    joinedText = " ".join(tokens)
    # replace spaces before the specified punctuations that the join added
    text = re.sub(r'\s+([,.?!"()\'])', r'\1', joinedText)
    return text





