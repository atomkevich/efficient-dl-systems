from typing import Optional
import random
from typing import List, Tuple
import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import Sampler, IterableDataset
from torchtext.data.utils import get_tokenizer
import random
from collections import defaultdict, deque
from typing import Iterator, List, Optional
from torch.utils.data import Sampler

# We will use torchtext's basic_english tokenizer instead of this custom implementation
# from torchtext.data.utils import get_tokenizer


MAX_LENGTH = 640


class BrainDataset(Dataset):
    """WikiText dataset with naive padding - everything padded to fixed max_length"""

    def __init__(self, data_path: str, max_length: int = MAX_LENGTH):
        self.max_length = max_length
        self.tokenizer = get_tokenizer("basic_english")

        # Create vocabulary from the dataset
        word_list = set()
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line := line.strip():
                    tokens = self.tokenizer(line)
                    word_list.update(tokens)

        # 0 — PAD, начинаем индексацию с 1
        self.word_to_idx = {word: idx + 1 for idx, word in enumerate(sorted(word_list))}
        self.vocab_size = len(self.word_to_idx) + 1

        # Load and tokenize all texts
        self.tokenized_texts = []
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line := line.strip():
                    # Tokenize text
                    tokens = [self.word_to_idx[word] for word in self.tokenizer(line)]
                    # Truncate if needed
                    tokens = tokens[:max_length]
                    # Pad to max_length
                    if len(tokens) < max_length:
                        tokens.extend([0] * (max_length - len(tokens)))  # 0 is padding token
                    self.tokenized_texts.append(tokens)

    def __len__(self):
        return len(self.tokenized_texts)

    def __getitem__(self, idx: int):
        """
        Return input and target tensors for next-token prediction.
        Inputs:  sequence[:-1]
        Targets: sequence[1:]
        """
        seq = torch.tensor(self.tokenized_texts[idx], dtype=torch.long)

        # inputs: всё кроме последнего
        inputs = seq[:-1]
        # targets: всё кроме первого
        targets = seq[1:]

        return inputs, targets


class BigBrainDataset(Dataset):
    """
    WikiText dataset that defers padding to collate_fn (BIG BRAIN).
    - Читает строки
    - Токенизирует
    - Мапит в индексы
    - Обрезает до MAX_LENGTH
    - НЕ паддит (это делает collate_fn)
    Возвращает: tuple(raw_text: str, ids: torch.Tensor)
    """
    def __init__(self, data_path: str, max_length: int = MAX_LENGTH) -> None:
        self.max_length = max_length
        self.tokenizer = get_tokenizer("basic_english")

        # строим словарь по всему файлу
        vocab = set()
        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                vocab.update(self.tokenizer(line))

        # 0: PAD, 1: UNK
        offset = 2
        self.word_to_idx = {w: i + offset for i, w in enumerate(sorted(vocab))}
        self.vocab_size = len(self.word_to_idx) + offset

        # храним кортежи (сырой текст, ids_tensor)
        self.samples: List[Tuple[str, torch.Tensor]] = []
        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                tokens = self.tokenizer(line)
                ids = [self.word_to_idx.get(w, 1) for w in tokens]
                ids = ids[: self.max_length]
                ids_t = torch.as_tensor(ids, dtype=torch.long)
                self.samples.append((line, ids_t))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[str, torch.Tensor]:
        return self.samples[idx]
    

class UltraDuperBigBrainDataset(BigBrainDataset):
    """
    UltraDuperBigBrainDataset наследует BigBrainDataset.
    Отличие:
      - хранит длины каждого примера,
      - возвращает (raw_text, ids, length).
    """

    def __init__(self, data_path: str, max_length: int = MAX_LENGTH) -> None:
        # вызываем конструктор родителя -> он создаёт self.samples = [(raw, ids), ...]
        super().__init__(data_path, max_length)

        # добавим длины
        self.lengths: List[int] = [ids.size(0) for (_, ids) in self.samples]

    def __getitem__(self, idx: int) -> Tuple[str, torch.Tensor]:
        raw, ids = self.samples[idx]
        return raw, ids

def collate_fn(
    batch: List[Tuple[str, torch.Tensor]], max_length: Optional[int] = MAX_LENGTH
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Pad each sequence of the incoming sequences list
    :param batch: list of (raw_text, ids_tensor)
    :param max_length: максимум длины (для Brain); здесь ограничивает верхнюю границу L
    :return: (inputs, targets) формы [B, L],
             где targets — next-token (сдвиг на 1), последний токен строки = PAD (0)
    """
    seqs = [ids for (_, ids) in batch]
    raw_max = max(s.size(0) for s in seqs)
    L = min(raw_max, max_length)
    B = len(seqs)

    inputs = torch.zeros((B, L), dtype=torch.long)
    targets = torch.full((B, L), fill_value=-100, dtype=torch.long)

    for i, s in enumerate(seqs):
        n = min(s.size(0), L)
        inputs[i, :n] = s[:n]
        if n > 1:
            targets[i, :n-1] = s[1:n]

    # [B, L] -> [L, B]
    return inputs.transpose(0, 1), targets.transpose(0, 1)

def collate_fn_ultra(
    batch: List[Tuple[str, torch.Tensor, int]], max_length: Optional[int] = MAX_LENGTH
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Pad each sequence of the incoming sequences list
    :param batch: list of (raw_text, ids_tensor, length)
    :param max_length: максимум длины (для Brain); здесь ограничивает верхнюю границу L
    :return: (inputs, targets) формы [B, L],
             где targets — next-token (сдвиг на 1), последний токен строки = PAD (0)
    """
    seqs = [ids for (_, ids, _) in batch]
    raw_max = max(s.size(0) for s in seqs)
    L = min(raw_max, max_length)
    B = len(seqs)

    inputs = torch.zeros((B, L), dtype=torch.long)
    targets = torch.full((B, L), fill_value=-100, dtype=torch.long)

    for i, s in enumerate(seqs):
        n = min(s.size(0), L)
        inputs[i, :n] = s[:n]
        if n > 1:
            targets[i, :n-1] = s[1:n]

    # [B, L] -> [L, B]
    return inputs.transpose(0, 1), targets.transpose(0, 1)



class UltraDuperBigBrainBatchSampler(Sampler[List[int]]):
    """
    Возвращает списки индексов для DataLoader (batch_sampler=...).
    Условия:
      - в батче max_len - min_len ≤ k,
      - случайная выборка,
      - __init__ — O(n),
      - __iter__ — O(batch_size) (число возможных длин ограничено MAX_LENGTH).
    """
    def __init__(
        self,
        dataset: UltraDuperBigBrainDataset,
        batch_size: int,
        k: int = 5,
        seed: Optional[int] = 42,
    ) -> None:
        super().__init__(dataset)
        self.batch_size = batch_size
        self.k = k
        self.seed = seed

        # O(n): длина -> список индексов
        buckets = defaultdict(list)
        for idx, L in enumerate(dataset.lengths):
            buckets[L].append(idx)

        self.buckets = {L: deque(self._shuffle(lst)) for L, lst in buckets.items()}
       
        self.lengths_sorted = sorted(self.buckets.keys())
    
        self.alive = {L for L, dq in self.buckets.items() if dq}

    def _shuffle(self, lst: List[int]) -> List[int]:
        rng = random.Random(self.seed)
        rng.shuffle(lst)
        return lst

    def __iter__(self) -> Iterator[List[int]]:
        # отдельный RNG на итерацию (эпоху) — можно варьировать seed для reshuffle по эпохам
        rng = random.Random(self.seed)
        buckets = {L: deque(dq) for L, dq in self.buckets.items()}  # локальные копии указателей
        alive = {L for L, dq in buckets.items() if dq}
        lengths_sorted = self.lengths_sorted
        B, k = self.batch_size, self.k

        def pick_anchor() -> Optional[int]:
            return rng.choice(tuple(alive)) if alive else None

        while alive:
            L = pick_anchor()
            if L is None:
                break

            # окно длин [L, L+k] ∩ alive
            window = [ℓ for ℓ in lengths_sorted if (L <= ℓ <= L + k) and (ℓ in alive)]
            if not window:
                if not buckets[L]:  # выкинуть опустевшую длину
                    alive.discard(L)
                continue

            batch: List[int] = []
            while len(batch) < B and window:
                ℓ = rng.choice(window)
                idx = buckets[ℓ].popleft()
                batch.append(idx)
                if not buckets[ℓ]:
                    alive.discard(ℓ)
                    window = [x for x in window if x != ℓ]

            if batch:
                yield batch

    def __len__(self) -> int:
        total = sum(len(dq) for dq in self.buckets.values())
        return (total + self.batch_size - 1) // self.batch_size