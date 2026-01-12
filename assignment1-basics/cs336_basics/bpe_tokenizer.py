import os
import regex as re
from collections import Counter
from tqdm import tqdm
from typing import BinaryIO
import multiprocessing


def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))

def pretokenization(corpus, special_tokens):
    import re as builtin_re
    if special_tokens:
        pattern = "|".join(builtin_re.escape(token) for token in special_tokens)
        corpus_segments = builtin_re.split(f"({pattern})", corpus)
        corpus_segments = [segment for segment in corpus_segments if segment]
    else:
        corpus_segments = [corpus]
    # len_size = 0
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    word_freq = Counter()
    for part in tqdm(corpus_segments):
        if part in special_tokens:
            continue
        else:
            pretokens = re.findall(PAT, part)
            # len_size += len(pretokens)
            for pretoken in pretokens:
                byte_tuple = tuple(bytes([b]) for b in pretoken.encode('utf-8'))
                word_freq[byte_tuple] += 1
    # print(len_size)
    return word_freq

def bpe_trainer(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    # 1. Initialize Vocab
    vocab = {}
    merges = []
    token_id = 0
    
    # Special tokens first
    for token in special_tokens:
        vocab[token_id] = token.encode('utf-8')
        token_id += 1
    
    # Initial byte tokens
    for i in range(256):
        vocab[token_id] = bytes([i])
        token_id += 1
        
    # 2. Pre-tokenization / Reading
    results = []
    with open(input_path, "rb") as f:
        f.seek(0, os.SEEK_END)
        size = f.tell()
        f.seek(0)
        
        if size < 1024 * 1024: # 1MB
             txt = f.read().decode("utf-8", errors="ignore")
             word_freq = pretokenization(txt, special_tokens)
        else:
             processed_chunks = []
             num_processes = 8
             boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")
             chunks = []
             for start, end in zip(boundaries[:-1], boundaries[1:]):
                 f.seek(start)
                 chunks.append((f.read(end - start).decode("utf-8", errors="ignore"), special_tokens))
             
             with multiprocessing.Pool(processes=num_processes) as pool:
                for chunk in chunks:
                #  processed_chunks = pool.starmap(pretokenization, chunks)
                    processed_chunks.append(pretokenization(chunk))
                 
             word_freq = Counter()
             for chunk_freq in processed_chunks:
                 word_freq.update(chunk_freq)

    print(f"total unique words: {len(word_freq)}")
    
    # 3. Build data structures for efficient merging
    byte_to_id = {bytes([i]): i + len(special_tokens) for i in range(256)}
    
    words = []
    counts = []
    
    pair_stats = {}
    pair_locations = {}
    
    for i, (word_bytes_tuple, count) in enumerate(word_freq.items()):
        ids = [byte_to_id[b] for b in word_bytes_tuple]
        words.append(ids)
        counts.append(count)
        
        for j in range(len(ids) - 1):
            pair = (ids[j], ids[j+1])
            pair_stats[pair] = pair_stats.get(pair, 0) + count
            if pair not in pair_locations:
                pair_locations[pair] = set()
            pair_locations[pair].add(i)

    print(f"Performing {vocab_size - token_id} merges")
    
    while token_id < vocab_size:
        if not pair_stats:
            break
            
        # Tie-breaking: maximize frequency, then maximize the byte representation of the pair
        # to match reference implementation which uses (bytes1, bytes2) as keys.
        best_pair = max(
            pair_stats, 
            key=lambda p: (pair_stats[p], vocab[p[0]], vocab[p[1]])
        )
        
        if pair_stats[best_pair] < 1:
            break
            
        new_token_bytes = vocab[best_pair[0]] + vocab[best_pair[1]]
        vocab[token_id] = new_token_bytes
        merges.append((vocab[best_pair[0]], vocab[best_pair[1]]))
        
        occurrences = pair_locations.pop(best_pair)
        del pair_stats[best_pair]
        
        for word_idx in occurrences:
            w_ids = words[word_idx]
            w_count = counts[word_idx]
            
            # 1. Remove old pairs statistics for this word
            for j in range(len(w_ids) - 1):
                p = (w_ids[j], w_ids[j+1])
                if p == best_pair: continue
                pair_stats[p] -= w_count
                if pair_stats[p] == 0:
                    del pair_stats[p]
                if p in pair_locations:
                    pair_locations[p].discard(word_idx)
                    if not pair_locations[p]:
                        del pair_locations[p]

            # 2. Construct new word
            new_w_ids = []
            i = 0
            while i < len(w_ids):
                if i < len(w_ids) - 1 and w_ids[i] == best_pair[0] and w_ids[i+1] == best_pair[1]:
                    new_w_ids.append(token_id)
                    i += 2
                else:
                    new_w_ids.append(w_ids[i])
                    i += 1
            
            words[word_idx] = new_w_ids
            
            # 3. Add new pairs statistics
            for j in range(len(new_w_ids) - 1):
                p = (new_w_ids[j], new_w_ids[j+1])
                pair_stats[p] = pair_stats.get(p, 0) + w_count
                if p not in pair_locations:
                    pair_locations[p] = set()
                pair_locations[p].add(word_idx)

        token_id += 1
        if (token_id + 1) % 50 == 0:
            print(f"Completed {token_id + 1} merges...")
            
    return vocab, merges

if __name__ == '__main__':
    bpe_trainer("data/owt_train.txt", 32000, ["<|endoftext|>"])
    # bpe_trainer("data/TinyStoriesV2-GPT4-train.txt", 10000, ["<|endoftext|>"])
    # bpe_trainer("tests/fixtures/corpus.en", 1000, ["<|endoftext|>"])
    # bpe_trainer("sample.txt", 300, ["<|endoftext|>"])