import os
import regex as re
from collections import Counter
from tqdm import tqdm

def token_replace(token, inverted_dict, out_list):
    # print(type(token))
    if(type(token) == type(43)):
        if token > 255:
            token_replace(inverted_dict[token][0], inverted_dict, out_list)
        else:
            out_list.append(token)
    else:
        for x in token:
            if x > 255:
                token_replace(inverted_dict[x], inverted_dict, out_list)
            else:
                out_list.append(x)
    return out_list

def bpe_trainer(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    token_id = 0
    vocab = {}
    merges = []    
    for token in special_tokens:
        vocab[token_id] = token.encode('utf-8')
        token_id += 1

    for i in range(256):
        vocab[token_id] = bytes([i])
        token_id += 1
    
    with open(input_path) as f:
        corpus = f.read()

    import re as builtin_re
    if special_tokens:
        pattern = "|".join(builtin_re.escape(token) for token in special_tokens)
        corpus_segments = builtin_re.split(f"({pattern})", corpus)
        corpus_segments = [segment for segment in corpus_segments if segment]
        # print(corpus_segments)
    else:
        corpus_segments = [corpus]
    
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    word_freq = Counter()
    for part in tqdm(corpus_segments):
        if part in special_tokens:
            continue
        else:
            pretokens = re.findall(PAT, part)
            for pretoken in pretokens:
                byte_tuple = tuple(bytes([b]) for b in pretoken.encode('utf-8'))
                word_freq[byte_tuple] += 1

    print(f"total unique words: {len(word_freq)}")
    print(f"Performing {vocab_size - token_id} merges")
    
    while token_id < vocab_size:
        new_dict = {}
        counts = {}
        for entry, freq in word_freq.items():
            # print(entry, freq)
            for pair in zip(entry, entry[1:]):
                counts[pair] = counts.get(pair, 0) + freq
        (chr1, chr2), freq = max(counts.items(), key= lambda item: (item[1], item[0]))
        # print(chr1, chr2)
        # print(sorted_counts)
        if len(counts) > 0:
            
            new_token = bytes(chr1) + bytes(chr2)
            vocab[token_id] = new_token
            merges.append((chr1, chr2))
            # print(f"looking for pair {chr1}, {chr2}")
            for entry in word_freq:
                # old_entry = entry
                # entry = list(entry)
                replace_entry = []
                i = 0
                while i < len(entry):
                    if (i < len(entry)- 1) and ((entry[i], entry[i+1]) == (chr1, chr2)):
                        replace_entry.append(new_token)
                        i+=2
                    else:
                        replace_entry.append(entry[i])
                        i+=1
                        # print(entry, old_entry)
                
                # print(replace_entry, entry)
                new_dict[tuple(replace_entry)] = word_freq[entry]
                # del new_dict[entry]
        token_id += 1

        word_freq = new_dict
        if (token_id + 1) % 50 == 0:
            print(f"Completed {token_id + 1} merges...")
    print(vocab)
    print(merges)
    return vocab, merges

if __name__ == '__main__':
    bpe_trainer("tests/fixtures/corpus.en", 1000, ["<|endoftext|>"])
    # bpe_trainer("sample.txt", 300, ["<|endoftext|>"])