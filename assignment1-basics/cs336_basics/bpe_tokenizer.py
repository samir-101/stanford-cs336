import os
import regex as re
from collections import Counter

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

def bpe_trianer(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    token_id = 0
    vocab = {}
    merges = []
    corpus_words_list = []
    

    for i in range(256):
        vocab[token_id] = bytes([i])
        token_id += 1

    for token in special_tokens:
        vocab[token_id] = token
        token_id += 1
    
    with open(input_path) as f:
        corpus = f.read()

    import re as builtin_re
    if special_tokens:
        pattern = "|".join(builtin_re.escape(token) for token in special_tokens)
        corpus_segments = builtin_re.split(f"({pattern})", corpus)
        corpus_segments = [segment for segment in corpus_segments if segment]
    else:
        corpus_segments = [corpus]
    
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    # corpus = re.findall(PAT, corpus)
    word_freq = Counter()
    for part in corpus_segments:
        # print(f"words are : {part}")
        if part in special_tokens:
            continue
        else:
            sub_list = []
            pretokens = re.findall(PAT, part)
            
            for pretoken in pretokens:
                byte_tuple = tuple(bytes([b]) for b in pretoken.encode('utf-8'))
                sub_list.append(list(bytes([b]) for b in pretoken.encode('utf-8')))
                word_freq[byte_tuple] += 1
            corpus_words_list.append(sub_list)

    print(f"total unique words: {len(word_freq)}")
    
    
    while token_id < vocab_size:
        counts = {}
        for entry in corpus_words_list:
            # print(entry)
            for word in entry:
                # entry = list(entry)
                for pair in zip(word, word[1:]):
                    # print(pair)
                    counts[pair] = counts.get(pair,0)+1
            sorted_counts = {k:v for k,v in sorted(counts.items(), key= lambda item: (item[1], item[0]), reverse = True)}
        if len(sorted_counts) > 0:
            chr1, chr2 = next(iter(sorted_counts))
            # merges.append((bytes(chr1)), bytes(chr2))
            
            new_token = bytes(chr1) + bytes(chr2)

            vocab[token_id] = new_token
            # print(f"{chr1} + {chr2} == {new_token}")
            merges.append((chr1, chr2))

            for entry in corpus_words_list[0]:
                entry = entry
                for i, char in enumerate(entry):
                    if (len(entry) >= 2) & (i < len(entry)-1):
                        if (entry[i], entry[i+1]) == (chr1, chr2):
                            entry[i] = new_token
                            entry[i+1:] = entry[i+2:]
            token_id += 1
        if (token_id + 1) % 50 == 0:
            print(f"Completed {token_id + 1} merges...")
    print(merges)
    print(vocab)
    return vocab, merges

if __name__ == '__main__':
    # bpe_trianer("tests/fixtures/tinystories_sample_5M.txt", 1000, ["<|endoftext|>"])
    bpe_trianer("sample.txt", 270, ["<|endoftext|>"])