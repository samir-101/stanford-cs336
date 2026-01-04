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
    for token in special_tokens:
        vocab[token_id] = token
        token_id += 1

    for i in range(256):
        vocab[token_id] = bytes([i])
        token_id += 1
    
    with open(input_path) as f:
        corpus = f.read()
    pattern = "|".join(re.escape(token) for token in special_tokens)
    corpus_segments = re.split(f"({pattern})", corpus)
    corpus_segments = [segment for segment in corpus_segments if segment]
    
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    corpus = re.findall(PAT, corpus)
    word_freq = Counter()
    for part in corpus_segments:
        if part in special_tokens:
            continue
        else:
            pretokens = re.findall(PAT, part)
            for pretoken in pretokens:
                byte_tuple = tuple(bytes([b]) for b in pretoken.encode('utf-8'))
                word_freq[byte_tuple] += 1
    print(f"total unique words: {word_freq}")
    
    # for new_id in range(start_token, vocab_size):
    #     counts = {}
    #     for entry in tokens:
    #         for pair in zip(entry, entry[1:]):
    #             counts[pair] = counts.get(pair,0)+1
    #     sorted_counts = {v: k for k, v in sorted(counts.items(), key=lambda item: item[1], reverse = True)}
    #     if len(sorted_counts.keys()) > 0:
    #         chr1, chr2 = sorted_counts.get(max(sorted_counts.keys()))
    #         merges[(chr1, chr2)] = new_id
    #         for entry in tokens:
    #             for i, char in enumerate(entry):
    #                 if (len(entry) >= 2) & (i < len(entry)-1):
    #                     if (entry[i], entry[i+1]) == (chr1, chr2):
    #                         entry[i] = new_id
    #                         entry[i+1:] = entry[i+2:]
    #         new_id += 1
    # vocabs = []
    # for vocab in merges.items():
    #     vocabs.append(vocab[0])

    # entry = {}

    # inverted_merges = {v: k for k, v in sorted(merges.items(), key=lambda item: item[1], reverse = True)}

    # for merge in inverted_merges.items():
    #     sub_entry = []
    #     token_replace(merge[1], inverted_merges, sub_entry)
    #     entry[merge[0]] = sub_entry
    # print(entry)
    # return entry, vocabs

if __name__ == '__main__':
    bpe_trianer("sample.txt", 300, [])