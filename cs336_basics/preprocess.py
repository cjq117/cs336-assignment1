import os
import json
import numpy as np
from typing import List,Dict
from cs336_basics.tokenizer import BPETokenizer

def bytes_to_unicode():
    """
    返回一个字节到可见 Unicode 字符的映射字典。
    该映射确保所有 256 个字节值都有对应的可见字符
    """
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))

def load_train_tokenizer(vocab_path:str,merges_path:str,special_tokens:list[str]):
    print(f"正在从{os.path.dirname(vocab_path)} 加载分词器...")
    byte_encoder=bytes_to_unicode()
    byte_decoder={v:k for k,v in byte_encoder.items()}
    with open(vocab_path,"r",encoding="utf_8")as f:
        vocab_raw=json.load(f)
        vocab={
            int(k):bytes([byte_decoder[c] for c in v])
            for v,k in vocab_raw.items()
        }

    merges=[]
    with open(merges_path,"r",encoding="utf_8")as f:
      for line in f:
           line=line.rstrip('\n')
           if not line:continue
           parts =line.split(' ')
           if len(parts)==2:
               p1=bytes([byte_decoder[c] for c in parts[0]])
               p2=bytes([byte_decoder[c] for c in parts[1]])
               merges.append((p1,p2))
    print(f"成功加载词表，当前词表规模：{len(vocab)}")
    return BPETokenizer(vocab,merges,special_tokens)

def process_corpus(input_txt: str, output_bin: str, tokenizer: BPETokenizer, chunk_size_mb: int = 50):
    # 1. 内部生成器：负责按块从硬盘读取文本
    def file_chunk_generator(file_path, size):
        with open(file_path, "r", encoding="utf-8") as f:
            while True:
                chunk = f.read(size)
                if not chunk:
                    break
                yield chunk

    # 2. 检查与准备
    if not os.path.exists(input_txt):
        raise FileNotFoundError(f"找不到语料文件: {input_txt}")

    chunk_size = 1024 * 1024 * chunk_size_mb
    if os.path.exists(output_bin):
        os.remove(output_bin)

    print(f"使用 encode_iterable 开始流式预处理...")

    # 3. 核心流式逻辑
    # 创建文本块生成器
    chunks = file_chunk_generator(input_txt, chunk_size)
    # 丢进 encode_iterable，得到一个“不停吐出 ID”的生成器
    token_stream = tokenizer.encode_iterable(chunks)

    total_tokens = 0
    # 为了高效写入磁盘，依然需要一个小缓存（Buffer）
    # 每积攒 100 万个 Token 写入一次硬盘
    write_batch_size = 1_000_000
    token_buffer = []

    with open(output_bin, "ab") as f_out:
        for token_id in token_stream:
            token_buffer.append(token_id)
            if len(token_buffer) >= write_batch_size:
                np_ids = np.array(token_buffer, dtype=np.uint16)
                f_out.write(np_ids.tobytes())
                total_tokens += len(token_buffer)
                token_buffer = []

        # 处理最后剩余的 buffer
        if token_buffer:
            np_ids = np.array(token_buffer, dtype=np.uint16)
            f_out.write(np_ids.tobytes())
            total_tokens += len(token_buffer)

    print(f"处理完成! 总 Token: {total_tokens}")
       

def main():
    BASE_DIR="D:/keyan/cs336-assignment1/bpe_test_output"
    input_file="C:/Users/a7655/AppData/Local/Temp/bpe_check_test.txt"
    output_file="D:/keyan/cs336-assignment1/bpe_test_output/check.bin"

    vocab_json=os.path.join(BASE_DIR,"vocab.json")
    merges_txt=os.path.join(BASE_DIR,"merges.txt")
    special_tokens=["<|endoftext|>"]
    #加载分词器
    tokenizer=load_train_tokenizer(vocab_json,merges_txt,special_tokens)
    #执行数据清洗和预处理
    process_corpus(input_file,output_file,tokenizer)
    ids = np.fromfile(output_file, dtype=np.uint16).tolist()
    decoded_text = tokenizer.decode(ids)
    decoded_file = os.path.join(BASE_DIR, "check_decoded.txt")
    with open(decoded_file, "w", encoding="utf-8", newline="") as f:
        f.write(decoded_text)
if __name__=="__main__":
    main()