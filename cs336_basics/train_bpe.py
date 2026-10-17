import os
from collections import defaultdict,Counter
#Counter:自动统计元素出现的次数
#defaultdict:字典中没有某一个键，自动提供初始值
import regex as re
import json

def train_bpe(
    input_path: str|os.PathLike,
    vocab_size: int,#词表大小
    special_tokens: list[str],
) -> tuple[dict[int,bytes],list[tuple[bytes,bytes]]]:
    #函数返回:
    #1.vocab:dict[int,bytes]
    #2.merges:list[tuple[bytes,bytes]]
    #tuple:把这两个部分一起返回

    #初始化词表
    vocab={i:bytes([i]) for i in range(256)}
    #计算还需要合并的次数
    num_merges=vocab_size-256-len(special_tokens)
    #打开文件，把内容读出来放在text里面
    with open(input_path,"r",encoding="utf_8") as f:
        text=f.read()
    #处理文本（把特殊tokens删掉）
    if special_tokens:
    #首先把special_tokens中的特殊字符转义一下(用re.escape())，再用|连接起来
        special_regex="|".join(re.escape(t) for t in special_tokens)
    #用re.split()把text分割开来，基于特殊字符。用（）把special_regex括起来，这样，分割之后返回的列表里面也会包含特殊字符
        parts=re.split(f"({special_regex})",text)
        #train_segments就是把parts中的特殊字符删掉的部分
        train_segments=[p for p in parts if p not in special_tokens]
    else:
        train_segments=[text]
    #使用GPT-2的BPE预分词正则表达式
    gpt2_pat = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

    #raw_counts类似于下面这种
    '''raw_counts = {
    (b'H', b'i'): 50,
    (b' ', b't', b'h', b'e', b'r', b'e'): 100,
    (b'!'): 50,
    (b'\xe4', b'\xbd', b'\xa0', b'\xe5', b'\xa5', b'\xbd'): 20, # 你好
}'''
    raw_counts=Counter()
    #在train_segments里面找每一个分割之后的小片段
    for segment in train_segments:
    #在小片段里面找到所有单词
        words=gpt2_pat.findall(segment)
        for word in words:
           raw_counts[tuple(bytes([b]) for b in word.encode("utf_8"))]+=1
    #words_list和counts_list存储raw_counts对应的字节序列和出现次数
    words_list=[]
    counts_list=[]
    for word_tuple,freq in raw_counts.items():
        words_list.append(list(word_tuple))
        counts_list.append(freq)
    #创建一个字典，来存储类似于这样的结构：{(byte_a,byte_b):frequency}
    stats=defaultdict(int)
    #倒排索引pair->{包含这个pair的单词在words_list下标集合}，用于快速找到需要更新的单词
    indices=defaultdict(set)
    #计算每一对pair的stats和indices
    for idx,word in enumerate(words_list):
        freq=counts_list[idx]
        for i in range(len(word)-1):
            pair=(word[i],word[i+1])
            stats[pair]+=freq
            indices[pair].add(idx)

    #定义merges(表示哪两个pair合并)
    merges=[]
    #for循环遍历迭代次数
    for _ in range(num_merges):
        if not stats:
            break
        #找到stats里面出现频率最高，且字典序最大的pair
        #先比较x[1]频率，如果频率相等再比较x[0]字典序
        best_pair=max(stats.items(),key=lambda x:(x[1],x[0]))[0]
        if stats[best_pair]<=0:
            break
        merges.append(best_pair)
        #合并之后的序列比如:new_token=lo""
        new_token=best_pair[0]+best_pair[1]
        #用indices找到所有包含best_pair单词的下标
        relevant_indices=list(indices[best_pair])
        #遍历这些单词并修改pair的indices和stats
        for idx in relevant_indices:
            word=words_list[idx]
            freq=counts_list[idx]
            #找到这个单词里面的每一个pair
            i=0
            while i<len(word)-1:
                #如果if成立则在word中就匹配到了
                if word[i]==best_pair[0] and word[i+1]==best_pair[1]:
                    #匹配到之后，更新这个best_pair的左邻居和右邻居
                    #i=0时，是没有左邻居的
                    if i>0:
                        prev_pair=(word[i-1],word[i])
                        stats[prev_pair]-=freq
                        if stats[prev_pair]==0:
                            del stats[prev_pair]
                            #右邻居
                    if i<len(word)-2:
                        next_pair=(word[i+1],word[i+2])
                        stats[next_pair]-=freq
                        if stats[next_pair]==0:
                            del stats[next_pair]
                    word[i]=new_token
                    del word[i+1]
                    #更新左邻居和右邻居
                    if i>0:
                        new_prev=(word[i-1],word[i])
                        stats[new_prev]+=freq
                        indices[new_prev].add(idx)
                    #右邻居
                    if i<len(word)-1:
                        new_next=(word[i],word[i+1])
                        stats[new_next]+=freq
                        indices[new_next].add(idx)
                else:
                    #没有匹配上，i+1
                    i+=1
        if best_pair in stats:del stats[best_pair]
        if best_pair in indices:del indices[best_pair]
    #合并结束之后
    #构建新的词表
    for pair in merges:
        new_id=len(vocab)
        vocab[new_id]=pair[0]+pair[1]

    #添加特殊tokens
    for special_token in special_tokens:
        s_bytes=special_token.encode("utf_8")
        vocab[len(vocab)]=s_bytes

    return vocab,merges

def bytes_to_unicode():
    """
    创建一个映射，将 0-255 字节映射为一组可见的 Unicode 字符。
    这是 GPT-2 源码中的标准做法。
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


def save_tokenizer_files(vocab, merges, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    # 初始化映射表
    byte_encoder = bytes_to_unicode()

    # 词表保存
    # 使用 byte_encoder 将 bytes 转换为可见字符串
    json_vocab = {
        "".join(byte_encoder[b] for b in v): k
        for k, v in vocab.items()
    }
    with open(os.path.join(out_dir, "vocab.json"), "w", encoding="utf-8") as f:
        json.dump(json_vocab, f, indent=4)

    # 合并规则保存
    with open(os.path.join(out_dir, "merges.txt"), "w", encoding="utf-8") as f:
        for p1, p2 in merges:
            # 同样转换 p1 和 p2
            s1 = "".join(byte_encoder[b] for b in p1)
            s2 = "".join(byte_encoder[b] for b in p2)
            f.write(f"{s1} {s2}\n")


def main():
    input_path="C:/Users/a7655/AppData/Local/Temp/bpe_train_test.txt"
    vocab_size=512
    special_tokens=["<|endoftext|>"]
    output_dir="D:/keyan/cs336-assignment1/bpe_test_output"
    print(f"开始训练BPE分词器（目标词表大小为：{vocab_size}）...")
    vocab,merges=train_bpe(input_path,vocab_size,special_tokens)
    #保存结果
    save_tokenizer_files(vocab,merges,output_dir)

if __name__=="__main__":
    main()

           




    
    
