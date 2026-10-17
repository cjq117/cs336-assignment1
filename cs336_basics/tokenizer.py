import regex as re
from collections.abc import Iterable

class BPETokenizer:
    #初始化部分
    def __init__(self,vocab:dict[int,bytes],merges:list[tuple[bytes,bytes]],special_tokens:list[str]|None = None):
       self.vocab=vocab
       self.id_to_byte=vocab
       self.byte_to_id={v:k for k,v in vocab.items()}
       self.merges={pair:i for i ,pair in enumerate(merges)}
       self.special_tokens=special_tokens or []
       if self.special_tokens:
           #按照长度从长到短排序（reverse=true）,这样会优先匹配最长的特殊标记（防止<|a|><|b|>被错误拆分）
           sorted_special = sorted(self.special_tokens,key=len,reverse=True)
           special_pattern="|".join(re.escape(t) for t in sorted_special)
           self.special_regex=re.compile(special_pattern)
       else:
           self.special_regex=None
       #gpt2官方预分词正则表达式
       self.gpt2_pat = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

    def encode(self,text:str)->list[int]:
        if not text:
            return []
        #如果没有特殊字符直接进行编码
        if not self.special_regex:
            return self._encode_text_segment(text)

        tokens=[]
        #记录上一次匹配结束的位置
        last_pos=0
        #使用finditer遍历文本中所有符合特殊标记模式的匹配项
        #finditer能够提供match.start()和match.end(),能够知道特殊标记在哪里开始，在哪里结束
        for match in self.special_regex.finditer(text):
            pre_text=text[last_pos:match.start()]
            if pre_text:
                tokens.extend(self._encode_text_segment(pre_text))
            #match.group()拿到的是被识别出来的特殊字符串
            special_tok=match.group()
            tokens.append(self.byte_to_id[special_tok.encode("utf_8")])
            last_pos=match.end()
        remaining_text=text[last_pos:]
        if remaining_text:
            tokens.extend(self._encode_text_segment(remaining_text))

        return tokens


    def _encode_text_segment(self,text:str)->list[int]:
            #对不含特殊tokens的文本应用·BPE合并逻辑
            ids=[]
            pre_tokens=self.gpt2_pat.findall(text)
            for p_tok in pre_tokens:
                #"Hello"->[b'H', b'e', b'l', b'l', b'o']
                byte_parts=[bytes([b]) for b in p_tok.encode("utf_8")]
                while len(byte_parts)>=2:
                    best_pair=None
                    min_rank=float("inf")
                    for i in range(len(byte_parts)-1):
                        pair=(byte_parts[i],byte_parts[i+1])
                        if pair in self.merges:
                            rank=self.merges[pair]
                            if rank<min_rank:
                                min_rank=rank
                                best_pair=pair
                    if best_pair is None:
                        break
                    #找到一个best_pair之后，将这个best_pair合并之后的byte_parts序列
                    new_byte_part=[]
                    i=0
                    while i<len(byte_parts):
                        if i<len(byte_parts)-1 and (byte_parts[i],byte_parts[i+1])==best_pair:
                            new_byte_part.append(best_pair[0]+best_pair[1])
                            i+=2
                        else:
                            new_byte_part.append(byte_parts[i])
                            i+=1
                    byte_parts=new_byte_part
                for part in byte_parts:
                    ids.append(self.byte_to_id[part])
            return ids

        
    def decode(self,ids:list[int])->str:
        byte_segments=[self.id_to_byte[i] for i in ids]
        full_bytes=b"".join(byte_segments)
        #full_bytes = b"hello world"
        #utf8 解码得到字符串："hello world"
        #errors="replace"如果遇到损坏、非法 utf‑8 字节，不会抛异常崩溃，会用 � 替换乱码字节
        return full_bytes.decode("utf_8",errors="replace")

    def encode_iterable(self,iterable:Iterable[str])->Iterable[int]:
        for chunk in iterable:
            yield from self.encode(chunk)
                                    
                        

        
        
