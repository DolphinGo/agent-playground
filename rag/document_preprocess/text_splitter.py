"""文本切分器：模仿 LangChain 的 RecursiveCharacterTextSplitter 策略。

策略核心：
- 按一组"从强到弱"的分隔符递归切分文本（空段落 > 换行 > 中文句号 > 空格），尽量在语义边界处断开，最后再退化为按固定长度硬切。
- 相邻两块之间保留 chunk_overlap 长度的重叠字符，弥补"边界信息被切断"的损失。
- chunk_size / chunk_overlap 均按【字符】计。对中文，250+50 字符远低于bge-large-zh-v1.5 上限 512 token，可避免 embedding 时被静默截断。
"""

# 分隔符优先级：从"强语义边界"到"弱边界"（空串永远是最后兜底）
# 输入是 MarkItDown 转出的 markdown，故先按标题(#)再按段落(\n\n)等结构边界切分，
# 让每个 chunk 尽量保持章节/段落语义完整。
# "\n# " "\n## " 标题 "\n\n\n" 空段落 "\n\n" 空行 "\n" 换行
SEPARATORS = ["\n# ", "\n## ", "\n\n\n", "\n\n", "\n", "。", "；", "！", "？", "，", "、", " ", ""]

def _merge_splits(splits: list[str], separator: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """把切好的小片段贪心合并成不超过 chunk_size 的块，并叠加 overlap。"""
    chunks: list[str] = []
    current = ""
    for split in splits:
        # 若当前块再加一段会超限，则收拢当前块，并用上一块的尾部 overlap 起新块
        if len(current) + (len(separator) if current else 0) + len(split) > chunk_size:
            if current:
                chunks.append(current)
            if chunk_overlap > 0 and chunks:
                current = chunks[-1][-chunk_overlap:]
            else:
                current = ""
            # overlap 本身已占用长度，若上一段过长，这里直接截断保护
            if len(current) >= chunk_size:
                current = current[:chunk_size]
        if separator and (not current or not current.endswith(separator)):
            current += separator
        current += split
    if current:
        chunks.append(current)
    return chunks


def _recursive_split(text: str, separators: list[str], chunk_size: int, chunk_overlap: int) -> list[str]:
    """按分隔符优先级递归切分；返回的每块长度不超过 chunk_size。"""
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]

    # 找到当前优先级下"确实出现在文本中"的第一个分隔符（空串必定出现，保证有解）
    chosen = next((s for s in separators if s in text), None)
    if chosen is None:
        # 理论不会走到（空串兜底），安全起见按固定长度硬切
        return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

    # 用更低优先级继续处理"仍超限"的子片段
    lower_seps = separators[separators.index(chosen) + 1:]
    parts = text.split(chosen)
    splits: list[str] = []
    for i, part in enumerate(parts):
        piece = (chosen + part) if i > 0 else part  # 保留分隔符在片首，维持语义
        if len(piece) > chunk_size:
            splits.extend(_recursive_split(piece, lower_seps, chunk_size, 0))
        else:
            splits.append(piece)

    return _merge_splits(splits, "", chunk_size, chunk_overlap)


def split_text(text: str, chunk_size: int = 250, chunk_overlap: int = 50) -> list[str]:
    """
    对一段长文本做切分：返回若干 chunk，每个 chunk 长度（字符）不超过 chunk_size，
    相邻 chunk 之间带 chunk_overlap 的重叠。

    args:
        text: 原始文本
        chunk_size: 单块最大字符数（默认 250，适配中文 embedding 上限）
        chunk_overlap: 相邻两块重叠的字符数（默认 50）
    """
    text = (text or "").strip() # 只处理开头和结尾两端空格
    if not text:
        return []
    return _recursive_split(text, SEPARATORS, chunk_size, chunk_overlap)