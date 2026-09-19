from pathlib import Path
from dataclasses import dataclass, field
from markitdown import MarkItDown

# 为什么用 MarkItDown 不会"丢格式"？
#   直接提取 PDF 文本的问题是"视觉换行"太多：排版为了换行而断行（例如一段话被硬折成两行），若按行切分会把同一句话劈成两半。
#   MarkItDown 做的不是抹掉结构，而是把杂乱排版"整理"成规律的 markdown 语义结构：
#   - 同一段落内的视觉断行 -> 合并成一行（中间保留一个空格）；
#   - 真正的段落边界（含分页）-> 归一成空行 \n\n；
#   - 章节标题 -> 归一成 # / ## markdown 标题。
#   这份整理后的 text 正好喂给 text_splitter：它按标题(#) -> 段落(\n\n) -> 中文标点（。；！？，）的优先级递归切分，
#   从而在每个 chunk 尽量保持章节/段落语义完整。
#   若不用 MarkItDown 直接丢原始提取文本，\n 遍地都是，反而会被切得很碎。
#   PDF 里真正"丢"的只是图片/表格等无法进文本的内容，而这些本就不参与文本切分。

"""文档对象：承载一份解析后的文档全文与元数据。"""
@dataclass
class Document:
    """
    一份被解析出来的文档。

    属性：
        text:     文档全文（解析/提取后的文本）
        metadata: 元数据 dict，如 {"source": 文件名, "path": 绝对路径, "ext": 后缀}
        chunks:   切分后的 chunk 列表（由 DocumentParser.split 或外层填充）
    """

    text: str = ""
    metadata: dict = field(default_factory=dict)
    chunks: list[str] = field(default_factory=list)

    def __repr__(self) -> str:
        return (f"Document(metadata={self.metadata}, "
                f"text_len={len(self.text)}, chunks={len(self.chunks)})")


class DocumentParser:
    """文档解析器：用 MarkItDown 把磁盘上的原始文档解析成 Document（全文 + 元数据）。"""

    def __init__(self):
        self._converter = MarkItDown()

    def parse(self, path: str | Path) -> Document:
        """按扩展名解析单份文档，返回解析后的 Document。"""
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"文档不存在: {p}")

        metadata = {
            "source": p.name,       # 文件名（用作 RAG 的 source 标识）
            "path": str(p.resolve()),
            "ext": p.suffix.lower(),
        }

        # 失败会抛异常，由调用方捕获提示（如 rag_manager）。
        text = self._converter.convert(str(p)).text_content

        # PDF 提取时，无法映射到 Unicode 的字形会变成 (cid:127) 这样的占位符（这里 127 是原文的项目符号）。
        # 它只是排版符号，没有语义，直接删掉，避免噪声进入知识库。
        text = text.replace("(cid:127)", "")

        return Document(text=text.strip(), metadata=metadata)
