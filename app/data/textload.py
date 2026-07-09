"""
因果知识库构建与检索工具。

功能：
- 自动扫描 `app/data/` 下所有 .txt / .md / .pdf，分派对应 Loader 加载、切片、入 Chroma。
- 基于 (mtime, size) 的增量缓存：未变更的文件跳过 embed，节省时间与额度。
- 给每个 chunk 写入 metadata={"source": 文件名, "page": 页码(仅 PDF)}，便于检索结果引用出处。
- 按扩展名采用不同的 chunk 参数：PDF 段落更长，使用更大的窗口。

对外 API：
- build_causal_vector_store(force=False)：兼容旧接口，等价于 build_from_directory。
- build_from_directory(dir_path=None, force=False) -> {filename: chunk_count}
- build_from_file(file_path, force=False, cache=None) -> int
- search_causal_knowledge(query, k=4) -> List[dict]
    每个 dict 形如 {"content": ..., "source": "xxx.pdf", "page": 41}

作为脚本运行：
    python -m data.textload         # 仅入库未变更过的新文件 / 变更过的文件
    python -m data.textload --force # 强制全部重新入库
    python -m data.textload --query "什么是混杂变量？"  # 跑一次检索
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

try:
    from common.logger import logger
except Exception:  # 允许独立脚本运行
    import logging
    logger = logging.getLogger("causal_textload")
    if not logger.handlers:
        logging.basicConfig(level=logging.INFO,
                            format="%(asctime)s - %(levelname)s - %(name)s - %(message)s")

load_dotenv()


# ---------------------------------------------------------------------------
# 路径与常量
# ---------------------------------------------------------------------------

_DATA_DIR = os.path.dirname(os.path.abspath(__file__))
_APP_DIR = os.path.dirname(_DATA_DIR)
DEFAULT_DATA_DIR = _DATA_DIR
DEFAULT_PERSIST_DIR = os.path.join(_APP_DIR, "chroma_db")
DEFAULT_COLLECTION = "causal_agent"
CACHE_FILE = os.path.join(DEFAULT_PERSIST_DIR, ".kb_cache.json")

SUPPORTED_EXTS = {".txt", ".md", ".pdf"}

# 按扩展名设置切片大小（PDF 段落明显更长）
CHUNK_PARAMS = {
    ".txt": (300, 60),
    ".md":  (300, 60),
    ".pdf": (900, 150),
}

SEPARATORS = ["\n\n", "\n", "。", "！", "？", ".", "?", "!", "，", ",", " ", ""]


# ---------------------------------------------------------------------------
# 内部单例
# ---------------------------------------------------------------------------

_embedding: Optional[DashScopeEmbeddings] = None
_vector_store: Optional[Chroma] = None


def _get_embedding() -> DashScopeEmbeddings:
    global _embedding
    if _embedding is None:
        _embedding = DashScopeEmbeddings(
            model="text-embedding-async-v2",
            dashscope_api_key=os.getenv("DASHSCOPE_API_KEY"),
        )
    return _embedding


def _get_store(persist_dir: str = DEFAULT_PERSIST_DIR,
               collection: str = DEFAULT_COLLECTION) -> Chroma:
    """获取（或创建）持久化的 Chroma 向量库实例。"""
    global _vector_store
    if _vector_store is None:
        os.makedirs(persist_dir, exist_ok=True)
        _vector_store = Chroma(
            collection_name=collection,
            embedding_function=_get_embedding(),
            persist_directory=persist_dir,
        )
    return _vector_store


# ---------------------------------------------------------------------------
# 增量缓存（mtime + size 签名）
# ---------------------------------------------------------------------------

def _load_cache() -> Dict[str, Dict[str, Any]]:
    if not os.path.exists(CACHE_FILE):
        return {}
    try:
        with open(CACHE_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data if isinstance(data, dict) else {}
    except Exception as exc:
        logger.warning(f"读取知识库缓存失败，将重建: {exc}")
        return {}


def _save_cache(cache: Dict[str, Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(CACHE_FILE), exist_ok=True)
    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)


def _file_signature(path: str) -> str:
    st = os.stat(path)
    return f"{int(st.st_mtime)}-{st.st_size}"


# ---------------------------------------------------------------------------
# 文档加载与切分
# ---------------------------------------------------------------------------

def _load_documents(file_path: str):
    """按扩展名分派 Loader，统一在 metadata 中补充 source/page。"""
    ext = Path(file_path).suffix.lower()
    fname = os.path.basename(file_path)

    if ext == ".pdf":
        loader = PyPDFLoader(file_path)
        docs = loader.load()
    elif ext in {".txt", ".md"}:
        loader = TextLoader(file_path, encoding="utf-8")
        docs = loader.load()
    else:
        raise ValueError(f"不支持的文件类型: {ext}")

    for d in docs:
        md = d.metadata or {}
        md.setdefault("source", fname)
        # PyPDFLoader 把 page 写在 metadata['page']（0-indexed）；其它 loader 没有页码
        d.metadata = md
    return docs


def _split(docs, ext: str):
    chunk_size, chunk_overlap = CHUNK_PARAMS.get(ext, (400, 80))
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=SEPARATORS,
        length_function=len,
    )
    return splitter.split_documents(docs)


# ---------------------------------------------------------------------------
# 构建
# ---------------------------------------------------------------------------

def build_from_file(file_path: str,
                    force: bool = False,
                    cache: Optional[Dict[str, Dict[str, Any]]] = None) -> int:
    """加载单个文件并入库；返回新写入的 chunk 数（命中缓存返回 0）。"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(file_path)
    ext = Path(file_path).suffix.lower()
    if ext not in SUPPORTED_EXTS:
        logger.info(f"跳过不支持的文件类型: {os.path.basename(file_path)}")
        return 0

    if cache is None:
        cache = _load_cache()

    fname = os.path.basename(file_path)
    sig = _file_signature(file_path)
    entry = cache.get(fname) or {}
    if not force and entry.get("sig") == sig:
        logger.info(f"[skip] {fname}（缓存命中）")
        return 0

    store = _get_store()

    docs = _load_documents(file_path)
    chunks = _split(docs, ext)
    if not chunks:
        cache[fname] = {"sig": sig, "ids": []}
        return 0

    # 先写新索引，成功后再删旧索引并更新 cache，确保失败时不会写入残缺 cache。
    ids = [f"{fname}::{sig}::chunk_{i:04d}" for i in range(len(chunks))]
    store.add_documents(documents=chunks, ids=ids)

    old_ids = entry.get("ids", [])
    if old_ids:
        try:
            store.delete(ids=old_ids)
        except Exception as exc:
            logger.warning(f"删除旧索引失败 {fname}: {exc}")

    cache[fname] = {"sig": sig, "ids": ids}
    logger.info(f"[done] {fname}: {len(chunks)} 块")
    return len(chunks)


def clean_orphan_cache_entries(
    dir_path: Optional[str] = None,
    cache: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Dict[str, int]:
    """清理 cache 中已不存在源文件的孤儿索引。"""
    dir_path = dir_path or DEFAULT_DATA_DIR
    cache = cache if cache is not None else _load_cache()
    if not cache:
        return {"orphans": 0, "deleted_ids": 0}
    store = _get_store()

    deleted_ids = 0
    orphans = 0
    to_remove: list[str] = []
    for fname, entry in list(cache.items()):
        full = os.path.join(dir_path, fname)
        if os.path.exists(full):
            continue
        orphans += 1
        ids = entry.get("ids") or []
        if ids:
            try:
                store.delete(ids=ids)
                deleted_ids += len(ids)
            except Exception as exc:
                logger.warning(f"删除孤儿索引失败 {fname}: {exc}")
        to_remove.append(fname)

    for fname in to_remove:
        cache.pop(fname, None)
    _save_cache(cache)
    return {"orphans": orphans, "deleted_ids": deleted_ids}


def rebuild_cache_from_directory(dir_path: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
    """按磁盘文件重建 cache（不重算 embedding，不写向量）。"""
    dir_path = dir_path or DEFAULT_DATA_DIR
    rebuilt: Dict[str, Dict[str, Any]] = {}
    for entry in sorted(os.listdir(dir_path)):
        full = os.path.join(dir_path, entry)
        if not os.path.isfile(full):
            continue
        ext = Path(entry).suffix.lower()
        if ext not in SUPPORTED_EXTS:
            continue
        rebuilt[entry] = {"sig": _file_signature(full), "ids": []}
    _save_cache(rebuilt)
    return rebuilt


def build_from_directory(dir_path: Optional[str] = None,
                         force: bool = False) -> Dict[str, int]:
    """扫描目录下所有 .txt/.md/.pdf，增量入库。返回 {filename: 新写入块数}。"""
    dir_path = dir_path or DEFAULT_DATA_DIR
    if not os.path.isdir(dir_path):
        raise NotADirectoryError(dir_path)

    cache = _load_cache()
    result: Dict[str, int] = {}
    for entry in sorted(os.listdir(dir_path)):
        full = os.path.join(dir_path, entry)
        if not os.path.isfile(full):
            continue
        ext = Path(entry).suffix.lower()
        if ext not in SUPPORTED_EXTS:
            continue
        try:
            n = build_from_file(full, force=force, cache=cache)
        except Exception as exc:
            logger.warning(f"加载失败 {entry}: {exc}")
            n = 0
        result[entry] = n
    _save_cache(cache)
    return result


def build_causal_vector_store(kb_file: Optional[str] = None,
                              force: bool = False) -> int:
    """兼容旧接口：

    - 不传 kb_file 时，等价于 build_from_directory，扫描整个 data/。
    - 传入文件路径时，仅构建该文件。

    返回新写入的总 chunk 数。
    """
    if kb_file:
        return build_from_file(kb_file, force=force)
    summary = build_from_directory(force=force)
    return sum(summary.values())


# ---------------------------------------------------------------------------
# 检索
# ---------------------------------------------------------------------------

def search_causal_knowledge(query: str, k: int = 4) -> List[Dict[str, Any]]:
    """在本地因果知识库中检索相关片段。

    Returns:
        相关片段列表，按相关性排序。每项形如：
        {"content": str, "source": "xxx.pdf", "page": 41 | None}
        PDF 的 page 为 0-indexed；txt/md 无页码字段。
    """
    store = _get_store()
    results = store.similarity_search(query, k=k)
    out: List[Dict[str, Any]] = []
    for doc in results:
        md = doc.metadata or {}
        out.append({
            "content": doc.page_content,
            "source": md.get("source", "unknown"),
            "page": md.get("page"),
        })
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _print_summary(summary: Dict[str, int]) -> None:
    total = sum(summary.values())
    print(f"\n构建完成，本次写入 {total} 个新 chunk。详细：")
    for name, n in summary.items():
        flag = f"+{n}" if n > 0 else "skip"
        print(f"  {flag:>6}  {name}")
    print(f"\n持久化目录：{DEFAULT_PERSIST_DIR}")


if __name__ == "__main__":
    args = sys.argv[1:]
    force = "--force" in args
    query = None
    if "--query" in args:
        i = args.index("--query")
        query = args[i + 1] if i + 1 < len(args) else None

    summary = build_from_directory(force=force)
    _print_summary(summary)

    test_q = query or "什么是混杂变量？"
    print(f"\n--- 测试检索: {test_q} ---")
    for i, hit in enumerate(search_causal_knowledge(test_q, k=3), start=1):
        page = hit.get("page")
        head = f"[{i}] 来源: {hit['source']}" + (f" · p.{page + 1}" if isinstance(page, int) else "")
        print(head)
        snippet = hit["content"].strip().replace("\n", " ")
        if len(snippet) > 240:
            snippet = snippet[:240] + "…"
        print("    " + snippet)
        print("-" * 60)
