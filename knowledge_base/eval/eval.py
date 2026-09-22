"""
RAGAS evaluation script for the current RAG workflow.

读取 eval/qa.csv 中的问题集，调用 knowledge.processor.query_process.main_graph.query_app
获取 RAG 最终 answer 与 reranked_docs，然后使用 ragas 计算评估指标，并将结果写入
eval/qa_eval.csv。
"""

from __future__ import annotations

import csv
import logging
import sys
import uuid
from pathlib import Path
from typing import Any, Dict, List, Tuple

from langchain_core.embeddings import Embeddings


# eval.py 位于 eval 目录内。直接执行 python eval/eval.py 时，Python 默认只把
# eval 目录加入 sys.path；这里主动把项目根目录加入 sys.path，保证 knowledge 包可导入。
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

INPUT_CSV = PROJECT_ROOT / "eval" / "qa.csv"
OUTPUT_CSV = PROJECT_ROOT / "eval" / "qa_eval.csv"

OUTPUT_COLUMNS = [
    "question",
    "context",
    "answer",
    "ground_truth",
    "faithfulness",
    "answer_relevancy",
    "context_precision",
    "context_recall",
    "answer_correctness",
]

logger = logging.getLogger("ragas_eval")


def step_1_read_qa_csv(input_path: Path) -> List[Dict[str, str]]:
    """
    读取评估问题集。

    详细说明：
    1. 使用 utf-8-sig 编码读取 CSV，可兼容带 UTF-8 BOM 的文件。
    2. 要求文件至少包含 question 与 ground_truth 两列。
    3. 自动跳过空 question 行，避免空问题进入 RAG 流程导致无意义调用。
    4. 返回统一的字典列表，后续步骤只依赖 question 与 ground_truth 两个字段。
    """
    if not input_path.exists():
        raise FileNotFoundError(f"问题集文件不存在: {input_path}")

    with input_path.open("r", encoding="utf-8-sig", newline="") as file_obj:
        reader = csv.DictReader(file_obj)
        fieldnames = set(reader.fieldnames or [])
        missing_columns = {"question", "ground_truth"} - fieldnames
        if missing_columns:
            raise ValueError(f"问题集缺少必要列: {', '.join(sorted(missing_columns))}")

        rows: List[Dict[str, str]] = []
        for row in reader:
            question = (row.get("question") or "").strip()
            ground_truth = (row.get("ground_truth") or "").strip()
            if question:
                rows.append({"question": question, "ground_truth": ground_truth})

    logger.info("读取问题集完成，共 %s 条有效问题", len(rows))
    return rows


def step_2_run_rag_flow(qa_rows: List[Dict[str, str]]) -> List[Dict[str, Any]]:
    """
    逐条调用当前项目的 RAG 流程。

    详细说明：
    1. 从 main_graph.py 导入全局 query_app，确保评估对象就是当前项目真实 RAG 入口。
    2. 每个问题使用独立 session_id 与 task_id，避免历史对话在问题之间串扰。
    3. 使用 create_default_state 补齐 LangGraph 状态默认字段，减少字段缺失风险。
    4. 从最终 state 中读取 answer 与 reranked_docs；reranked_docs 会被转换成 ragas 需要的
       contexts: List[str]。
    """
    from knowledge.processor.query_process.main_graph import query_app
    from knowledge.processor.query_process.state import create_default_state

    evaluated_rows: List[Dict[str, Any]] = []

    for index, row in enumerate(qa_rows, start=1):
        question = row["question"]
        logger.info("[%s/%s] 调用 RAG: %s", index, len(qa_rows), question)

        session_id = f"eval_session_{uuid.uuid4().hex}"
        task_id = f"eval_task_{uuid.uuid4().hex}"
        initial_state = create_default_state(
            original_query=question,
            session_id=session_id,
            task_id=task_id,
            message_id="",
            is_stream=False,
        )

        final_state = query_app.invoke(initial_state)
        answer = str(final_state.get("answer") or "")
        reranked_docs = final_state.get("reranked_docs") or []
        contexts = step_2_extract_contexts_from_reranked_docs(reranked_docs)

        evaluated_rows.append(
            {
                "question": question,
                "ground_truth": row["ground_truth"],
                "answer": answer,
                "contexts": contexts,
                "context": step_2_format_context_for_csv(contexts),
            }
        )

    return evaluated_rows


def step_2_extract_contexts_from_reranked_docs(reranked_docs: List[Any]) -> List[str]:
    """
    从 query_app 最终 state 的 reranked_docs 中提取上下文文本。

    详细说明：
    1. 当前项目的 rerank_node 会将文档格式化为 dict，正文位于 content 字段。
    2. 为了增强兼容性，如果遇到非 dict 文档，会退化为 str(doc)。
    3. 会过滤空字符串，避免 ragas 因空上下文产生无效样本。
    4. 如果最终没有可用上下文，返回 [""]，保证 ragas 的 retrieved_contexts/contexts 字段仍是列表。
    """
    contexts: List[str] = []

    for doc in reranked_docs:
        if isinstance(doc, dict):
            content = str(doc.get("content") or "").strip()
        else:
            content = str(doc or "").strip()

        if content:
            contexts.append(content)

    return contexts or [""]


def step_2_format_context_for_csv(contexts: List[str]) -> str:
    """
    将上下文列表格式化为 CSV 单元格可读文本。

    详细说明：
    1. ragas 评估使用 contexts 列表，但输出 CSV 的 context 列需要便于人工查看。
    2. 每段上下文使用 [1]、[2] 这样的序号标记。
    3. 多段上下文之间用空行分隔，方便在表格软件中展开查看。
    """
    return "\n\n".join(f"[{index}] {context}" for index, context in enumerate(contexts, start=1))


def step_3_build_ragas_dataset(evaluated_rows: List[Dict[str, Any]]):
    """
    构建 ragas 可消费的数据集。

    详细说明：
    1. 新版 ragas 常用字段是 user_input、response、retrieved_contexts、reference。
    2. 旧版 ragas 常用字段是 question、answer、contexts、ground_truth。
    3. 这里同时提供两套字段，提升脚本对不同 ragas 版本的兼容性。
    4. 返回 datasets.Dataset，由 ragas.evaluate 直接消费。
    """
    try:
        from datasets import Dataset
    except ImportError as exc:
        raise ImportError("缺少 datasets 依赖，请先安装: pip install datasets") from exc

    ragas_rows: List[Dict[str, Any]] = []
    for row in evaluated_rows:
        ragas_rows.append(
            {
                # ragas v0.2+ / v0.3+ 常用字段
                "user_input": row["question"],
                "response": row["answer"],
                "retrieved_contexts": row["contexts"],
                "reference": row["ground_truth"],
                # ragas v0.1 常用字段
                "question": row["question"],
                "answer": row["answer"],
                "contexts": row["contexts"],
                "ground_truth": row["ground_truth"],
            }
        )

    return Dataset.from_list(ragas_rows)


class ProjectBgeM3Embeddings(Embeddings):
    """
    将 knowledge.utils 中的 BGE-M3 向量工具适配为 LangChain Embeddings 接口。

    详细说明：
    1. ragas 的 answer_relevancy 与 answer_correctness 需要 embedding 能力。
    2. 项目已有 knowledge.utils.bge_m3_embedding_util，这里直接复用项目工具。
    3. BGE-M3 工具会返回 dense 与 sparse 两类向量；ragas 语义相似度使用 dense 向量即可。
    """

    def __init__(self) -> None:
        from knowledge.utils.bge_m3_embedding_util import get_beg_m3_embedding_model

        self.embedding_model = get_beg_m3_embedding_model()
        if self.embedding_model is None:
            raise ValueError("BGE-M3 embedding 模型初始化失败，请检查 BGE_M3_PATH/BGE_DEVICE 配置")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        生成文档向量。

        详细说明：
        1. 入参 texts 是 ragas/LangChain 传入的一批文本。
        2. 调用项目已有 generate_hybrid_embeddings 生成 BGE-M3 向量。
        3. 返回 dense 向量列表，满足 LangChain Embeddings 的返回格式。
        """
        from knowledge.utils.bge_m3_embedding_util import generate_hybrid_embeddings

        safe_texts = [str(text) if text is not None and str(text).strip() else " " for text in texts]
        embedding_result = generate_hybrid_embeddings(self.embedding_model, safe_texts)
        if not embedding_result or "dense" not in embedding_result:
            raise ValueError("BGE-M3 embedding 生成失败")
        return embedding_result["dense"]

    def embed_query(self, text: str) -> List[float]:
        """
        生成单条查询向量。

        详细说明：
        1. LangChain Embeddings 要求查询向量单独实现 embed_query。
        2. 这里复用 embed_documents，保证查询和文档使用同一套向量逻辑。
        """
        return self.embed_documents([text])[0]


def step_4_prepare_ragas_model_tools() -> Tuple[Any, Any]:
    """
    准备 ragas 评估所需的 LLM 与 Embedding。

    详细说明：
    1. LLM 使用 knowledge.utils.llm_client_util.get_llm_client，与项目问答流程保持同源配置。
    2. Embedding 使用 ProjectBgeM3Embeddings，内部复用 knowledge.utils 的 BGE-M3 工具。
    3. 如果当前 ragas 版本提供 LangchainLLMWrapper/LangchainEmbeddingsWrapper，则包装后传入；
       如果未提供，则直接传入 LangChain 兼容对象。
    """
    from knowledge.utils.llm_client_util import get_llm_client

    llm_client = get_llm_client(temperature=0.0)
    if llm_client is None:
        raise ValueError("LLM 客户端初始化失败，请检查 OPENAI_API_KEY/OPENAI_API_BASE/ITEM_MODEL 配置")

    embedding_client = ProjectBgeM3Embeddings()

    try:
        from ragas.llms import LangchainLLMWrapper
        from ragas.embeddings import LangchainEmbeddingsWrapper

        return LangchainLLMWrapper(llm_client), LangchainEmbeddingsWrapper(embedding_client)
    except ImportError:
        return llm_client, embedding_client


def step_4_prepare_ragas_metrics() -> List[Any]:
    """
    准备 ragas 五个评估指标。

    详细说明：
    1. 优先使用旧版常见实例名：faithfulness、answer_relevancy、context_precision、
       context_recall、answer_correctness。
    2. 如果当前 ragas 版本不提供这些实例名，则回退到类名实例化方式。
    3. 回退类名中 ResponseRelevancy 的输出名仍通常是 answer_relevancy；
       ContextPrecisionWithReference 类的输出列名可能随版本变化，写 CSV 时会做别名映射。
    """
    try:
        from ragas.metrics import (
            answer_correctness,
            answer_relevancy,
            context_precision,
            context_recall,
            faithfulness,
        )

        return [
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
            answer_correctness,
        ]
    except ImportError:
        from ragas.metrics import (
            AnswerCorrectness,
            Faithfulness,
            LLMContextPrecisionWithReference,
            LLMContextRecall,
            ResponseRelevancy,
        )

        return [
            Faithfulness(),
            ResponseRelevancy(),
            LLMContextPrecisionWithReference(),
            LLMContextRecall(),
            AnswerCorrectness(),
        ]


def step_5_evaluate_with_ragas(ragas_dataset: Any) -> List[Dict[str, Any]]:
    """
    调用 ragas.evaluate 执行评估。

    详细说明：
    1. 导入 ragas.evaluate 作为统一评估入口。
    2. 将步骤 4 准备好的 LLM、Embedding 和五个指标传入。
    3. 将 ragas 的 EvaluationResult 转换为普通 dict 列表，方便后续写 CSV。
    """
    try:
        from ragas import evaluate
    except ImportError as exc:
        raise ImportError("缺少 ragas 依赖，请先安装: pip install ragas") from exc

    evaluator_llm, evaluator_embeddings = step_4_prepare_ragas_model_tools()
    metrics = step_4_prepare_ragas_metrics()

    result = evaluate(
        ragas_dataset,
        metrics=metrics,
        llm=evaluator_llm,
        embeddings=evaluator_embeddings,
    )

    return step_5_convert_ragas_result_to_records(result)


def step_5_convert_ragas_result_to_records(ragas_result: Any) -> List[Dict[str, Any]]:
    """
    将 ragas 评估结果转换为行记录。

    详细说明：
    1. 常见 ragas 版本的返回对象支持 to_pandas()。
    2. 如果返回对象不支持 to_pandas()，但本身是 dict，则按 dict 兜底转换。
    3. 该函数只负责结构转换，不负责业务字段合并。
    """
    if hasattr(ragas_result, "to_pandas"):
        result_df = ragas_result.to_pandas()
        return result_df.to_dict(orient="records")

    if isinstance(ragas_result, dict):
        keys = list(ragas_result.keys())
        row_count = max(
            (len(value) for value in ragas_result.values() if isinstance(value, list)),
            default=0,
        )
        return [
            {
                key: ragas_result[key][index]
                if isinstance(ragas_result[key], list) and index < len(ragas_result[key])
                else ragas_result[key]
                for key in keys
            }
            for index in range(row_count)
        ]

    raise TypeError(f"无法识别的 ragas 评估结果类型: {type(ragas_result)!r}")


def step_6_write_eval_csv(
    output_path: Path,
    evaluated_rows: List[Dict[str, Any]],
    score_rows: List[Dict[str, Any]],
) -> None:
    """
    写出最终评估结果 CSV。

    详细说明：
    1. 严格按照用户要求输出列头：
       question, context, answer, ground_truth, faithfulness, answer_relevancy,
       context_precision, context_recall, answer_correctness。
    2. 使用 utf-8-sig 编码写入，也就是 UTF-8 BOM，方便 Excel 正确识别中文。
    3. ragas 不同版本可能输出略有不同的指标列名，写入前会用别名表统一映射到目标列名。
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8-sig", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()

        for index, row in enumerate(evaluated_rows):
            score_row = score_rows[index] if index < len(score_rows) else {}
            writer.writerow(
                {
                    "question": row["question"],
                    "context": row["context"],
                    "answer": row["answer"],
                    "ground_truth": row["ground_truth"],
                    "faithfulness": step_6_pick_metric_score(score_row, "faithfulness"),
                    "answer_relevancy": step_6_pick_metric_score(score_row, "answer_relevancy"),
                    "context_precision": step_6_pick_metric_score(score_row, "context_precision"),
                    "context_recall": step_6_pick_metric_score(score_row, "context_recall"),
                    "answer_correctness": step_6_pick_metric_score(score_row, "answer_correctness"),
                }
            )

    logger.info("评估结果已写入: %s", output_path)


def step_6_pick_metric_score(score_row: Dict[str, Any], metric_name: str) -> Any:
    """
    从 ragas 单行结果中提取指定指标分数。

    详细说明：
    1. 优先按目标列名直接读取。
    2. 如果当前 ragas 版本使用新版指标列名，则按别名映射读取。
    3. 如果仍未找到，返回空字符串，避免 CSV 写入 None 或报错。
    """
    aliases = {
        "faithfulness": ["faithfulness"],
        "answer_relevancy": ["answer_relevancy", "response_relevancy"],
        "context_precision": [
            "context_precision",
            "llm_context_precision_with_reference",
            "context_precision_with_reference",
        ],
        "context_recall": ["context_recall", "llm_context_recall"],
        "answer_correctness": ["answer_correctness"],
    }

    for alias in aliases.get(metric_name, [metric_name]):
        if alias in score_row:
            return score_row[alias]

    return ""


def step_7_main() -> None:
    """
    评估脚本主流程。

    详细说明：
    1. 初始化日志，便于观察当前执行到哪个问题。
    2. 串联读取问题集、调用 RAG、构造 ragas 数据集、执行评估、写出 CSV 五个核心步骤。
    3. 所有路径都基于当前项目根目录计算，避免执行目录不同导致文件找不到。
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    qa_rows = step_1_read_qa_csv(INPUT_CSV)
    evaluated_rows = step_2_run_rag_flow(qa_rows)
    ragas_dataset = step_3_build_ragas_dataset(evaluated_rows)
    score_rows = step_5_evaluate_with_ragas(ragas_dataset)
    step_6_write_eval_csv(OUTPUT_CSV, evaluated_rows, score_rows)


if __name__ == "__main__":
    step_7_main()
