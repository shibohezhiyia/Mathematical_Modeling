"""
大模型智能分析模块

支持外部 OpenAI 兼容 API 和本地 Ollama 模型，
用于数据集解读、训练结果分析、错误日志诊断。
"""

import textwrap
from typing import Dict, Any, List
from dataclasses import dataclass, field
from urllib.parse import urlparse

import requests


@dataclass
class LLMConfig:
    """大模型配置"""
    provider: str = "openai"  # "openai"、"deepseek" 或 "ollama"
    base_url: str = "https://api.openai.com/v1"
    api_key: str = field(default="", repr=False)
    model_name: str = "gpt-4o"
    timeout: int = 120

    def validate(self) -> "LLMConfig":
        self.provider = str(self.provider).strip().lower()
        if self.provider not in {"openai", "openai_compatible", "deepseek", "ollama"}:
            raise ValueError("LLM provider 必须是 openai/openai_compatible/deepseek/ollama")
        self.base_url = str(self.base_url).strip().rstrip("/")
        self.model_name = str(self.model_name).strip()
        if not self.base_url or not self.model_name:
            raise ValueError("LLM 服务地址和模型名称不能为空")
        if len(self.base_url) > 2048 or len(self.model_name) > 200:
            raise ValueError("LLM 服务地址或模型名称过长")
        if not isinstance(self.timeout, int) or isinstance(self.timeout, bool) or not 5 <= self.timeout <= 300:
            raise ValueError("LLM 请求超时必须是 5 到 300 秒的整数")
        parsed = urlparse(self.base_url)
        if not parsed.hostname or parsed.scheme not in {"http", "https"}:
            raise ValueError("LLM 服务地址必须是完整的 HTTP(S) URL")
        if parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ValueError("LLM 服务地址不能包含账号、查询参数或锚点")
        if self.provider == "deepseek":
            if parsed.scheme != "https" or parsed.hostname.lower() != "api.deepseek.com":
                raise ValueError("DeepSeek 官方 API 地址必须是 https://api.deepseek.com")
            if parsed.path.rstrip("/") not in {"", "/v1"}:
                raise ValueError("DeepSeek API 地址只支持根地址或 /v1")
            if not str(self.api_key).strip():
                raise ValueError("DeepSeek API Key 不能为空")
        return self


class LLMClient:
    """统一的 LLM 调用客户端"""

    def __init__(self, config: LLMConfig):
        self.config = config.validate()

    @staticmethod
    def _http_error_detail(error: requests.exceptions.HTTPError) -> str:
        response = error.response
        if response is None:
            return str(error)
        detail = ""
        try:
            payload = response.json()
            raw = payload.get("error", payload) if isinstance(payload, dict) else payload
            detail = raw.get("message", "") if isinstance(raw, dict) else str(raw)
        except (ValueError, TypeError):
            detail = ""
        detail = str(detail).strip()[:500]
        return f"HTTP {response.status_code}" + (f"：{detail}" if detail else "")

    def _call_openai_api(self, messages: List[Dict[str, Any]]) -> str:
        """调用 OpenAI 兼容 API"""
        config = self.config
        url = f"{config.base_url.rstrip('/')}/chat/completions"
        headers = {"Content-Type": "application/json"}
        if config.api_key:
            headers["Authorization"] = f"Bearer {config.api_key}"

        payload = {
            "model": config.model_name,
            "messages": messages,
            "temperature": 0.7,
            "max_tokens": 4096,
        }
        resp = requests.post(url, headers=headers, json=payload, timeout=config.timeout)
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"]

    def _call_ollama_native(self, messages: List[Dict[str, Any]]) -> str:
        """调用 Ollama 原生 /api/chat API（fallback）"""
        config = self.config
        base = config.base_url.rstrip('/')
        if base.endswith('/v1'):
            base = base[:-3]
        url = f"{base}/api/chat"

        payload = {
            "model": config.model_name,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": 0.7,
                "num_predict": 4096,
            }
        }
        resp = requests.post(url, json=payload, timeout=config.timeout)
        resp.raise_for_status()
        data = resp.json()
        if "message" in data:
            return data["message"]["content"]
        if "response" in data:
            return data["response"]
        raise ValueError(f"Ollama 返回格式异常: {list(data.keys())}")

    def chat_completion(self, messages: List[Dict[str, Any]]) -> str:
        """
        调用大模型对话接口
        优先使用 OpenAI 兼容 API，失败时 fallback 到 Ollama 原生 API
        """
        config = self.config
        last_error = None

        # 1. 先尝试 OpenAI 兼容 API
        try:
            return self._call_openai_api(messages)
        except requests.exceptions.ConnectionError as e:
            raise ConnectionError(
                f"无法连接到 LLM 服务 ({config.base_url})。"
                f"请检查 URL 是否正确，或确认 Ollama 已启动。"
            ) from e
        except requests.exceptions.Timeout as e:
            raise TimeoutError(
                f"LLM 请求超时 ({config.timeout}s)。请稍后重试或选择更快的模型。"
            ) from e
        except requests.exceptions.HTTPError as e:
            last_error = self._http_error_detail(e)
            # OpenAI 兼容外部 API 不存在 Ollama 原生回退，直接保留可操作错误。
        except (ValueError, KeyError, IndexError, TypeError) as e:
            last_error = str(e)
            # OpenAI 兼容 API 失败，继续尝试原生 API

        # 2. Fallback 到 Ollama 原生 API（仅对 ollama provider）
        if config.provider == "ollama":
            try:
                return self._call_ollama_native(messages)
            except requests.exceptions.HTTPError as e2:
                status = e2.response.status_code if e2.response else "?"
                detail = ""
                try:
                    detail = e2.response.json().get("error", "")
                except Exception:
                    # 限定 Exception 避免吞掉 KeyboardInterrupt / SystemExit
                    pass
                if "not found" in str(detail).lower() or status == 404:
                    raise ValueError(
                        f"模型 '{config.model_name}' 未找到。"
                        f"请先运行: ollama pull {config.model_name}"
                    ) from e2
                raise ValueError(
                    f"Ollama API 错误 ({status}): {detail or str(e2)}"
                ) from e2

        # 3. 都不是 ollama，抛出原始错误
        raise ValueError(
            f"{'DeepSeek' if config.provider == 'deepseek' else 'LLM'} 调用失败: "
            f"{last_error}。请检查 API Key、模型名称和账户额度。"
        )

    def list_models(self) -> List[str]:
        """用轻量模型列表请求测试连接，不生成内容也不保存密钥。"""
        config = self.config
        headers = {"Accept": "application/json"}
        if config.api_key:
            headers["Authorization"] = f"Bearer {config.api_key}"
        base = config.base_url.rstrip("/")
        if config.provider == "ollama":
            if base.endswith("/v1"):
                base = base[:-3]
            url = f"{base}/api/tags"
        else:
            url = f"{base}/models"
        try:
            response = requests.get(url, headers=headers, timeout=min(config.timeout, 20), allow_redirects=False)
            if 300 <= response.status_code < 400:
                raise ValueError("LLM 服务返回了不允许的重定向")
            response.raise_for_status()
            payload = response.json()
        except requests.exceptions.Timeout as exc:
            raise TimeoutError("LLM 连接测试超时") from exc
        except requests.exceptions.ConnectionError as exc:
            raise ConnectionError(f"无法连接到 LLM 服务 ({config.base_url})") from exc
        except requests.exceptions.HTTPError as exc:
            raise ValueError(f"LLM 连接测试失败：{self._http_error_detail(exc)}") from exc
        except ValueError:
            raise
        except (TypeError, KeyError) as exc:
            raise ValueError("LLM 模型列表响应格式异常") from exc
        entries = payload.get("models", []) if config.provider == "ollama" else payload.get("data", [])
        return sorted({str(item.get("name") or item.get("id")) for item in entries if isinstance(item, dict) and (item.get("name") or item.get("id"))})


class AnalysisPromptBuilder:
    """Prompt 构建器"""

    SYSTEM_PROMPT = (
        "你是一位资深的数据科学与机器学习专家。"
        "请用中文回答，结构清晰，使用 Markdown 格式。"
        "分析要专业、具体、可操作，避免空洞的套话。"
    )

    @classmethod
    def build_eda_prompt(cls, df_info: Dict, eda_data: Dict) -> List[Dict[str, str]]:
        """构建数据集解读 Prompt"""
        shape = df_info.get("shape", [0, 0])
        columns = df_info.get("columns", [])
        type_info = df_info.get("columns", [])
        memory_mb = df_info.get("memory_mb", 0)

        # 数值统计摘要
        stats = eda_data.get("statistics", {})
        stats_text = ""
        for col, s in list(stats.items())[:10]:
            stats_text += f"\n- {col}: 均值={s.get('mean')}, 标准差={s.get('std')}, 最小值={s.get('min')}, 最大值={s.get('max')}, 中位数={s.get('median')}"

        # 类别分布
        cat_counts = eda_data.get("categorical_counts", {})
        cat_text = ""
        for col, counts in list(cat_counts.items())[:5]:
            top = list(counts.items())[:5]
            cat_text += f"\n- {col}: " + ", ".join([f"{k}({v})" for k, v in top])

        # 相关性
        corr = eda_data.get("correlation", {})
        corr_cols = corr.get("columns", [])
        corr_text = f"数值列: {', '.join(corr_cols)}" if corr_cols else "无足够数值列计算相关性"

        # 缺失值
        missing_summary = []
        for c in type_info:
            if isinstance(c, dict) and c.get("missing_rate", 0) > 0:
                missing_summary.append(f"{c['column']}: {c['missing_rate']*100:.1f}%")
        missing_text = "\n".join(missing_summary) if missing_summary else "无缺失值"

        content = textwrap.dedent(f"""\
            请分析以下数据集：

            ## 基本信息
            - 数据规模: {shape[0]} 行 × {shape[1]} 列
            - 内存占用: {memory_mb} MB
            - 列名: {', '.join(columns)}

            ## 数值列统计摘要
            {stats_text if stats_text else '无数值列'}

            ## 类别列分布（前5个）
            {cat_text if cat_text else '无类别列'}

            ## 相关性
            {corr_text}

            ## 缺失值情况
            {missing_text}

            请提供以下分析（使用 Markdown 格式）：
            1. **数据概览与质量评估** — 整体数据质量如何？有哪些明显问题？
            2. **关键发现与异常提示** — 是否有异常值、分布不均衡、高相关性等问题？
            3. **特征工程建议** — 针对当前数据，推荐哪些特征工程操作？
            4. **适合的分析方向** — 基于数据特点，推荐分类、回归还是聚类？为什么？
        """)

        return [
            {"role": "system", "content": cls.SYSTEM_PROMPT},
            {"role": "user", "content": content},
        ]

    @classmethod
    def build_result_prompt(cls, result_data: Dict) -> List[Dict[str, str]]:
        """构建训练结果分析 Prompt"""
        task_type = result_data.get("task_type", "unknown")
        leaderboard = result_data.get("leaderboard", [])
        decision = result_data.get("decision", {})
        preprocessing = result_data.get("preprocessing", {})
        ensemble_weights = result_data.get("ensemble_weights", {})

        # 排行榜文本
        lb_text = ""
        for row in leaderboard[:8]:
            lb_text += "\n" + " | ".join([f"{k}={v}" for k, v in row.items()])

        # 决策报告
        dec_text = ""
        if decision:
            dec_text = textwrap.dedent(f"""\
                - 推荐模型: {decision.get('recommended_name', 'N/A')}
                - 置信度: {decision.get('confidence', 0):.0%}
                - 推荐理由: {decision.get('recommendation_reason', 'N/A')}
                - 模式: {decision.get('mode_description', 'N/A')}
                - 风险: {', '.join(decision.get('risks', [])) or '无'}
            """)

        # 融合权重
        weights_text = ""
        if ensemble_weights:
            weights_text = "\n".join([f"- {k}: {v:.2f}" for k, v in ensemble_weights.items()])

        content = textwrap.dedent(f"""\
            请分析以下机器学习训练结果：

            ## 任务信息
            - 任务类型: {task_type}
            - 原始特征数: {preprocessing.get('original_features', 'N/A')}
            - 编码后特征数: {preprocessing.get('encoded_features', 'N/A')}
            - 选择后特征数: {preprocessing.get('selected_features', 'N/A')}

            ## 模型排行榜
            {lb_text if lb_text else '无排行榜数据'}

            ## 决策报告
            {dec_text if dec_text else '无决策报告'}

            ## 融合权重
            {weights_text if weights_text else '未启用融合'}

            请提供以下分析（使用 Markdown 格式）：
            1. **模型选择建议** — 推荐哪个模型？为什么？是否需要融合？
            2. **超参数优化建议** — 针对 Top 模型，有哪些调参方向？
            3. **潜在问题诊断** — 是否存在过拟合、欠拟合、数据泄漏等风险？
            4. **进一步提升性能的建议** — 下一步可以做什么？
        """)

        return [
            {"role": "system", "content": cls.SYSTEM_PROMPT},
            {"role": "user", "content": content},
        ]

    @classmethod
    def build_error_prompt(
        cls,
        error_msg: str,
        stack_trace: str,
        context: Dict[str, Any],
    ) -> List[Dict[str, str]]:
        """构建错误诊断 Prompt"""
        data_info = context.get("data_info", {})
        config = context.get("config", {})

        content = textwrap.dedent(f"""\
            请诊断以下机器学习训练过程中的错误：

            ## 错误信息
            ```
            {error_msg}
            ```

            ## 堆栈跟踪
            ```
            {stack_trace[:2000] if stack_trace else '无'}
            ```

            ## 数据上下文
            - 数据规模: {data_info.get('shape', 'N/A')}
            - 任务类型: {config.get('task_type', '自动推断')}
            - 目标列: {config.get('target_col', '无')}
            - 编码策略: {config.get('encoding', 'auto')}
            - 特征选择: {config.get('feature_selection', 'mi')}
            - CV折数: {config.get('n_splits', 5)}

            请提供以下分析（使用 Markdown 格式）：
            1. **错误根因分析** — 这个错误的根本原因是什么？
            2. **修复建议** — 具体的修复步骤是什么？
            3. **预防措施** — 如何避免类似错误再次发生？
        """)

        return [
            {"role": "system", "content": cls.SYSTEM_PROMPT},
            {"role": "user", "content": content},
        ]


class LLMAnalyzer:
    """大模型分析器 orchestrator"""

    def __init__(self, config: LLMConfig):
        self.client = LLMClient(config)

    def analyze(
        self,
        analysis_type: str,
        session_data: Dict[str, Any],
        images: List[Dict[str, Any]] | None = None,
    ) -> str:
        """
        执行 LLM 分析

        Args:
            analysis_type: "eda" | "result" | "error"
            session_data: Flask session 中的数据字典
            images: 已由 Web 层校验的图片 data URL；按 OpenAI 兼容多模态格式附加

        Returns:
            Markdown 格式的分析文本
        """
        builder = AnalysisPromptBuilder()

        if analysis_type == "eda":
            df_info = session_data.get("df_info", {})
            eda_data = session_data.get("eda_data", {})
            messages = builder.build_eda_prompt(df_info, eda_data)

        elif analysis_type == "result":
            result_data = session_data.get("model_result", {})
            messages = builder.build_result_prompt(result_data)

        elif analysis_type == "error":
            error_msg = session_data.get("train_error", "未知错误")
            stack_trace = session_data.get("train_error_stack", "")
            context = {
                "data_info": session_data.get("df_info", {}),
                "config": session_data.get("train_config", {}),
            }
            messages = builder.build_error_prompt(error_msg, stack_trace, context)

        else:
            raise ValueError(f"未知的分析类型: {analysis_type}")

        if images:
            if self.client.config.provider == "ollama":
                raise ValueError("图片输入请使用 DeepSeek Vision 或其他支持 image_url 的 OpenAI 兼容模型")
            messages = attach_multimodal_images(messages, images)

        return self.client.chat_completion(messages)


def attach_multimodal_images(
    messages: List[Dict[str, Any]],
    images: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """将图片以 OpenAI 兼容的 content parts 格式附加到最后一条 user 消息。

    返回新消息列表，不修改 PromptBuilder 生成的原始对象，避免同一会话重试时图片累加。
    """
    if not images:
        return messages
    result = [dict(message) for message in messages]
    target_index = next(
        (index for index in range(len(result) - 1, -1, -1)
         if result[index].get("role") == "user"),
        None,
    )
    if target_index is None:
        raise ValueError("多模态消息缺少 user 消息")

    target = result[target_index]
    text = target.get("content", "")
    parts: List[Dict[str, Any]] = [{"type": "text", "text": str(text)}]
    for image in images:
        data_url = str(image.get("data_url", "")).strip()
        if not data_url:
            raise ValueError("图片数据为空")
        parts.append({
            "type": "image_url",
            "image_url": {"url": data_url, "detail": "high"},
        })
    target["content"] = parts
    return result


def get_default_configs() -> Dict[str, Dict[str, Any]]:
    """获取默认配置模板"""
    return {
        "openai": {
            "name": "OpenAI 兼容 API",
            "base_url": "https://api.openai.com/v1",
            "model_name": "gpt-4o",
            "model_options": ["gpt-4o", "gpt-4o-mini"],
            "needs_api_key": True,
        },
        "deepseek": {
            "name": "DeepSeek API",
            "base_url": "https://api.deepseek.com",
            "model_name": "deepseek-v4-pro",
            "model_options": ["deepseek-v4-pro", "deepseek-v4-flash", "deepseek-v4-flash-vision-exp"],
            "needs_api_key": True,
        },
        "ollama": {
            "name": "Ollama 本地模型",
            "base_url": "http://localhost:11434/v1",
            "model_name": "llama3",
            "needs_api_key": False,
        },
    }
