from dataclasses import dataclass
from typing import Optional, TypeVar
from datetime import datetime
import ast
import re
import tiktoken
from openai import OpenAI
from loguru import logger
import json
RawPaperItem = TypeVar('RawPaperItem')

SUMMARY_LANGUAGE_NAME = "中文"


def _get_generation_kwargs(llm_params: dict) -> dict:
    return dict(llm_params.get('generation_kwargs', {}))


def _truncate_text_for_llm(text: str, max_tokens: int) -> str:
    try:
        enc = tiktoken.get_encoding("cl100k_base")
        tokens = enc.encode(text)
        return enc.decode(tokens[:max_tokens])
    except Exception as exc:
        logger.warning(f"Failed to load tokenizer for prompt truncation: {exc}")
        return text[: max_tokens * 4]


def _parse_affiliation_list(content: str) -> list[str]:
    match = re.search(r'\[.*?\]', content, flags=re.DOTALL)
    if match is not None:
        candidate = match.group(0)
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            parsed = ast.literal_eval(candidate)
    else:
        parsed = [part.strip() for part in re.split(r'[;\n]', content) if part.strip()]

    if not isinstance(parsed, list):
        raise ValueError("Affiliation response is not a list.")

    affiliations = []
    seen = set()
    for affiliation in parsed:
        if isinstance(affiliation, dict):
            affiliation = affiliation.get("affiliation") or affiliation.get("institution")
        if affiliation is None:
            continue
        affiliation = str(affiliation).strip().strip('"\'.')
        if not affiliation or affiliation in {"[]", "None", "null"}:
            continue
        key = affiliation.casefold()
        if key not in seen:
            affiliations.append(affiliation)
            seen.add(key)
    return affiliations


def _strip_latex_markup(text: str) -> str:
    text = re.sub(r'\\(?:textsuperscript|textbf|textit|emph|mathrm|rm|it|bf)\{([^{}]*)\}', r'\1', text)
    text = re.sub(r'\\[a-zA-Z]+\*?(?:\[[^\]]*\])?', ' ', text)
    text = re.sub(r'[{}$^_*~`\\]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip(" ,;")


def _dedupe_preserving_order(values: list[str]) -> list[str]:
    deduped = []
    seen = set()
    for value in values:
        value = value.strip().strip('"\'.')
        if not value:
            continue
        key = value.casefold()
        if key not in seen:
            deduped.append(value)
            seen.add(key)
    return deduped


def _extract_affiliation_context(full_text: str) -> str:
    context_parts = []

    document_match = re.search(r'\\begin\{document\}', full_text)
    document_start = document_match.start() if document_match else 0
    context_parts.append(full_text[document_start:document_start + 12000])

    affiliation_patterns = [
        r'\\(?:affiliation|affil|institute|institution|address|orgname)\*?(?:\[[^\]]*\])?\s*\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}',
        r'\\(?:author|thanks)\*?(?:\[[^\]]*\])?\s*\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}',
    ]
    for pattern in affiliation_patterns:
        context_parts.extend(re.findall(pattern, full_text, flags=re.DOTALL | re.IGNORECASE))

    lines = full_text.splitlines()
    keyword_pattern = re.compile(
        r'affiliat|institute|university|college|school|laborator|department|academy|'
        r'google|microsoft|openai|meta|nvidia|bytedance|tencent|alibaba|huawei',
        re.IGNORECASE,
    )
    selected_lines = []
    for index, line in enumerate(lines):
        if keyword_pattern.search(line):
            start = max(0, index - 2)
            end = min(len(lines), index + 3)
            selected_lines.extend(lines[start:end])
    context_parts.append("\n".join(selected_lines[:300]))

    return _truncate_text_for_llm("\n\n".join(part for part in context_parts if part), 6000)


def _extract_affiliations_by_patterns(text: str) -> list[str]:
    candidates = []
    command_pattern = re.compile(
        r'\\(?:affiliation|affil|institute|institution|address|orgname)\*?(?:\[[^\]]*\])?\s*\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}',
        re.DOTALL | re.IGNORECASE,
    )
    candidates.extend(match.group(1) for match in command_pattern.finditer(text))

    institution_pattern = re.compile(
        r'([A-Z][A-Za-z&.\-\s]+(?:University|Institute|Laboratory|Lab|College|School|Academy|'
        r'Corporation|Inc\.?|Ltd\.?|LLC|Research|Center|Centre|Department)[A-Za-z&.,\-\s]*)'
    )
    candidates.extend(match.group(1) for match in institution_pattern.finditer(_strip_latex_markup(text)))

    cleaned = []
    for candidate in candidates:
        candidate = _strip_latex_markup(candidate)
        candidate = re.sub(r'\b(email|e-mail|corresponding author|equal contribution)\b.*', '', candidate, flags=re.IGNORECASE)
        parts = [part.strip() for part in re.split(r';|\n|\band\b', candidate) if part.strip()]
        for part in parts:
            if len(part) < 4 or "@" in part:
                continue
            cleaned.append(part)
    return _dedupe_preserving_order(cleaned)[:10]

@dataclass
class Paper:
    source: str
    title: str
    authors: list[str]
    abstract: str
    url: str
    pdf_url: Optional[str] = None
    full_text: Optional[str] = None
    tldr: Optional[str] = None
    affiliations: Optional[list[str]] = None
    score: Optional[float] = None

    def _generate_tldr_with_llm(self, openai_client:OpenAI,llm_params:dict) -> str:
        prompt = (
            "请根据下面的论文信息撰写一段中文详细总结。"
            "总结必须使用中文，不要使用英文回答。"
            "请覆盖研究问题、动机、方法、主要技术思路、实验或证据、关键发现，以及文中可见的局限性。"
            "保持内容充实但不要空泛。"
            "不要使用 Markdown、标题符号、项目符号或加粗语法；请输出适合 HTML 邮件直接显示的纯文本段落。\n\n"
        )
        if self.title:
            prompt += f"Title:\n {self.title}\n\n"

        if self.abstract:
            prompt += f"Abstract: {self.abstract}\n\n"

        if self.full_text:
            prompt += f"Preview of main content:\n {self.full_text}\n\n"

        if not self.full_text and not self.abstract:
            logger.warning(f"Neither full text nor abstract is provided for {self.url}")
            return "Failed to generate TLDR. Neither full text nor abstract is provided"
        prompt = _truncate_text_for_llm(prompt, 8000)
        
        response = openai_client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": f"You write accurate, detailed scientific paper summaries. Your entire answer must be in {SUMMARY_LANGUAGE_NAME}. Do not use Markdown formatting.",
                },
                {"role": "user", "content": prompt},
            ],
            **_get_generation_kwargs(llm_params)
        )
        tldr = response.choices[0].message.content
        return tldr
    
    def generate_tldr(self, openai_client:OpenAI,llm_params:dict) -> str:
        try:
            tldr = self._generate_tldr_with_llm(openai_client,llm_params)
            self.tldr = tldr
            return tldr
        except Exception as e:
            logger.warning(f"Failed to generate tldr of {self.url}: {e}")
            tldr = self.abstract
            self.tldr = tldr
            return tldr

    def _generate_affiliations_with_llm(self, openai_client:OpenAI,llm_params:dict) -> Optional[list[str]]:
        if self.full_text is not None:
            authors = ', '.join(self.authors)
            affiliation_context = _extract_affiliation_context(self.full_text)
            prompt = (
                "Extract the top-level institutional affiliations for the paper authors from the paper text. "
                "Return only a JSON array of unique affiliation strings, ordered by first appearance in author order. "
                "Use top-level institutions such as universities, companies, or research institutes. "
                "If no affiliation is present, return [].\n\n"
                f"Title:\n{self.title}\n\nAuthors:\n{authors}\n\nAuthor and affiliation context:\n{affiliation_context}"
            )
            prompt = _truncate_text_for_llm(prompt, 7000)
            response = openai_client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": "You extract author affiliations from scientific papers. Return only valid JSON. Do not include explanations, markdown, or code fences.",
                    },
                    {"role": "user", "content": prompt},
                ],
                **_get_generation_kwargs(llm_params)
            )
            affiliations = _parse_affiliation_list(response.choices[0].message.content)
            if not affiliations:
                affiliations = _extract_affiliations_by_patterns(affiliation_context)
            return affiliations
    
    def generate_affiliations(self, openai_client:OpenAI,llm_params:dict) -> Optional[list[str]]:
        try:
            affiliations = self._generate_affiliations_with_llm(openai_client,llm_params)
            self.affiliations = affiliations
            return affiliations
        except Exception as e:
            logger.warning(f"Failed to generate affiliations of {self.url}: {e}")
            self.affiliations = None
            return None
@dataclass
class CorpusPaper:
    title: str
    abstract: str
    added_date: datetime
    paths: list[str]
