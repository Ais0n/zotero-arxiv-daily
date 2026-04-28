"""Tests for zotero_arxiv_daily.protocol: Paper.generate_tldr, Paper.generate_affiliations."""

import pytest

from tests.canned_responses import make_sample_paper, make_stub_openai_client


@pytest.fixture()
def llm_params():
    return {
        "language": "English",
        "generation_kwargs": {"model": "gpt-4o-mini", "max_tokens": 16384},
    }


# ---------------------------------------------------------------------------
# generate_tldr
# ---------------------------------------------------------------------------


def test_tldr_returns_response(llm_params):
    client = make_stub_openai_client()
    paper = make_sample_paper()
    result = paper.generate_tldr(client, llm_params)
    assert result == "这是一段中文论文总结。"
    assert paper.tldr == result


def test_tldr_without_abstract_or_fulltext(llm_params):
    client = make_stub_openai_client()
    paper = make_sample_paper(abstract="", full_text=None)
    result = paper.generate_tldr(client, llm_params)
    assert "Failed to generate TLDR" in result


def test_tldr_falls_back_to_abstract_on_error(llm_params):
    paper = make_sample_paper()

    # Client whose create() raises
    from types import SimpleNamespace

    broken_client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=lambda **kw: (_ for _ in ()).throw(RuntimeError("API down")))
        )
    )
    result = paper.generate_tldr(broken_client, llm_params)
    assert result == paper.abstract


def test_tldr_truncates_long_prompt(llm_params):
    client = make_stub_openai_client()
    paper = make_sample_paper(full_text="word " * 10000)
    result = paper.generate_tldr(client, llm_params)
    assert result is not None


def test_tldr_forces_chinese_even_when_config_says_english(llm_params):
    from types import SimpleNamespace

    captured = {}

    def create(**kwargs):
        captured["messages"] = kwargs["messages"]
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="中文总结"))]
        )

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    paper = make_sample_paper()
    assert paper.generate_tldr(client, llm_params) == "中文总结"
    request_text = str(captured["messages"])
    assert "中文" in request_text
    assert "must be in 中文" in request_text


def test_tldr_uses_configured_model(llm_params):
    from types import SimpleNamespace

    captured = {}

    def create(**kwargs):
        captured["model"] = kwargs["model"]
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="中文总结"))]
        )

    llm_params["generation_kwargs"]["model"] = "custom-model"
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    paper = make_sample_paper()
    paper.generate_tldr(client, llm_params)

    assert captured["model"] == "custom-model"


# ---------------------------------------------------------------------------
# generate_affiliations
# ---------------------------------------------------------------------------


def test_affiliations_returns_parsed_list(llm_params):
    client = make_stub_openai_client()
    paper = make_sample_paper()
    result = paper.generate_affiliations(client, llm_params)
    assert isinstance(result, list)
    assert "TsingHua University" in result
    assert "Peking University" in result


def test_affiliations_none_without_fulltext(llm_params):
    client = make_stub_openai_client()
    paper = make_sample_paper(full_text=None)
    result = paper.generate_affiliations(client, llm_params)
    assert result is None


def test_affiliations_deduplicates(llm_params):
    """The stub returns two distinct affiliations, so no dedup needed.
    But confirm the set() dedup in the code doesn't break anything.
    """
    client = make_stub_openai_client()
    paper = make_sample_paper()
    result = paper.generate_affiliations(client, llm_params)
    assert len(result) == len(set(result))


def test_affiliations_malformed_llm_output(llm_params):
    """LLM returns affiliations without JSON brackets. Should still fall back gracefully."""
    from types import SimpleNamespace

    def create_no_brackets(**kwargs):
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content="TsingHua University, Peking University"),
                )
            ]
        )

    client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=create_no_brackets)
        )
    )
    paper = make_sample_paper()
    result = paper.generate_affiliations(client, llm_params)
    assert result == ["TsingHua University, Peking University"]


def test_affiliations_error_returns_none(llm_params):
    from types import SimpleNamespace

    broken_client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=lambda **kw: (_ for _ in ()).throw(RuntimeError("boom")))
        )
    )
    paper = make_sample_paper()
    result = paper.generate_affiliations(broken_client, llm_params)
    assert result is None
    assert paper.affiliations is None


def test_affiliations_pattern_fallback_after_empty_llm_response(llm_params):
    from types import SimpleNamespace

    def create_empty_list(**kwargs):
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="[]"))]
        )

    client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create_empty_list))
    )
    paper = make_sample_paper(
        full_text=(
            "\\newcommand{\\lots}{x}\n" * 500
            + "\\begin{document}\n"
            + "\\title{Sample}\n"
            + "\\author{Alice}\n"
            + "\\affiliation{Department of Computer Science, Example University}\n"
            + "\\begin{abstract}Test abstract.\\end{abstract}"
        )
    )

    result = paper.generate_affiliations(client, llm_params)
    assert any("Example University" in affiliation for affiliation in result)
