from .protocol import Paper
import html
import math
import re


framework = """
<!DOCTYPE HTML>
<html>
<head>
  <style>
    .star-wrapper {
      font-size: 1.3em; /* 调整星星大小 */
      line-height: 1; /* 确保垂直对齐 */
      display: inline-flex;
      align-items: center; /* 保持对齐 */
    }
    .half-star {
      display: inline-block;
      width: 0.5em; /* 半颗星的宽度 */
      overflow: hidden;
      white-space: nowrap;
      vertical-align: middle;
    }
    .full-star {
      vertical-align: middle;
    }
  </style>
</head>
<body>

<div>
    __CONTENT__
</div>

<br><br>
<div>
To unsubscribe, remove your email in your Github Action setting.
</div>

</body>
</html>
"""

def get_empty_html():
  block_template = """
  <table border="0" cellpadding="0" cellspacing="0" width="100%" style="font-family: Arial, sans-serif; border: 1px solid #ddd; border-radius: 8px; padding: 16px; background-color: #f9f9f9;">
  <tr>
    <td style="font-size: 20px; font-weight: bold; color: #333;">
        No Papers Today. Take a Rest!
    </td>
  </tr>
  </table>
  """
  return block_template

def _format_text(value: str | None) -> str:
    if value is None:
        return ""
    return html.escape(str(value)).replace("\n", "<br>")


def _format_summary(value: str | None) -> str:
    if value is None:
        return ""

    lines = str(value).replace("\r\n", "\n").replace("\r", "\n").split("\n")
    parts = []
    in_ul = False
    in_ol = False

    def close_lists():
        nonlocal in_ul, in_ol
        if in_ul:
            parts.append("</ul>")
            in_ul = False
        if in_ol:
            parts.append("</ol>")
            in_ol = False

    def inline_markdown(text: str) -> str:
        text = html.escape(text.strip())
        text = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', text)
        text = re.sub(r'__(.+?)__', r'<strong>\1</strong>', text)
        text = re.sub(r'(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)', r'<em>\1</em>', text)
        return text

    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            close_lists()
            parts.append('<div style="height: 8px; line-height: 8px;">&nbsp;</div>')
            continue

        heading = re.match(r'^(#{1,6})\s+(.+)$', line)
        unordered = re.match(r'^[-*+]\s+(.+)$', line)
        ordered = re.match(r'^\d+[.)]\s+(.+)$', line)

        if heading:
            close_lists()
            parts.append(
                '<div style="font-weight: bold; margin: 8px 0 4px 0;">'
                + inline_markdown(heading.group(2))
                + '</div>'
            )
        elif unordered:
            if in_ol:
                parts.append("</ol>")
                in_ol = False
            if not in_ul:
                parts.append('<ul style="margin: 6px 0 6px 20px; padding: 0;">')
                in_ul = True
            parts.append('<li style="margin: 3px 0;">' + inline_markdown(unordered.group(1)) + '</li>')
        elif ordered:
            if in_ul:
                parts.append("</ul>")
                in_ul = False
            if not in_ol:
                parts.append('<ol style="margin: 6px 0 6px 20px; padding: 0;">')
                in_ol = True
            parts.append('<li style="margin: 3px 0;">' + inline_markdown(ordered.group(1)) + '</li>')
        else:
            close_lists()
            parts.append('<p style="margin: 0 0 8px 0;">' + inline_markdown(line) + '</p>')

    close_lists()
    return "".join(parts)


def get_block_html(title:str, authors:str, rate:str, tldr:str, pdf_url:str, affiliations:str=None):
    block_template = """
    <table border="0" cellpadding="0" cellspacing="0" width="100%" style="font-family: Arial, sans-serif; border: 1px solid #ddd; border-radius: 8px; padding: 16px; background-color: #f9f9f9;">
    <tr>
        <td style="font-size: 20px; font-weight: bold; color: #333;">
            {title}
        </td>
    </tr>
    <tr>
        <td style="font-size: 14px; color: #666; padding: 8px 0;">
            {authors}
            <br>
            <i>{affiliations}</i>
        </td>
    </tr>
    <tr>
        <td style="font-size: 14px; color: #333; padding: 8px 0;">
            <strong>Relevance:</strong> {rate}
        </td>
    </tr>
    <tr>
        <td style="font-size: 14px; color: #333; padding: 8px 0;">
            <strong>中文详细总结:</strong><br>{tldr}
        </td>
    </tr>

    <tr>
        <td style="padding: 8px 0;">
            <a href="{pdf_url}" style="display: inline-block; text-decoration: none; font-size: 14px; font-weight: bold; color: #fff; background-color: #d9534f; padding: 8px 16px; border-radius: 4px;">PDF</a>
        </td>
    </tr>
</table>
"""
    return block_template.format(
        title=_format_text(title),
        authors=_format_text(authors),
        rate=_format_text(rate),
        tldr=_format_summary(tldr),
        pdf_url=html.escape(str(pdf_url or "")),
        affiliations=_format_text(affiliations),
    )

def get_stars(score:float):
    full_star = '<span class="full-star">⭐</span>'
    half_star = '<span class="half-star">⭐</span>'
    low = 6
    high = 8
    if score <= low:
        return ''
    elif score >= high:
        return full_star * 5
    else:
        interval = (high-low) / 10
        star_num = math.ceil((score-low) / interval)
        full_star_num = int(star_num/2)
        half_star_num = star_num - full_star_num * 2
        return '<div class="star-wrapper">'+full_star * full_star_num + half_star * half_star_num + '</div>'


def render_email(papers:list[Paper]) -> str:
    parts = []
    if len(papers) == 0 :
        return framework.replace('__CONTENT__', get_empty_html())
    
    for p in papers:
        #rate = get_stars(p.score)
        rate = round(p.score, 1) if p.score is not None else 'Unknown'
        author_list = [a for a in p.authors]
        num_authors = len(author_list)
        if num_authors <= 5:
            authors = ', '.join(author_list)
        else:
            authors = ', '.join(author_list[:3] + ['...'] + author_list[-2:])
        if p.affiliations:
            affiliations = p.affiliations[:5]
            affiliations = ', '.join(affiliations)
            if len(p.affiliations) > 5:
                affiliations += ', ...'
        else:
            affiliations = 'Affiliation not found'
        parts.append(get_block_html(p.title, authors, rate, p.tldr, p.pdf_url, affiliations))

    content = '<br>' + '</br><br>'.join(parts) + '</br>'
    return framework.replace('__CONTENT__', content)
