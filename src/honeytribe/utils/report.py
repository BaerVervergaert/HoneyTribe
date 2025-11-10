from typing import Literal

import matplotlib.pyplot as plt
from pathlib import Path
import io
import base64
import pandas as pd
from datetime import datetime

from jinja2 import Template



class ReportRenderer:
    _TEMPLATE = """
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <title>{{title}}</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                max-width: 1200px;
                margin: 0 auto;
                padding: 20px;
                line-height: 1.6;
            }
            h1 {
                color: #2c3e50;
                border-bottom: 3px solid #3498db;
                padding-bottom: 10px;
            }
            h2 {
                color: #34495e;
                margin-top: 30px;
            }
            .meta {
                color: #7f8c8d;
                font-size: 0.9em;
                margin-bottom: 30px;
            }
            .figure {
                margin: 20px 0;
                text-align: center;
            }
            .figure img {
                max-width: 100%;
                height: auto;
                border: 1px solid #ddd;
                padding: 5px;
            }
            .figure-caption {
                font-style: italic;
                color: #555;
                margin-top: 10px;
            }
            table {
                border-collapse: collapse;
                width: 100%;
                margin: 20px 0;
            }
            th, td {
                border: 1px solid #ddd;
                padding: 12px;
                text-align: left;
            }
            th {
                background-color: #3498db;
                color: white;
            }
            tr:nth-child(even) {
                background-color: #f2f2f2;
            }
        </style>
    </head>
    <body>
        <h1>{{ title }}</h1>
        <div class="meta">Generated on {{ date }}</div>

        {{ content }}

    </body>
    </html>
    """
    @staticmethod
    def _addiction_to_content(foo):
        def wrapper(self, *args, inplace: bool = True, **kwargs):
            result = foo(self, *args, **kwargs)
            if inplace:
                self.add_html(result)
            return result
        return wrapper
    def __init__(self, template: str = None, title: str = "Report"):
        self.template = Template(template if template else self._TEMPLATE)
        self.title = title
        self.parts = []
    def add_html(self, html_content: str) -> None:
        self.parts.append(html_content)
    def add_css(self, css_content: str) -> None:
        style_tag = f'<style>{css_content}</style>'
        self.parts.append(style_tag)
    def partial_render(self, parts: list[str]) -> str:
        content = "\n".join(parts)
        return content
    def render_html(self) -> str:
        content = self.partial_render(self.parts)
        html_output = self.template.render(
            title=self.title,
            date=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            content=content
        )
        return html_output
    def set_title(self, title: str) -> None:
        self.title = title
    def render_pdf(self, output_path: str|Path) -> None:
        # Import weasyprint lazily so it remains an optional dependency
        try:
            import weasyprint  # type: ignore
        except Exception as exc:
            raise ImportError(
                "weasyprint is required to render PDFs. Install the 'visual' extra: pip install .[visual]"
            ) from exc
        output_path = Path(output_path)
        html_content = self.render_html()
        weasyprint.HTML(string=html_content).write_pdf(output_path)
    def save_html(self, output_path: str|Path) -> None:
        output_path = Path(output_path)
        html_content = self.render_html()
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)

    @_addiction_to_content
    def add_section(self, section_title: str, section_content: str) -> str:
        section_html = f"<h2>{section_title}</h2>\n{section_content}"
        return section_html
    @_addiction_to_content
    def add_image(self, image_path: str|Path, alt_text: str = "") -> str:
        image_path = Path(image_path)
        image_html = f'<img src="{image_path.as_uri()}" alt="{alt_text}"/>'
        return image_html
    @_addiction_to_content
    def add_matplotlib_figure(self, fig: plt.Figure, alt_text: str = "") -> str:
        buf = io.BytesIO()
        fig.savefig(buf, format='png')
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        image_html = f'<img src="data:image/png;base64,{img_base64}" alt="{alt_text}"/>'
        return image_html
    @_addiction_to_content
    def add_table(self, df: pd.DataFrame, caption: str = "", **kwargs) -> str:
        table_html = df.to_html(**kwargs)
        if caption:
            table_html = f"<caption>{caption}</caption>\n" + table_html
        return table_html
    @_addiction_to_content
    def add_paragraph(self, text: str) -> str:
        paragraph_html = f"<p>{text}</p>"
        return paragraph_html
    @_addiction_to_content
    def add_heading(self, text: str, level: Literal[1, 2, 3, 4, 5, 6] = 1) -> str:
        heading_html = f"<h{level}>{text}</h{level}>"
        return heading_html
    @_addiction_to_content
    def add_text(self, text: str) -> str:
        return self.add_paragraph(text)