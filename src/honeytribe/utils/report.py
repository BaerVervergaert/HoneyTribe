from typing import Literal

import matplotlib.pyplot as plt
import weasyprint
from pathlib import Path
import io
import base64
import pandas as pd




class ReportRenderer:
    _TEMPLATE = """
    <html>
    <head>
    <meta charset="utf-8">
    <title>{title}</title>
    </head>
    <body>
    {content}
    </body>
    </html>
    """
    @staticmethod
    def _addiction_to_content(foo):
        def wrapper(self, *args, inplace: bool = False, section_title:str|None = None, **kwargs):
            result = foo(self, *args, **kwargs)
            if section_title:
                result = self.add_section(section_title, result)
            if inplace:
                self.add_html(result)
            return result
        return wrapper
    def __init__(self, template: str = None, title: str = "Report"):
        if template is not None:
            self._TEMPLATE = template
        self.title = title
        self.parts = []
    def add_html(self, html_content: str) -> None:
        self.parts.append(html_content)
    def add_css(self, css_content: str) -> None:
        style_tag = f'<style>{css_content}</style>'
        self.parts.append(style_tag)
    def render_html(self) -> str:
        content = "\n".join(self.parts)
        html_output = self._TEMPLATE.format(
            title=self.title,
            content=content
        )
        return html_output
    def set_title(self, title: str) -> None:
        self.title = title
    def render_pdf(self, output_path: str|Path) -> None:
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