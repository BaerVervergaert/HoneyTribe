from honeytribe.utils.report import ReportRenderer

class EDAReport(ReportRenderer):
    _addiction_to_content = ReportRenderer._addiction_to_content
    def __init__(self, template: str = None, title: str = "Exploratory Data Analysis Report"):
        super().__init__(template=template, title=title)