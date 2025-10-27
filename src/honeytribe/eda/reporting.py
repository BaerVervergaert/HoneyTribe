from honeytribe.utils.report import ReportRenderer
from honeytribe.eda.summary import EDASummaryBase, SavedResultsMixin


class EDAReport(ReportRenderer):
    _addiction_to_content = ReportRenderer._addiction_to_content
    def __init__(self, summaries: list[EDASummaryBase], template: str = None, title: str = "Exploratory Data Analysis Report"):
        super().__init__(template=template, title=title)
        self.summaries = summaries
    @_addiction_to_content
    def add_summary(self, summary: EDASummaryBase) -> str:
        if isinstance(summary, SavedResultsMixin) and (not summary.results_available) and isinstance(summary, EDASummaryBase):
            print("Warning: Summary results not available before compute. Computing now.")
            summary.compute()
        if isinstance(summary, SavedResultsMixin) and not summary.results_available:
            raise ValueError("Compute was expected to save results, but didn't")
        name = summary.name
        parts = []
        try:
            table = summary.table()
        except NotImplementedError:
            ...
        else:
            table_part = self.add_table(table, inplace=False)
            parts.append(table_part)
        try:
            plots = summary.plot()
        except NotImplementedError:
            ...
        else:
            for label, (fig, ax) in plots.items():
                plot_part = self.add_matplotlib_figure(fig, alt_text=label, inplace=False)
                parts.append(plot_part)
        if len(parts) >= 1:
            return self.add_section(name, self.partial_render(parts), inplace=False)
        else:
            raise ValueError('Summary has nothing to render!')
    @_addiction_to_content
    def add_summaries(self) -> str:
        parts = []
        for summary in self.summaries:
            part = self.add_summary(summary, inplace=False)
            parts.append(part)
        return self.partial_render(parts)