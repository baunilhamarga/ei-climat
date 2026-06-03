from abc import ABC, abstractmethod
# pyrefly: ignore [missing-import]
import pandas as pd

class BaseTab(ABC):
    """Abstract base class representing a tab panel within the dashboard."""
    
    @abstractmethod
    def render(self, data: dict[str, pd.DataFrame], selected_acorns: list[str]) -> None:
        """Renders the components of this tab.
        
        Args:
            data: Dictionary containing pandas DataFrames of the cached artifacts.
            selected_acorns: List of currently selected ACORN groups from the UI.
        """
        pass
