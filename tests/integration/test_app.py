import pytest
import sys
from unittest.mock import patch, MagicMock

try:
    from streamlit.testing.v1 import AppTest
    HAS_APPTEST = True
except ImportError:
    HAS_APPTEST = False

@pytest.mark.skipif(not HAS_APPTEST, reason="Requires Streamlit >= 1.28 for AppTest")
def test_streamlit_app_loads_successfully():
    """
    Programmatically boots the Streamlit application in a headless state 
    to ensure the UI layout renders without import or compilation exceptions.
    """
    # Initialize the AppTest environment pointing to our app
    at = AppTest.from_file("app.py")
    
    at.run(timeout=10)
    
    assert not at.exception, f"Streamlit app crashed on load: {at.exception}"
    
    # Validate the title rendered correctly
    assert "SymboLR" in at.title[0].value

def test_plot_helpers():
    """Validates the Altair chart generation logic safely handles empty states."""
    from app import plot_archive, plot_schedules
    from gp.map_elites import MAPElitesArchive
    import numpy as np
    
    archive = MAPElitesArchive(size_bins=10, com_bins=10, time_steps=10)
    
    # Should safely return an empty Streamlit block representation
    chart1 = plot_archive(archive)
    assert chart1 is not None
    
    chart2 = plot_schedules([], np.array([0.0, 1.0]))
    assert chart2 is not None