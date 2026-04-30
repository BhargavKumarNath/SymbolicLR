import pytest

try:
    from streamlit.testing.v1 import AppTest

    HAS_APPTEST = True
except ImportError:
    HAS_APPTEST = False


@pytest.mark.skipif(not HAS_APPTEST, reason="Requires Streamlit >= 1.28 for AppTest")
def test_streamlit_app_loads_successfully():
    """
    Boot the Streamlit dashboard headlessly and ensure the core controls render
    without import or runtime exceptions.
    """
    at = AppTest.from_file("app.py")
    at.run(timeout=10)

    assert not at.exception, f"Streamlit app crashed on load: {at.exception}"
    assert any("Start Evolution" in button.label for button in at.button), "Run button did not render."
    assert at.radio, "Navigation radio did not render."
