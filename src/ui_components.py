"""Local interactive UI assets; model inputs never enter executable markup."""
from pathlib import Path

from streamlit.components.v2 import component

ASSETS = Path(__file__).resolve().parents[1] / "assets"

crystal_viewer = component(
    "perovskite_crystal",
    html=(ASSETS / "crystal.html").read_text(encoding="utf-8"),
    css=(ASSETS / "crystal.css").read_text(encoding="utf-8"),
    js=(ASSETS / "crystal.js").read_text(encoding="utf-8"),
)
