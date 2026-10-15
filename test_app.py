# Tests that the Qubit Lab app loads and responds to clicks. Run with:  pytest

from streamlit.testing.v1 import AppTest


def test_app_loads():
    app = AppTest.from_file("app.py").run(timeout=30)
    assert not app.exception


def test_bell_preset_builds_entangled_state():
    app = AppTest.from_file("app.py").run(timeout=30)
    bell = [b for b in app.button if b.label == "Bell pair"][0]
    bell.click().run(timeout=30)
    assert not app.exception
    text = " ".join(m.value for m in app.markdown)
    assert "0.707|00⟩ + 0.707|11⟩" in text


def test_add_gate_and_undo():
    app = AppTest.from_file("app.py").run(timeout=30)
    add = [b for b in app.button if b.label == "Add gate"][0]
    add.click().run(timeout=30)
    assert "0.707|00⟩ + 0.707|10⟩" in " ".join(m.value for m in app.markdown)  # H on q0
    undo = [b for b in app.button if b.label == "Undo"][0]
    undo.click().run(timeout=30)
    assert "|ψ⟩ = 1|00⟩" in " ".join(m.value for m in app.markdown)


def test_grover_tab_with_more_qubits():
    app = AppTest.from_file("app.py").run(timeout=30)
    app.slider(key="grover_qubits").set_value(6).run(timeout=30)
    assert not app.exception
