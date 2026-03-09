import marimo

__generated_with = "0.20.4"
app = marimo.App()


@app.cell
def _():
    import datetime
    import marimo as mo
    return datetime, mo


@app.cell
def _(datetime, mo):
    # --- Widget definitions ---
    # Reading .value on any widget in a downstream cell makes that cell reactive.

    slider = mo.ui.slider(start=0, stop=100, value=50, label="Slider Value")
    dropdown = mo.ui.dropdown(
        options=["Option A", "Option B", "Option C"],
        value="Option A",
        label="Dropdown Choice",
    )
    text_input = mo.ui.text(placeholder="Type something...", label="User Input")
    checkbox = mo.ui.switch(value=False, label="Enable Feature X")
    date_picker = mo.ui.date(value=datetime.date.today(), label="Select Date")
    button = mo.ui.button(label="Update Manually")
    return button, checkbox, date_picker, dropdown, slider, text_input


@app.cell
def _(button, checkbox, date_picker, dropdown, mo, slider, text_input):
    # Status display — re-runs automatically whenever any widget value changes.
    # button.value tracks click count, so clicking it also triggers a refresh.
    _status = mo.md(f"""
    ### Current State
    * **Slider:** {slider.value}
    * **Dropdown:** {dropdown.value}
    * **Text:** "{text_input.value}"
    * **Feature X enabled:** {checkbox.value}
    * **Date:** {date_picker.value}
    * *(Manual refreshes: {button.value})*
    """)

    _tab1 = mo.vstack([mo.md("#### Basic Inputs"), slider, dropdown, text_input])
    _tab2 = mo.vstack([mo.md("#### Extra Settings"), checkbox, date_picker, button])

    mo.vstack([
        mo.md("## Component Test Dashboard"),
        mo.tabs({"Main Controls": _tab1, "Settings": _tab2}),
        mo.md("---"),
        _status,
    ])


if __name__ == "__main__":
    app.run()
