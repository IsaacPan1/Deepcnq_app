"""Generate the concise CNQ end-user manual.

Run from cnq_app:

    python make_manual.py
    python make_manual.py --out MANUAL_REVISED.docx

The generated Word manual is committed alongside this script for convenience.
"""
from __future__ import annotations

import argparse
from pathlib import Path


HERE = Path(__file__).resolve().parent
OUT = HERE / "MANUAL.docx"
STARTUP_INFOGRAPHIC = HERE / "docs" / "images" / "cnq-start-three-steps.png"
WORKFLOW_INFOGRAPHIC = HERE / "docs" / "images" / "cnq-interface-workflow.png"


def build(path: Path) -> Path:
    try:
        from docx import Document
        from docx.enum.table import WD_ALIGN_VERTICAL, WD_TABLE_ALIGNMENT
        from docx.enum.text import WD_ALIGN_PARAGRAPH
        from docx.oxml import OxmlElement
        from docx.oxml.ns import qn
        from docx.shared import Inches, Pt, RGBColor
    except ImportError as exc:  # noqa: BLE001
        raise SystemExit("python-docx is required: pip install python-docx") from exc

    doc = Document()
    section = doc.sections[0]
    section.top_margin = Inches(0.78)
    section.bottom_margin = Inches(0.72)
    section.left_margin = Inches(0.88)
    section.right_margin = Inches(0.88)

    navy = "1F2A55"
    pale_blue = "F1F5FB"
    border_gray = "D9D9D9"

    def set_style_font(style_name, name, size, *, bold=False):
        style = doc.styles[style_name]
        style.font.name = name
        style.font.size = Pt(size)
        style.font.bold = bold
        style.font.color.rgb = RGBColor(0, 0, 0)
        style._element.rPr.rFonts.set(qn("w:ascii"), name)
        style._element.rPr.rFonts.set(qn("w:hAnsi"), name)
        return style

    normal = set_style_font("Normal", "Aptos", 11.2)
    normal.paragraph_format.space_after = Pt(6)
    normal.paragraph_format.line_spacing = 1.08

    title_style = set_style_font("Title", "Aptos Display", 28, bold=True)
    title_style.paragraph_format.space_after = Pt(12)
    title_p_pr = title_style._element.get_or_add_pPr()
    title_border = title_p_pr.find(qn("w:pBdr"))
    if title_border is not None:
        title_p_pr.remove(title_border)

    h1_style = set_style_font("Heading 1", "Aptos Display", 18, bold=True)
    h1_style.paragraph_format.space_before = Pt(16)
    h1_style.paragraph_format.space_after = Pt(8)
    h1_style.paragraph_format.keep_with_next = True

    h2_style = set_style_font("Heading 2", "Aptos Display", 13.5, bold=True)
    h2_style.paragraph_format.space_before = Pt(12)
    h2_style.paragraph_format.space_after = Pt(5)
    h2_style.paragraph_format.keep_with_next = True

    for list_style in ("List Bullet", "List Number"):
        style = set_style_font(list_style, "Aptos", 11.2)
        style.paragraph_format.space_after = Pt(4)

    def h1(text):
        doc.add_heading(text, level=1)

    def h2(text):
        doc.add_heading(text, level=2)

    def para(text="", *, align=None, keep=False):
        paragraph = doc.add_paragraph()
        paragraph.add_run(text)
        if align is not None:
            paragraph.alignment = align
        if keep:
            paragraph.paragraph_format.keep_with_next = True
        return paragraph

    def labeled_para(label, text):
        paragraph = doc.add_paragraph()
        lead = paragraph.add_run(label)
        lead.bold = True
        paragraph.add_run(text)
        return paragraph

    def bullets(items):
        for item in items:
            doc.add_paragraph(str(item), style="List Bullet")

    def steps(items):
        for number, item in enumerate(items, start=1):
            paragraph = doc.add_paragraph()
            paragraph.paragraph_format.left_indent = Inches(0.30)
            paragraph.paragraph_format.first_line_indent = Inches(-0.24)
            paragraph.paragraph_format.space_after = Pt(4)
            paragraph.add_run(f"{number}.  ")
            paragraph.add_run(str(item))

    def code(text):
        paragraph = doc.add_paragraph()
        paragraph.paragraph_format.left_indent = Inches(0.22)
        paragraph.paragraph_format.right_indent = Inches(0.22)
        paragraph.paragraph_format.space_before = Pt(2)
        paragraph.paragraph_format.space_after = Pt(6)
        paragraph.paragraph_format.keep_together = True
        run = paragraph.add_run(text)
        run.font.name = "Consolas"
        run._element.rPr.rFonts.set(qn("w:ascii"), "Consolas")
        run._element.rPr.rFonts.set(qn("w:hAnsi"), "Consolas")
        run.font.size = Pt(9.8)
        shading = OxmlElement("w:shd")
        shading.set(qn("w:fill"), "F3F3F3")
        paragraph._p.get_or_add_pPr().append(shading)
        return paragraph

    def figure(image_path, alt_text, caption, width):
        if not image_path.exists():
            raise FileNotFoundError(f"Manual image not found: {image_path}")
        paragraph = doc.add_paragraph()
        paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
        paragraph.paragraph_format.space_before = Pt(5)
        paragraph.paragraph_format.space_after = Pt(3)
        paragraph.paragraph_format.keep_with_next = True
        shape = paragraph.add_run().add_picture(str(image_path), width=Inches(width))
        shape._inline.docPr.set("descr", alt_text)
        caption_paragraph = doc.add_paragraph()
        caption_paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
        caption_paragraph.paragraph_format.space_after = Pt(10)
        caption_run = caption_paragraph.add_run(caption)
        caption_run.italic = True
        caption_run.font.size = Pt(9.5)
        caption_run.font.color.rgb = RGBColor.from_string("666666")

    def set_cell_shading(cell, fill):
        tc_pr = cell._tc.get_or_add_tcPr()
        shading = tc_pr.find(qn("w:shd"))
        if shading is None:
            shading = OxmlElement("w:shd")
            tc_pr.append(shading)
        shading.set(qn("w:fill"), fill)

    def set_cell_borders(cell):
        tc_pr = cell._tc.get_or_add_tcPr()
        borders = tc_pr.first_child_found_in("w:tcBorders")
        if borders is None:
            borders = OxmlElement("w:tcBorders")
            tc_pr.append(borders)
        for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
            tag = "w:" + edge
            element = borders.find(qn(tag))
            if element is None:
                element = OxmlElement(tag)
                borders.append(element)
            element.set(qn("w:val"), "single")
            element.set(qn("w:sz"), "6")
            element.set(qn("w:color"), border_gray)

    def set_cell_margins(cell, top=100, start=120, bottom=100, end=120):
        tc_pr = cell._tc.get_or_add_tcPr()
        margins = tc_pr.first_child_found_in("w:tcMar")
        if margins is None:
            margins = OxmlElement("w:tcMar")
            tc_pr.append(margins)
        for name, value in (("top", top), ("start", start),
                            ("bottom", bottom), ("end", end)):
            node = margins.find(qn("w:" + name))
            if node is None:
                node = OxmlElement("w:" + name)
                margins.append(node)
            node.set(qn("w:w"), str(value))
            node.set(qn("w:type"), "dxa")

    def table(headers, rows, widths):
        word_table = doc.add_table(rows=1, cols=len(headers))
        word_table.alignment = WD_TABLE_ALIGNMENT.CENTER
        word_table.autofit = False
        header_properties = word_table.rows[0]._tr.get_or_add_trPr()
        repeat_header = OxmlElement("w:tblHeader")
        repeat_header.set(qn("w:val"), "true")
        header_properties.append(repeat_header)

        for index, header in enumerate(headers):
            cell = word_table.rows[0].cells[index]
            cell.text = ""
            run = cell.paragraphs[0].add_run(header)
            run.bold = True
            run.font.color.rgb = RGBColor(255, 255, 255)
            cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
            cell.width = Inches(widths[index])
            set_cell_shading(cell, navy)
            set_cell_borders(cell)
            set_cell_margins(cell)

        for row_index, row in enumerate(rows, start=1):
            cells = word_table.add_row().cells
            for index, value in enumerate(row):
                cell = cells[index]
                cell.text = str(value)
                cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
                cell.width = Inches(widths[index])
                set_cell_borders(cell)
                set_cell_margins(cell)
                if row_index % 2 == 0:
                    set_cell_shading(cell, pale_blue)

        for row in word_table.rows:
            row_properties = row._tr.get_or_add_trPr()
            cant_split = OxmlElement("w:cantSplit")
            row_properties.append(cant_split)
            for cell in row.cells:
                for paragraph in cell.paragraphs:
                    paragraph.paragraph_format.space_after = Pt(0)
                    paragraph.paragraph_format.line_spacing = 1.05
                    for run in paragraph.runs:
                        run.font.name = "Aptos"
                        run._element.rPr.rFonts.set(qn("w:ascii"), "Aptos")
                        run._element.rPr.rFonts.set(qn("w:hAnsi"), "Aptos")
                        run.font.size = Pt(10.2)
        doc.add_paragraph()

    def add_page_number(paragraph):
        paragraph.alignment = WD_ALIGN_PARAGRAPH.RIGHT
        run = paragraph.add_run("CNQ User Manual   |   ")
        run.font.size = Pt(9)
        run.font.color.rgb = RGBColor.from_string("666666")
        begin = OxmlElement("w:fldChar")
        begin.set(qn("w:fldCharType"), "begin")
        instruction = OxmlElement("w:instrText")
        instruction.set(qn("xml:space"), "preserve")
        instruction.text = " PAGE "
        end = OxmlElement("w:fldChar")
        end.set(qn("w:fldCharType"), "end")
        run._r.extend([begin, instruction, end])

    add_page_number(section.footer.paragraphs[0])

    # Cover
    title = doc.add_paragraph(style="Title")
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    title.add_run("CNQ User Manual")
    subtitle = para("Quick guide for local censored quantile modeling",
                    align=WD_ALIGN_PARAGRAPH.CENTER)
    subtitle.runs[0].font.size = Pt(15)
    subtitle.runs[0].font.bold = True
    para("Start the app, train and save a model, then predict new subjects or project "
         "population trends. CNQ runs on your computer and uses a browser as its interface.",
         align=WD_ALIGN_PARAGRAPH.CENTER)
    para("After CNQ starts, routine use does not require Python programming.",
         align=WD_ALIGN_PARAGRAPH.CENTER)
    labeled_para("For company data. ",
                 "Follow your organization's approved software, storage, privacy, and "
                 "model validation procedures.")
    doc.add_page_break()

    # 1 Start CNQ
    h1("1 Start CNQ")
    para("The same startup procedure applies on Windows, macOS, and Linux. The first "
         "run installs the required packages and can take several minutes.")
    figure(
        STARTUP_INFOGRAPHIC,
        "Three step CNQ startup process: obtain CNQ, run python run.py, and open the local browser address.",
        "Figure 1. Start CNQ in three steps.",
        6.00,
    )

    h2("1 1 Obtain the app")
    bullets([
        "Approved zip: extract the full deepquant package to a local folder that is not synchronized by OneDrive, iCloud, or Dropbox.",
        "GitHub: clone recursively so the deepcnq submodule is included.",
    ])
    code("git clone --recursive https://github.com/IsaacPan1/Deepcnq_app")
    para("Do not use GitHub Download ZIP for the source repository because it may omit "
         "the deepcnq submodule.")

    h2("1 2 Go to the app folder and run CNQ")
    para("Open Terminal on macOS or Linux, or PowerShell or Command Prompt on Windows.")
    labeled_para("Easiest method. ",
                 "Type cd followed by one space. Drag the Deepcnq_app folder from File "
                 "Explorer or Finder into the terminal, then press Enter.")
    code("cd ")
    labeled_para("Or type the folder path. ",
                 "Put quotation marks around the path, especially when a folder name contains spaces.")
    code('cd "C:\\Users\\YourName\\Downloads\\Deepcnq_app"')
    para("Once the terminal is inside the main Deepcnq_app folder, copy and run:")
    code("cd cnq_app\n"
         "python run.py")
    para("The first time you run CNQ, allow 5-10 minutes for packages to download and "
         "wait until the terminal prompts you to open http://127.0.0.1:8000/.")
    labeled_para("To stop CNQ. ", "Return to the terminal and press Ctrl+C.")

    # 2 Prepare data
    h1("2 Prepare your data")
    para("Upload one CSV file with one header row and one row per subject. The app "
         "provides a sample dataset and a downloadable template.")
    bullets([
        "Follow up time: a positive number using one consistent unit, such as days or months.",
        "Event: 1 for an observed event and 0 for censoring, or two values that you map in CNQ.",
        "Covariates: numeric predictors. Represent categories with numeric indicator columns.",
        "Subject ID: an optional unique label that CNQ does not use as a predictor.",
        "Do not leave blank cells in selected columns, and keep one row per subject.",
        "Aim for about 100 observed events and at least 10 events in each data split.",
    ])

    h2("2 1 Local storage and privacy")
    para("CNQ runs locally, but uploaded files are not temporary. Training, prediction, "
         "and cohort CSV files are copied to cnq_app/jobs/uploads. Recent run folders "
         "under cnq_app/jobs also contain reports, plots, predictions, model artifacts, "
         "and result zip files. A training result zip includes a copy of the uploaded "
         "data named input_data.csv.")
    para("CNQ keeps the 20 most recent run folders by default, but uploaded files are "
         "not removed automatically. Treat the CNQ folder as containing company data "
         "and follow your organization's retention and deletion procedures.")

    # 3 Train and save
    training_heading = doc.add_heading("3 Train and save a model", level=1)
    training_heading.paragraph_format.page_break_before = True
    figure(
        WORKFLOW_INFOGRAPHIC,
        "CNQ workflow: import data, select covariates, Auto tune, save the model, then predict new subjects or project population trends.",
        "Figure 2. The standard CNQ workflow.",
        6.20,
    )
    steps([
        "Import the training CSV on the Train a model tab.",
        "Select the follow up time, event, optional subject ID, and numeric covariates.",
        "Leave Auto-tune selected and choose Standard search effort for a typical analysis.",
        "Select Find best settings, review the leaderboard, and select Use best settings.",
        "Select Start training, resolve any validation warnings, and review the results.",
        "When training finishes, select Save model, choose what to save, enter a name, and save it.",
    ])
    para("Training and Auto-tune do not automatically create a reusable model. Save model "
         "writes a self-contained .cnqmodel file to cnq_app/models. It remains available "
         "after restarting CNQ and appears in Predict and Project. If it is not listed, "
         "select Refresh list.")
    para("Saved models are not pruned with old runs. Download and archive any approved "
         "model that must be retained outside the app folder.")

    h2("3 1 Model choices")
    table(["Model", "Use"], [
        ["KAN CNQ", "Standard numeric tabular option. Included in Auto-tune."],
        ["MLP CNQ", "Numeric tabular baseline. Included in Auto-tune."],
        ["Trans CNQ or TransKAN CNQ", "Advanced option for approved numeric features derived from text or annotations, when attention is scientifically justified."],
    ], widths=[2.05, 4.35])
    labeled_para("Advanced settings. ",
                 "Use them only when the analysis plan requires specific model or "
                 "hyperparameter values. Raw text is not accepted; convert it to approved "
                 "numeric covariates before creating the CSV.")
    # 4 Predict
    h1("4 Predict new subjects")
    steps([
        "Choose a saved model or upload a .cnqmodel file.",
        "Upload a CSV containing the new subjects.",
        "Confirm that every model feature is mapped to the correct column. Use the model template if names do not match.",
        "Optionally select an ID column. Select time and event only when known outcomes are available for external validation.",
        "Run prediction and download the predictions CSV, HTML report, or results zip.",
    ])
    bullets([
        "The median and interval describe outcome variation among similar subjects, not uncertainty in the fitted model parameters.",
        "An out_of_range value of 1 means at least one covariate is outside its training range and the prediction is an extrapolation.",
        "External validation must use subjects that were not used for training or model selection.",
    ])

    # 5 Project
    h1("5 Project population trends")
    para("Use Project trends for aggregate questions such as expected events by a time "
         "or the time required to reach a target number of events.")
    table(["Projection basis", "What CNQ uses"], [
        ["Population size N", "The training population Kaplan Meier curve, scaled to N subjects. CNQ does not simulate covariates in this mode."],
        ["Uploaded cohort", "The uploaded subjects and their mapped covariates. Use this when the cohort mix differs from the training population."],
    ], widths=[2.0, 4.4])
    labeled_para("Sample cohort button. ",
                 "Select Upload a cohort and then Use simulated sample cohort to load 20 "
                 "simulated rows with the columns expected by the selected model.")
    steps([
        "Choose the model and projection basis.",
        "Enter one or more time points or a target number of events.",
        "Run the projection and review the cumulative event curve and uncertainty band.",
        "Download the projection CSV, report, or results zip.",
    ])
    para("CNQ projects only through the last observed event time in the training data. "
         "Times or targets beyond that follow up horizon are flagged and not extrapolated.")

    # 6 Interpretation
    interpretation_heading = doc.add_heading("6 Interpret results safely", level=1)
    interpretation_heading.paragraph_format.page_break_before = True
    bullets([
        "Predictions assume that new subjects resemble the training population. A different site, era, protocol, or inclusion rule can reduce calibration.",
        "Quantile intervals and projection bands describe outcome variation; neither represents every source of model parameter uncertainty.",
        "Out of range subjects require additional review, and external validation must use subjects not used for training or model selection.",
        "Company use requires an approved validation plan and independent assessment on relevant data.",
    ])

    # 7 Optional demo
    h1("7 Optional simulated demo")
    para("The demo is optional, does not open automatically, and is not required each "
         "time CNQ starts. All demo records are simulated.")
    steps([
        "On Train a model, select Use demo training data and start training.",
        "On Predict new subjects, choose Demo model simulated data. Build it first if prompted.",
        "Load 20 new patients and run prediction. The larger demo cohorts illustrate external validation and population shift.",
    ])

    # 8 Troubleshooting
    h1("8 Troubleshooting")
    table(["Problem", "Action"], [
        ["Python is not found", "Install Python 3.11, open a new terminal, and try python run.py or python3 run.py."],
        ["Packages fail to install", "Check the internet connection and run python run.py --reinstall."],
        ["Port 8000 is in use", "Run python run.py 8001 and open http://127.0.0.1:8001/."],
        ["Page is unformatted", "Use the exact local address and hard refresh with Ctrl+Shift+R or Cmd+Shift+R."],
        ["Projection is unavailable", "Save or rebuild the model with the current CNQ version so it includes the training curve."],
    ], widths=[2.15, 4.25])

    # 9 Update
    h1("9 Update a GitHub copy")
    para("This applies only to Git clones. Stop CNQ and run the following commands from cnq_app:")
    code("cd ..\n"
         "git pull --recurse-submodules")
    para("For an approved zip release, use your organization's approved update process.")

    path.parent.mkdir(parents=True, exist_ok=True)
    doc.save(str(path))
    return path


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Build the concise CNQ user manual.")
    parser.add_argument("-o", "--out", default=str(OUT), help="output .docx path")
    args = parser.parse_args(argv)
    out = build(Path(args.out))
    print(f"Wrote {out} ({out.stat().st_size // 1024} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
