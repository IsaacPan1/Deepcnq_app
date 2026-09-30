"""Generate ``MANUAL.docx`` -- the end-user manual -- with python-docx.

Run from ``cnq_app/`` (needs the optional docs dependency):

    pip install python-docx
    python make_manual.py            # writes MANUAL.docx
    python make_manual.py -o out.docx

The manual is kept as this script (version-controlled, reproducible); the built
``MANUAL.docx`` is committed alongside it for convenience.
"""
from __future__ import annotations

import argparse
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "MANUAL.docx"


def build(path: Path) -> Path:
    try:
        from docx import Document
        from docx.enum.table import WD_ALIGN_VERTICAL, WD_TABLE_ALIGNMENT
        from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_LINE_SPACING
        from docx.oxml import OxmlElement
        from docx.oxml.ns import qn
        from docx.shared import Inches, Pt, RGBColor
    except ImportError as exc:  # noqa: BLE001
        raise SystemExit("python-docx is required: pip install python-docx") from exc

    doc = Document()
    section = doc.sections[0]
    section.top_margin = Inches(0.72)
    section.bottom_margin = Inches(0.68)
    section.left_margin = Inches(0.82)
    section.right_margin = Inches(0.82)

    def set_style_font(style_name, name, size, *, bold=False, color="000000"):
        style = doc.styles[style_name]
        style.font.name = name
        style.font.size = Pt(size)
        style.font.bold = bold
        style.font.color.rgb = RGBColor.from_string(color)
        style._element.rPr.rFonts.set(qn("w:ascii"), name)
        style._element.rPr.rFonts.set(qn("w:hAnsi"), name)
        return style

    normal = set_style_font("Normal", "Aptos", 10.5)
    normal.paragraph_format.space_after = Pt(5)
    normal.paragraph_format.line_spacing_rule = WD_LINE_SPACING.SINGLE
    title_style = set_style_font("Title", "Aptos Display", 28, bold=True)
    title_style.paragraph_format.space_after = Pt(8)
    h1_style = set_style_font("Heading 1", "Aptos Display", 18, bold=True)
    h1_style.paragraph_format.space_before = Pt(14)
    h1_style.paragraph_format.space_after = Pt(6)
    h1_style.paragraph_format.keep_with_next = True
    h2_style = set_style_font("Heading 2", "Aptos Display", 13, bold=True)
    h2_style.paragraph_format.space_before = Pt(10)
    h2_style.paragraph_format.space_after = Pt(4)
    h2_style.paragraph_format.keep_with_next = True
    for list_style in ("List Bullet", "List Number"):
        s = set_style_font(list_style, "Aptos", 10.5)
        s.paragraph_format.space_after = Pt(3)

    navy = "1F2A55"
    pale_blue = "F1F5FB"
    border_gray = "D9D9D9"

    # ---- helpers ---------------------------------------------------------- #
    def h1(t):
        doc.add_heading(t, level=1)

    def h2(t):
        doc.add_heading(t, level=2)

    def para(t="", *, bold=False, italic=False, align=None, keep=False):
        p = doc.add_paragraph()
        run = p.add_run(t)
        run.bold, run.italic = bold, italic
        if align is not None:
            p.alignment = align
        if keep:
            p.paragraph_format.keep_with_next = True
        return p

    def labeled_para(label, text):
        p = doc.add_paragraph()
        lead = p.add_run(label)
        lead.bold = True
        p.add_run(text)
        return p

    def bullets(items):
        for it in items:
            doc.add_paragraph(str(it), style="List Bullet")

    def steps(items):
        for it in items:
            doc.add_paragraph(str(it), style="List Number")

    def code(lines):
        p = doc.add_paragraph()
        p.paragraph_format.left_indent = Inches(0.22)
        p.paragraph_format.right_indent = Inches(0.22)
        p.paragraph_format.space_before = Pt(3)
        p.paragraph_format.space_after = Pt(7)
        p.paragraph_format.keep_together = True
        run = p.add_run(lines)
        run.font.name = "Consolas"
        run._element.rPr.rFonts.set(qn("w:ascii"), "Consolas")
        run._element.rPr.rFonts.set(qn("w:hAnsi"), "Consolas")
        run.font.size = Pt(9.5)
        shd = OxmlElement("w:shd")
        shd.set(qn("w:fill"), "F3F3F3")
        p._p.get_or_add_pPr().append(shd)
        return p

    def set_cell_shading(cell, fill):
        tc_pr = cell._tc.get_or_add_tcPr()
        shd = tc_pr.find(qn("w:shd"))
        if shd is None:
            shd = OxmlElement("w:shd")
            tc_pr.append(shd)
        shd.set(qn("w:fill"), fill)

    def set_cell_borders(cell, color=border_gray, size="6"):
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
            element.set(qn("w:sz"), size)
            element.set(qn("w:color"), color)

    def set_cell_margins(cell, top=90, start=110, bottom=90, end=110):
        tc_pr = cell._tc.get_or_add_tcPr()
        tc_mar = tc_pr.first_child_found_in("w:tcMar")
        if tc_mar is None:
            tc_mar = OxmlElement("w:tcMar")
            tc_pr.append(tc_mar)
        for name, value in (("top", top), ("start", start),
                            ("bottom", bottom), ("end", end)):
            node = tc_mar.find(qn("w:" + name))
            if node is None:
                node = OxmlElement("w:" + name)
                tc_mar.append(node)
            node.set(qn("w:w"), str(value))
            node.set(qn("w:type"), "dxa")

    def table(headers, rows, widths=None):
        t = doc.add_table(rows=1, cols=len(headers))
        t.alignment = WD_TABLE_ALIGNMENT.CENTER
        t.autofit = False
        tr_pr = t.rows[0]._tr.get_or_add_trPr()
        tbl_header = OxmlElement("w:tblHeader")
        tbl_header.set(qn("w:val"), "true")
        tr_pr.append(tbl_header)
        for i, head in enumerate(headers):
            cell = t.rows[0].cells[i]
            cell.text = ""
            run = cell.paragraphs[0].add_run(head)
            run.bold = True
            run.font.color.rgb = RGBColor(255, 255, 255)
            cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
            set_cell_shading(cell, navy)
            set_cell_borders(cell)
            set_cell_margins(cell)
            if widths:
                cell.width = Inches(widths[i])
        for row in rows:
            cells = t.add_row().cells
            for i, val in enumerate(row):
                cells[i].text = str(val)
                cells[i].vertical_alignment = WD_ALIGN_VERTICAL.CENTER
                set_cell_borders(cells[i])
                set_cell_margins(cells[i])
                if len(t.rows) % 2 == 1:
                    set_cell_shading(cells[i], pale_blue)
                if widths:
                    cells[i].width = Inches(widths[i])
        for row in t.rows:
            for cell in row.cells:
                for p in cell.paragraphs:
                    p.paragraph_format.space_after = Pt(0)
                    for run in p.runs:
                        run.font.name = "Aptos"
                        run._element.rPr.rFonts.set(qn("w:ascii"), "Aptos")
                        run._element.rPr.rFonts.set(qn("w:hAnsi"), "Aptos")
                        run.font.size = Pt(9.5)
        doc.add_paragraph()
        return t

    def add_page_number(paragraph):
        paragraph.alignment = WD_ALIGN_PARAGRAPH.RIGHT
        run = paragraph.add_run("CNQ User Manual   |   ")
        run.font.size = Pt(9)
        run.font.color.rgb = RGBColor.from_string("666666")
        begin = OxmlElement("w:fldChar")
        begin.set(qn("w:fldCharType"), "begin")
        instr = OxmlElement("w:instrText")
        instr.set(qn("xml:space"), "preserve")
        instr.text = " PAGE "
        end = OxmlElement("w:fldChar")
        end.set(qn("w:fldCharType"), "end")
        run._r.extend([begin, instr, end])

    add_page_number(section.footer.paragraphs[0])

    # ---- title ------------------------------------------------------------ #
    title = doc.add_paragraph(style="Title")
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    title.add_run("CNQ User Manual")
    sub = para("Local browser application for censored noncrossing quantile regression",
               align=WD_ALIGN_PARAGRAPH.CENTER)
    sub.runs[0].font.size = Pt(14)
    sub.runs[0].font.bold = True
    para("Use this manual to install and start CNQ, upload a CSV file, train a model, "
         "and produce predictions or population projections. A simulated example is "
         "available if you want to try the workflow without using your own data.",
         align=WD_ALIGN_PARAGRAPH.CENTER)
    para("CNQ runs on your computer and opens in a web browser. After the app starts, "
         "you do not need to write Python code.", align=WD_ALIGN_PARAGRAPH.CENTER)
    doc.add_page_break()

    # ---- 1. before you begin --------------------------------------------- #
    h1("1 Before you begin")
    para("CNQ is a local browser application. The browser is the user interface, while "
         "a small Python program runs in a terminal window on the same computer. Keep "
         "that terminal window open for the entire session.")
    labeled_para("For company data. ",
                 "Follow your organization's rules for approved software, protected "
                 "data, storage locations, and model validation. If you cannot install "
                 "Python packages on your computer, ask IT to package or install the app.")

    h2("1 1 What you need")
    table(["Item", "Requirement"], [
        ["Operating system", "Windows 10 or 11, a current version of macOS, or Linux."],
        ["Python", "Python 3.11 is recommended. Versions 3.9 through 3.13 are supported by the launcher."],
        ["Web browser", "A current version of Microsoft Edge, Google Chrome, Firefox, or Safari."],
        ["Internet access", "Required during the first setup so Python packages can be downloaded."],
        ["Disk location", "Use a local folder. Avoid OneDrive, iCloud, and Dropbox because synchronization can lock setup files."],
        ["Git", "Needed only if you download the project with the Git commands in this manual."],
    ], widths=[1.45, 5.15])

    h2("1 2 Choose how to obtain CNQ")
    table(["Method", "Best for", "Important detail"], [
        ["Approved deepquant zip file", "Most analysts", "The package includes both cnq_app and deepcnq. Use the zip supplied by your project team or IT."],
        ["GitHub clone", "Developers and analysts who use Git", "Clone with --recursive so the deepcnq submodule is included."],
    ], widths=[1.75, 1.55, 3.3])
    labeled_para("Do not use GitHub Download ZIP for the source repository. ",
                 "That download may omit the deepcnq submodule. Use the approved "
                 "deepquant zip file or the recursive Git command below.")

    # ---- 2. first start --------------------------------------------------- #
    h1("2 Start CNQ for the first time")
    para("The same procedure applies on Windows, macOS, and Linux. The first start "
         "creates a private Python environment inside cnq_app and installs the required "
         "packages. This can take several minutes.")

    h2("2 1 Obtain the project")
    para("If you received the approved deepquant zip file, extract it to a local folder "
         "that is not synchronized by OneDrive, iCloud, or Dropbox. If you are cloning "
         "from GitHub, open a terminal in the folder where you want to keep the project and run:")
    code("git clone --recursive https://github.com/IsaacPan1/Deepcnq_app")

    h2("2 2 Open a terminal in cnq_app")
    para("Locate the cnq_app folder inside the extracted or cloned project. Open "
         "PowerShell, Command Prompt, or Terminal. Type cd followed by a space, drag "
         "the cnq_app folder into the terminal window, and press Enter.")

    h2("2 3 Start the app")
    para("Run the following command:", keep=True)
    code("python run.py")
    labeled_para("If the command is not recognized. ",
                 "Run python3 run.py instead. Use whichever command works on your "
                 "computer for all later instructions.")
    para("Wait for the following line. Keep the terminal window open while you use CNQ.")
    code("Serving CNQ app on 127.0.0.1:8000 (http://127.0.0.1:8000/)")
    steps([
        "Open your web browser.",
        "Type http://127.0.0.1:8000/ in the address bar and press Enter.",
        "On the Train a model tab, choose your CSV file, confirm the column mapping, run Auto-tune, apply the best settings, and select Start training.",
    ])
    labeled_para("What to expect. ",
                 "The first setup prints many installation messages. This is normal. "
                 "Later starts are much faster because the packages are already installed. "
                 "The demo links are optional and can be ignored.")

    h2("2 4 If the browser does not open")
    para("The launcher starts the local service but may not open a browser window. Open "
         "a browser yourself and enter the address exactly as shown:")
    code("http://127.0.0.1:8000/")
    para("This address refers to your own computer. It is not a public website. If you "
         "started the app on another port, replace 8000 with that port number.")
    para("When you are finished, return to the terminal and press Ctrl+C to stop CNQ.")

    # ---- 3. demo ---------------------------------------------------------- #
    h1("3 Optional simulated demo")
    para("You can skip this section and upload your own CSV immediately. The demo does "
         "not open automatically and is not required each time you start CNQ. Use it "
         "only when you want to check the installation or explore the workflow without "
         "using company data. All demo records are simulated.")
    steps([
        "Open the Train a model tab and select Use demo training data.",
        "Choose Advanced settings, keep the suggested values, choose Every 5 percent for the quantile levels, and select Start training.",
        "Open the Predict new subjects tab and choose Demo model simulated data.",
        "If the model is not built, select Build demo model and wait about two minutes.",
        "Under Or use demo new subjects, select 20 new patients and run the prediction.",
        "Review the two out-of-range flags, then try 500 with outcomes and 500 shifted population to see how validation changes when the population changes.",
    ])

    # ---- 4. what it does -------------------------------------------------- #
    h1("4 What CNQ does")
    para("CNQ predicts a distribution of survival time for each subject rather than a "
         "single time. It estimates several percentiles, called quantiles, which form a "
         "median prediction and prediction intervals. The quantiles are constrained so "
         "higher percentiles cannot fall below lower percentiles.")
    bullets([
        "Train a model on censored survival data or use Auto-tune to compare settings.",
        "Save a trained model and predict survival-time quantiles for new subjects.",
        "Project population trends, including expected events by a time or time to a target number of events.",
    ])

    # ---- 5. concepts ------------------------------------------------------ #
    h1("5 Key concepts")
    table(["Term", "Meaning"], [
        ["Follow-up time", "Time from the study start until the event or the subject's last contact. Values must be positive."],
        ["Event indicator", "1 means the event occurred. 0 means the observation was censored at the last contact."],
        ["Censoring", "The event was not observed before follow-up ended. CNQ accounts for censoring when fitting and evaluating models."],
        ["Quantile", "A percentile of predicted survival time. The 0.5 quantile is the median predicted time."],
        ["Noncrossing", "Higher quantiles are constrained to be at least as large as lower quantiles."],
        ["Survival curve", "The estimated probability of remaining event-free beyond a specified time."],
    ], widths=[1.55, 5.05])

    # ---- 6. data ---------------------------------------------------------- #
    h1("6 Prepare your data")
    para("Use one CSV file with one header row and one row per subject.")
    table(["Column", "Requirement", "Example"], [
        ["Follow-up time", "Positive numeric value in the same unit for every row", "12.4"],
        ["Event", "1 for event and 0 for censored, or two values that you map in the app", "1"],
        ["Covariates", "One or more numeric predictors. Represent categories with numeric indicator columns.", "age 58"],
        ["Subject ID", "Optional unique label. It is not used as a predictor.", "S001"],
    ], widths=[1.3, 4.2, 1.1])
    para("Example CSV content:")
    code("subject_id,time,event,age,biomarker,treatment\r\n"
         "S001,12.4,1,58,2.7,1\r\n"
         "S002,36.0,0,64,1.2,0")
    bullets([
        "Use no blank cells in the columns selected for analysis.",
        "Keep one row per subject and use unique subject IDs if an ID is included.",
        "Use positive times and a consistent unit such as days or months.",
        "Aim for at least about 100 events and enough observations that each data split contains at least 10 events.",
    ])
    para("The app checks the selected columns before training. Errors block training. "
         "Warnings require review and acknowledgement. Rows with invalid values in "
         "selected columns may be excluded and are reported in the validation summary.")

    # ---- 7. train --------------------------------------------------------- #
    h1("7 Train a model")
    h2("7 1 Upload")
    para("On the Train a model tab, choose a CSV file or drag it into the upload area. "
         "The app displays a preview before any model is trained.")
    h2("7 2 Map columns")
    para("Select the duration and event columns. Optionally select a subject ID and type "
         "the time unit used in the file. Choose the numeric covariates to include. If "
         "the event column does not use 0 and 1, tell the app which value represents an event.")
    h2("7 3 Use Auto-tune — the default")
    para("Auto-tune is selected automatically. It compares KAN-CNQ and MLP-CNQ settings "
         "on your data and ranks them by validation IPCW pinball loss. Lower validation "
         "pinball loss is better. For most analyses, use the Standard search effort.")
    steps([
        "Select the KAN-CNQ and MLP-CNQ model families to compare.",
        "Choose Quick, Standard, or Thorough search effort.",
        "Select Find best settings and wait for the leaderboard.",
        "Select Use best settings. CNQ applies the winning model and values.",
        "Continue to Start training.",
    ])
    labeled_para("Advanced settings. ",
                 "Choose this only when your analysis plan requires specific model or "
                 "hyperparameter values. It reveals the model, network, learning-rate, "
                 "batch-size, epoch, and early-stopping controls. Record any departures "
                 "from the approved analysis plan.")
    h2("7 4 Choose a model family")
    table(["Model", "When to use it"], [
        ["KAN-CNQ", "Use for ordinary numeric tabular covariates. It is included in Auto-tune."],
        ["MLP-CNQ", "Use for ordinary numeric tabular covariates as a strong baseline. It is included in Auto-tune."],
        ["Trans-CNQ or TransKAN-CNQ", "Consider only when the numeric covariates include information derived from text or structured annotations and attention is scientifically justified. Select these under Advanced settings."],
    ], widths=[2.0, 4.6])
    labeled_para("Raw text is not accepted. ",
                 "Convert notes, labels, or annotations into approved numeric covariates "
                 "before creating the CSV. For purely tabular data, use KAN-CNQ or "
                 "MLP-CNQ; this is why Auto-tune searches only those two families.")
    h2("7 5 Quantile levels")
    para("The default levels 0.1, 0.25, 0.5, 0.75, and 0.9 are a practical starting "
         "point. Every 10 percent, Every 5 percent, and Every 1 percent produce denser "
         "curves but require more computation. Extreme levels below 0.05 or above 0.95 "
         "depend on relatively few events and may be unstable.")
    h2("7 6 Data splits and reproducibility")
    para("Set the train, validation, and test proportions; the number of repeated splits; "
         "and the random seed. More repeated splits provide variation estimates but take "
         "longer. Deterministic mode is slower and is intended for exactly reproducible runs.")
    h2("7 7 Read the results")
    table(["Metric", "Interpretation"], [
        ["IPCW pinball", "Overall quantile prediction loss. Lower is better."],
        ["ICP 80 percent and 50 percent", "Observed interval coverage. Values closer to the stated interval level are better."],
        ["Uno C index", "Ability to rank subjects by risk. 0.5 is chance and 1.0 is perfect ranking."],
        ["Calibration", "Agreement between predicted and observed coverage across quantiles."],
        ["Crossing rate", "Expected to be 0 percent for the noncrossing models."],
    ], widths=[2.05, 4.55])
    para("The downloadable report also includes outcome, training, calibration, profile, "
         "coverage, feature importance, and repeated-split plots when those outputs are available.")

    # ---- 8. save ---------------------------------------------------------- #
    h1("8 Save a model")
    para("After training, use Save a model to create a .cnqmodel file for later use in "
         "the Predict new subjects and Project trends tabs.")
    table(["Save option", "What it contains"], [
        ["Final model refit on all data", "Retrains the selected model on all rows while holding out 15 percent for early stopping. This is usually the model to retain for reuse."],
        ["Ensemble of repeated split models", "Keeps the models from all repeated splits and averages their predictions."],
        ["Single split", "Keeps one trained split model."],
    ], widths=[2.25, 4.35])
    para("The app keeps only the most recent training runs on disk, with a default limit "
         "of 20. Save and archive each model that must be retained. A downloaded "
         ".cnqmodel file is self-contained and is not removed by run cleanup.")

    # ---- 9. predict ------------------------------------------------------- #
    h1("9 Predict new subjects")
    steps([
        "On the Predict new subjects tab, choose a saved model or upload a .cnqmodel file.",
        "Upload a CSV containing the new subjects.",
        "Review the column mapping. The app matches names automatically and shows how many model features were matched.",
        "If names differ, download the model template, match by position after checking the proposed pairs, or select columns manually.",
        "Optionally choose an ID column. Add time and event columns only when outcomes are known and the data are suitable for external validation.",
        "Select Run prediction and download the predictions CSV, HTML report, or results zip.",
    ])
    para("Each output row includes the requested quantiles, median, 80 percent interval, "
         "interval width, and an out_of_range flag. A value of 1 in out_of_range means "
         "at least one feature lies outside its training range.")
    labeled_para("External validation. ",
                 "Outcome-based metrics are meaningful only for subjects that were not "
                 "used to train the model. Do not mix training records into the validation file.")

    # ---- 10. project ------------------------------------------------------ #
    h1("10 Project population trends")
    para("The Project trends tab answers aggregate questions such as expected cumulative "
         "events by a specified time or the expected time to a target number of events.")
    table(["Projection basis", "Use"], [
        ["Population size N", "Scales the training-population Kaplan Meier curve to N subjects. No cohort file is required."],
        ["Uploaded cohort", "Uses the covariates in a new cohort after the same feature-mapping checks used for prediction."],
    ], widths=[2.0, 4.6])
    para("For a quick check of cohort mode, choose Upload a cohort and select Use "
         "simulated sample cohort. CNQ creates and uploads 20 simulated rows whose "
         "columns match the selected model. You can replace them with your own CSV at any time.")
    para("The app reports a cumulative-event curve, uncertainty band, answers to the "
         "requested questions, and downloadable files. Projections stop at the last "
         "observed event time in the training data. Targets beyond that follow-up "
         "horizon are flagged and are not extrapolated.")

    # ---- 11. limits ------------------------------------------------------- #
    h1("11 Interpret results and limitations")
    bullets([
        "A subject's quantile interval describes variation in outcomes among similar subjects. It is not a confidence interval for the fitted model parameters.",
        "Predictions assume that new subjects are sufficiently similar to the training population. A different site, era, protocol, or inclusion criterion can reduce calibration.",
        "Out-of-range predictions are extrapolations and should receive additional review.",
        "External validation requires unseen data that were not used for training or model selection.",
        "Projection uncertainty bands describe aggregate outcome or sampling variation. They do not quantify all model uncertainty.",
        "Population projections are supported only through the observed training follow-up horizon.",
    ])

    # ---- 12. troubleshooting --------------------------------------------- #
    h1("12 Troubleshooting")
    table(["Problem", "What to do"], [
        ["Terminal cannot find Python", "Install Python 3.11, close the terminal, and open a new terminal. Try python run.py, then python3 run.py if needed."],
        ["Terminal cannot find Git", "Install Git, close the terminal, and open a new terminal. Git is not required for an approved zip release."],
        ["Destination path already exists", "The project was already cloned. Use the existing folder or update it with Git instead of cloning again."],
        ["Repository not found", "For a Git copy, run git submodule update --init --recursive from the Deepcnq_app folder. For a release, confirm that deepcnq and cnq_app are sibling folders."],
        ["Port 8000 is already in use", "Run python run.py 8001 and open http://127.0.0.1:8001/. Replace python with python3 if that is the command you normally use."],
        ["Red package banner or installation failure", "Check the internet connection. From cnq_app, run python run.py --reinstall. Replace python with python3 if needed."],
        ["No Python at an old location", "Stop CNQ, delete the .venv folder inside cnq_app, and start the app again so the environment is rebuilt."],
        ["Page is unformatted or buttons do not respond", "Use the exact local address and perform a hard refresh with Ctrl+Shift+R on Windows or Cmd+Shift+R on macOS."],
        ["Setup fails in a synchronized folder", "Move the full deepquant or Deepcnq_app folder to a local nonsynchronized location and start again."],
        ["Demo model is not built", "On the Predict new subjects tab, select Build demo model and wait for it to complete."],
        ["Population projection is unavailable", "The saved model does not contain a training curve. Save or rebuild the model with a current version of CNQ. The simulated demo model is upgraded automatically."],
    ], widths=[2.3, 4.3])

    # ---- 13. update ------------------------------------------------------- #
    h1("13 Update a GitHub copy")
    para("This section applies only if you cloned the project with Git. Stop CNQ. From "
         "the cnq_app folder, run:")
    code("cd ..\r\n"
         "git pull --recurse-submodules")
    para("If your organization distributes approved zip releases, use its approved "
         "update procedure instead of replacing files yourself.")

    # ---- 14. glossary ----------------------------------------------------- #
    h1("14 Glossary")
    table(["Term", "Meaning"], [
        ["IPCW", "Inverse probability of censoring weighting, used to account for censored observations in evaluation."],
        ["Pinball loss", "Quantile regression loss. Lower values indicate smaller quantile prediction errors."],
        ["Kaplan Meier estimate", "Nonparametric estimate of the survival curve from observed times and event indicators."],
        ["Greenwood variance", "Variance formula used for the population survival estimate and projection band."],
        ["Uno C index", "Censoring-adjusted measure of how well the model ranks subjects by risk."],
        ["ICP", "Interval coverage probability, the fraction of observed outcomes contained in a predicted interval."],
        ["Horizon", "The last observed event time through which a population projection is supported."],
        ["cnqmodel file", "Self-contained saved model bundle with weights and metadata."],
    ], widths=[1.7, 4.9])

    doc.save(str(path))
    return path


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Build the CNQ user manual (MANUAL.docx).")
    parser.add_argument("-o", "--out", default=str(OUT), help="output .docx path")
    args = parser.parse_args(argv)
    out = build(Path(args.out))
    print(f"Wrote {out} ({out.stat().st_size // 1024} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
