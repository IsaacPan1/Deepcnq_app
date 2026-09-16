# CNQ App

A browser app for **censored non-crossing quantile regression**: predict the full survival-time distribution for each subject.
It opens in your web browser, but **everything runs on your own computer**. Your data is never uploaded anywhere.

First-time setup takes about 10 minutes.

---

## 1. Install Python and Git (one time only)

**Python 3.11:** https://www.python.org/downloads/release/python-3119/
(scroll down to *Files* and pick the Windows or macOS installer)

> **Windows:** on the first installer screen, tick **"Add python.exe to PATH"**.

**Git**
- **Windows:** install from https://git-scm.com/download/win (the default options are fine)
- **Mac:** Git is installed automatically the first time you run `git`. If a window asks to install "command line developer tools", click **Install**.

## 2. Download the app

Open a terminal:

- **Windows:** Start menu → type `cmd` → press Enter
- **Mac:** Cmd+Space → type `Terminal` → press Enter

Copy and paste:

**Windows**
```
cd /d %USERPROFILE%
git clone --recursive https://github.com/IsaacPan1/Deepcnq_app
```

**Mac**
```
cd ~
git clone --recursive https://github.com/IsaacPan1/Deepcnq_app
```

This creates a `Deepcnq_app` folder in your home folder.

## 3. Start the app

**Windows**
```
cd /d %USERPROFILE%\Deepcnq_app\cnq_app
python run.py
```

**Mac**
```
cd ~/Deepcnq_app/cnq_app
python3 run.py
```

The first time, it downloads what it needs, which takes a few minutes. Wait until you see:

```
Serving CNQ app on 127.0.0.1 port 8000
```

## 4. Open the app

Open this address in your browser:

### http://127.0.0.1:8000/

- Keep the terminal window **open** while you use the app.
- To stop the app, click the terminal and press **Ctrl+C**.

---

## Open it again later

Open a terminal and paste:

**Windows**
```
cd /d %USERPROFILE%\Deepcnq_app\cnq_app
python run.py
```

**Mac**
```
cd ~/Deepcnq_app/cnq_app
python3 run.py
```

Then open http://127.0.0.1:8000/

## Update to the latest version

**Windows**
```
cd /d %USERPROFILE%\Deepcnq_app
git pull --recurse-submodules
```

**Mac**
```
cd ~/Deepcnq_app
git pull --recurse-submodules
```

Then start the app as usual. If new packages are needed, they install automatically.

---

## Preparing your data

Upload **one CSV file** with **one row per subject**:

| Column | Format |
|---|---|
| Follow-up time | Number greater than 0, same unit for every row (days, months...) |
| Event | `1` = event happened, `0` = censored |
| Covariates | Numbers only; categories as 0/1 columns |
| Subject ID *(optional)* | Any unique value |

No blank cells in the columns you use. A sample dataset and a template are available in the app.
More details are in [`cnq_app/README.md`](cnq_app/README.md).

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `git` is not recognized (Windows) | Install Git from https://git-scm.com/download/win, then open a **new** terminal. |
| `python` is not recognized (Windows) | Reinstall Python and tick **"Add python.exe to PATH"**, then open a **new** terminal. |
| "destination path 'Deepcnq_app' already exists" | You already downloaded it. Skip to step 3, or see *Update to the latest version*. |
| "Repository not found" when starting the app | Run `git submodule update --init --recursive` inside the `Deepcnq_app` folder, then start again. |
| "Port 8000 is already in use" | Run `python run.py 8001` (Mac: `python3 run.py 8001`) and open http://127.0.0.1:8001/ |
| Page looks plain or buttons don't work | Use exactly `http://127.0.0.1:8000/`, then press **Ctrl+Shift+R** (Mac: **Cmd+Shift+R**). |
| Red banner about missing packages, or installation failed | Check your internet connection, then run `python run.py --reinstall` (Mac: `python3 run.py --reinstall`). |

Still stuck? [Open an issue](https://github.com/IsaacPan1/Deepcnq_app/issues) or contact **Isaac, your-email@unc.edu**.

---

## Citation

Huang S, Qu Z, Hua Z, Shen G, Tang R, Zhu H. *Non-crossing deep quantile regression for distributional survival prediction.* arXiv:2608.16864 (2026).
Method code: https://github.com/BIG-S2/deepcnq