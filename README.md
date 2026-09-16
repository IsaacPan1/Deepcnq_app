# deepquant

Two sibling folders:

- **`deepcnq/`** — the unmodified upstream repository
  ([BIG-S2/deepcnq](https://github.com/BIG-S2/deepcnq)) with the `deepquantreg`
  package and the model configs. Treated as read-only; nothing here edits it.
- **`cnq_app/`** — a local, single-user web app that wraps `deepquantreg` so you
  can train censored non-crossing quantile regression models from your own CSV,
  in the browser.

## Get started

See **[cnq_app/README.md](cnq_app/README.md)** for the copy-paste quick start
(install Python, unzip, `python run.py`, open the printed URL). Developer notes
are in [cnq_app/DEVELOPER.md](cnq_app/DEVELOPER.md).

## Update the upstream repo

```bash
cd deepcnq && git pull
```

or, to update and re-run the app's smoke test in one step:

```bash
cd cnq_app && python update_repo.py
```
