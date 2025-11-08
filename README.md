# my-codex-project

A tiny Python template that demonstrates how to structure a small command line
utility.  The project ships with a ``codex_project`` package that exposes a
simple greeting helper and includes automated tests so you can immediately run
``pytest`` to verify everything works.

## Getting started

1. Clone the repository locally:

   ```bash
   git clone https://github.com/YOUR_GITHUB_USERNAME/my-codex-project.git
   cd my-codex-project
   ```

   Replace ``YOUR_GITHUB_USERNAME`` with your actual GitHub handle.

2. (Optional) create and activate a virtual environment to isolate dependencies.

3. Run the command line interface:

   ```bash
   python -m codex_project.cli "Ada"
   ```

   The script prints ``Hello, Ada! Welcome to my-codex-project.``.  You can omit
   the name argument to fall back to a generic greeting.

## Testing

Execute the test suite with:

```bash
python -m pytest -q
```

Using ``python -m pytest`` ensures the current working directory is on
``PYTHONPATH`` so that ``codex_project`` can be imported without installing the
package first.
