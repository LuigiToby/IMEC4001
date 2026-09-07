from pathlib import Path
import shutil
import html
import re
import subprocess


ROOT = Path(__file__).resolve().parent.parent
SOLUCIONES = ROOT / "soluciones"
SITE = ROOT / "site"


def get_title(notebook):
    """Obtiene el primer título Markdown del notebook."""
    import json

    with open(notebook, "r", encoding="utf-8") as f:
        data = json.load(f)

    for cell in data.get("cells", []):
        if cell.get("cell_type") == "markdown":
            text = "".join(cell.get("source", []))
            match = re.search(r"^\s*#\s+(.+)$", text, re.MULTILINE)
            if match:
                return match.group(1).strip()

    return notebook.stem.replace("_", " ").title()


def convert_notebook(notebook, output):
    subprocess.run(
        [
            "jupyter",
            "nbconvert",
            "--to",
            "html",
            "--output-dir",
            str(output),
            str(notebook),
        ],
        check=True,
    )


def make_index(notebooks):
    cards = []

    for i, notebook in enumerate(notebooks, start=1):
        filename = notebook.stem + ".html"
        title = get_title(notebook)

        cards.append(
            f"""
            <a class="solution" href="{html.escape(filename)}">
                <div class="number">{i:02d}</div>
                <div>
                    <h2>{html.escape(title)}</h2>
                    <p>{html.escape(notebook.stem)}</p>
                </div>
                <div class="arrow">→</div>
            </a>
            """
        )

    content = "\n".join(cards)

    return f"""<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>IMEC 4001 — Soluciones</title>

    <style>
        * {{
            box-sizing: border-box;
        }}

        body {{
            margin: 0;
            background: #ffffff;
            color: #111111;
            font-family:
                -apple-system,
                BlinkMacSystemFont,
                "Segoe UI",
                Helvetica,
                Arial,
                sans-serif;
        }}

        main {{
            max-width: 900px;
            margin: 0 auto;
            padding: 80px 30px;
        }}

        header {{
            margin-bottom: 70px;
        }}

        .eyebrow {{
            font-size: 13px;
            letter-spacing: 0.12em;
            text-transform: uppercase;
            color: #777;
            margin-bottom: 15px;
        }}

        h1 {{
            margin: 0;
            font-size: clamp(42px, 8vw, 72px);
            line-height: 0.95;
            letter-spacing: -0.05em;
            font-weight: 700;
        }}

        .subtitle {{
            margin-top: 20px;
            color: #666;
            font-size: 17px;
        }}

        .solutions {{
            border-top: 1px solid #ddd;
        }}

        .solution {{
            display: grid;
            grid-template-columns: 60px 1fr 30px;
            align-items: center;
            gap: 20px;
            padding: 25px 0;
            border-bottom: 1px solid #ddd;
            text-decoration: none;
            color: inherit;
            transition: padding 0.15s ease;
        }}

        .solution:hover {{
            padding-left: 10px;
        }}

        .number {{
            color: #999;
            font-family: monospace;
            font-size: 14px;
        }}

        .solution h2 {{
            margin: 0 0 5px;
            font-size: 21px;
            font-weight: 600;
        }}

        .solution p {{
            margin: 0;
            color: #888;
            font-family: monospace;
            font-size: 13px;
        }}

        .arrow {{
            font-size: 22px;
        }}

        footer {{
            margin-top: 70px;
            padding-top: 20px;
            border-top: 1px solid #ddd;
            color: #888;
            font-size: 13px;
        }}

        @media (max-width: 600px) {{
            main {{
                padding: 50px 20px;
            }}

            .solution {{
                grid-template-columns: 40px 1fr 20px;
                gap: 10px;
            }}
        }}
    </style>
</head>

<body>
<main>

<header>
    <div class="eyebrow">Universidad de los Andes · 2026-2</div>
    <h1>IMEC 4001</h1>
    <div class="subtitle">Matemáticas Aplicadas — Soluciones</div>
</header>

<section class="solutions">
    {content}
</section>

<footer>
    Luis Alejandro Rodríguez · IMEC 4001
</footer>

</main>
</body>
</html>
"""


def main():
    if SITE.exists():
        shutil.rmtree(SITE)

    SITE.mkdir(parents=True)

    notebooks = sorted(SOLUCIONES.glob("*.ipynb"))

    if not notebooks:
        print("No se encontraron notebooks en soluciones/")
        return

    print(f"Encontrados {len(notebooks)} notebooks.")

    for notebook in notebooks:
        print(f"Convirtiendo: {notebook.name}")
        convert_notebook(notebook, SITE)

    index = make_index(notebooks)

    (SITE / "index.html").write_text(
        index,
        encoding="utf-8",
    )

    print(f"Sitio generado en: {SITE}")


if __name__ == "__main__":
    main()