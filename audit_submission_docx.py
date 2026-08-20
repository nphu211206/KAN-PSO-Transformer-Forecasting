"""Audit the deterministic structure of the JEAS submission DOCX."""

from __future__ import annotations

import argparse
from pathlib import Path
from xml.etree import ElementTree as ET
from zipfile import ZipFile


NS = {
    "cp": "http://schemas.openxmlformats.org/package/2006/metadata/core-properties",
    "dc": "http://purl.org/dc/elements/1.1/",
    "dcterms": "http://purl.org/dc/terms/",
    "w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main",
    "wp": "http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing",
}


def attr(element, namespace: str, name: str) -> str | None:
    return element.get(f"{{{NS[namespace]}}}{name}")


def require(condition: bool, message: str, failures: list[str]) -> None:
    if not condition:
        failures.append(message)


def audit(path: Path) -> None:
    failures: list[str] = []
    with ZipFile(path) as archive:
        names = set(archive.namelist())
        document = ET.fromstring(archive.read("word/document.xml"))
        styles = ET.fromstring(archive.read("word/styles.xml"))
        core = ET.fromstring(archive.read("docProps/core.xml"))
        story_parts = [
            ET.fromstring(archive.read(name))
            for name in names
            if name == "word/document.xml"
            or (name.startswith("word/header") and name.endswith(".xml"))
            or (name.startswith("word/footer") and name.endswith(".xml"))
        ]

    tables = document.findall(".//w:tbl", NS)
    drawings = document.findall(".//w:drawing", NS)
    descriptions = [
        element.get("descr", "").strip()
        for element in document.findall(".//wp:docPr", NS)
        if element.get("descr", "").strip()
    ]
    media = [name for name in names if name.startswith("word/media/")]
    placeholders = [
        element.text
        for element in document.findall(".//w:t", NS)
        if element.text and "[TO BE COMPLETED BY AUTHORS]" in element.text
    ]

    require(len(tables) == 9, f"expected 9 editable tables, found {len(tables)}", failures)
    require(len(drawings) == 12, f"expected 12 drawings, found {len(drawings)}", failures)
    require(len(media) == 12, f"expected 12 embedded media files, found {len(media)}", failures)
    require(
        len(descriptions) == len(drawings),
        f"expected alt text for every drawing, found {len(descriptions)}/{len(drawings)}",
        failures,
    )
    require(not placeholders, f"found {len(placeholders)} unresolved author placeholders", failures)

    table_shading = [element for table in tables for element in table.findall(".//w:shd", NS)]
    require(not table_shading, f"found {len(table_shading)} table shading elements", failures)
    require(
        sum(
            bool(rows and rows[0].find("w:trPr/w:tblHeader", NS) is not None)
            for rows in (table.findall("w:tr", NS) for table in tables)
        )
        == 9,
        "not every table has a repeating header row",
        failures,
    )

    section = document.find(".//w:sectPr", NS)
    require(section is not None, "missing section properties", failures)
    if section is not None:
        page_size = section.find("w:pgSz", NS)
        margins = section.find("w:pgMar", NS)
        line_numbers = section.find("w:lnNumType", NS)
        require(page_size is not None, "missing explicit page size", failures)
        if page_size is not None:
            require(
                attr(page_size, "w", "w") == "12240" and attr(page_size, "w", "h") == "15840",
                "page size is not US Letter portrait",
                failures,
            )
        require(margins is not None, "missing explicit page margins", failures)
        if margins is not None:
            for side in ("top", "right", "bottom", "left"):
                require(attr(margins, "w", side) == "1440", f"{side} margin is not 1 inch", failures)
        require(line_numbers is not None, "missing line numbering", failures)
        if line_numbers is not None:
            require(attr(line_numbers, "w", "countBy") == "1", "line numbering is not every line", failures)

    instructions = " ".join(
        text
        for part in story_parts
        for element in part.findall(".//w:instrText", NS)
        if (text := element.text)
    )
    require("PAGE" in instructions, "missing page-number field", failures)
    require(
        not document.findall(".//w:ins", NS) and not document.findall(".//w:del", NS),
        "tracked changes remain",
        failures,
    )
    require("word/comments.xml" not in names, "comments remain in the submission", failures)

    for element_name, label in (
        ("dc:creator", "creator"),
        ("cp:lastModifiedBy", "last modified by"),
        ("dc:title", "title metadata"),
        ("cp:keywords", "keyword metadata"),
    ):
        element = core.find(element_name, NS)
        value = (element.text or "").strip() if element is not None else ""
        require(not value, f"{label} is not blank", failures)

    normal_styles = [
        style
        for style in styles.findall(".//w:style", NS)
        if attr(style, "w", "styleId") == "Normal"
    ]
    require(len(normal_styles) == 1, "missing unique Normal style", failures)
    if normal_styles:
        normal = normal_styles[0]
        fonts = normal.find("w:rPr/w:rFonts", NS)
        size = normal.find("w:rPr/w:sz", NS)
        spacing = normal.find("w:pPr/w:spacing", NS)
        require(fonts is not None and attr(fonts, "w", "ascii") == "Times New Roman", "Normal font is not Times New Roman", failures)
        require(size is not None and attr(size, "w", "val") == "24", "Normal font is not 12 pt", failures)
        require(spacing is not None and attr(spacing, "w", "line") == "480", "Normal style is not double-spaced", failures)

    if failures:
        raise SystemExit("Submission DOCX audit failed:\n- " + "\n- ".join(failures))
    print(
        "Submission DOCX audit passed: "
        f"{len(tables)} tables, {len(drawings)} described drawings, Letter geometry, "
        "line/page numbering, blank metadata and clean review state."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("docx", type=Path)
    args = parser.parse_args()
    audit(args.docx)


if __name__ == "__main__":
    main()
