"""Remove Word comments and their package relationships from a DOCX."""

from __future__ import annotations

import argparse
from pathlib import Path
from tempfile import NamedTemporaryFile
from zipfile import ZIP_DEFLATED, ZipFile
from xml.etree import ElementTree as ET


W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
REL = "http://schemas.openxmlformats.org/package/2006/relationships"
CT = "http://schemas.openxmlformats.org/package/2006/content-types"


def strip_comments(path: Path) -> None:
    with NamedTemporaryFile(suffix=".docx", delete=False, dir=path.parent) as temp:
        temporary = Path(temp.name)
    try:
        with ZipFile(path) as source, ZipFile(temporary, "w", ZIP_DEFLATED) as target:
            for info in source.infolist():
                name = info.filename
                if name in {"word/comments.xml", "word/persons.xml"}:
                    continue
                data = source.read(name)
                if name == "word/document.xml":
                    root = ET.fromstring(data)
                    for element in list(root.iter()):
                        if element.tag in {
                            f"{{{W}}}commentRangeStart",
                            f"{{{W}}}commentRangeEnd",
                            f"{{{W}}}commentReference",
                        }:
                            parent = next((p for p in root.iter() if element in list(p)), None)
                            if parent is not None:
                                parent.remove(element)
                    data = ET.tostring(root, encoding="utf-8", xml_declaration=True)
                elif name == "word/_rels/document.xml.rels":
                    root = ET.fromstring(data)
                    for element in list(root):
                        if element.get("Type", "").endswith("/comments") or element.get("Target") in {"comments.xml", "persons.xml"}:
                            root.remove(element)
                    data = ET.tostring(root, encoding="utf-8", xml_declaration=True)
                elif name == "[Content_Types].xml":
                    root = ET.fromstring(data)
                    for element in list(root):
                        if element.get("PartName") in {"/word/comments.xml", "/word/persons.xml"}:
                            root.remove(element)
                    data = ET.tostring(root, encoding="utf-8", xml_declaration=True)
                target.writestr(info, data)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("docx", type=Path)
    strip_comments(parser.parse_args().docx)
    print("Removed Word comments and package relationships")
