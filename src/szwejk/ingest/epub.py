"""Low-level EPUB parsing utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import PurePosixPath
from zipfile import ZipFile
import xml.etree.ElementTree as ET


OPF_NS = {"opf": "http://www.idpf.org/2007/opf", "dc": "http://purl.org/dc/elements/1.1/"}
CONTAINER_NS = {"container": "urn:oasis:names:tc:opendocument:xmlns:container"}
XHTML_NS = {"xhtml": "http://www.w3.org/1999/xhtml"}


@dataclass(slots=True)
class SpineDocument:
    idref: str
    href: str
    media_type: str
    title: str
    body_xml: str


@dataclass(slots=True)
class EpubPackage:
    epub_path: str
    opf_path: str
    identifier: str | None
    language: str
    title: str
    spine_documents: list[SpineDocument]


def load_epub_package(epub_path: str) -> EpubPackage:
    with ZipFile(epub_path) as archive:
        opf_path = _read_rootfile_path(archive)
        opf_xml = archive.read(opf_path)
        opf_root = ET.fromstring(opf_xml)
        manifest = {
            item.attrib["id"]: (item.attrib["href"], item.attrib.get("media-type", ""))
            for item in opf_root.findall("./opf:manifest/opf:item", OPF_NS)
            if "id" in item.attrib and "href" in item.attrib
        }
        identifier = _find_text(opf_root, "./opf:metadata/dc:identifier")
        language = _find_text(opf_root, "./opf:metadata/dc:language") or "und"
        title = _find_text(opf_root, "./opf:metadata/dc:title") or PurePosixPath(epub_path).stem
        opf_dir = PurePosixPath(opf_path).parent

        spine_documents: list[SpineDocument] = []
        for itemref in opf_root.findall("./opf:spine/opf:itemref", OPF_NS):
            idref = itemref.attrib.get("idref")
            if not idref or idref not in manifest:
                continue
            href, media_type = manifest[idref]
            if media_type != "application/xhtml+xml":
                continue
            normalized_path = str(opf_dir / href)
            if _should_skip_document(href):
                continue
            body_xml, title_text = _read_xhtml_document(archive, normalized_path)
            spine_documents.append(
                SpineDocument(
                    idref=idref,
                    href=normalized_path,
                    media_type=media_type,
                    title=title_text or PurePosixPath(href).stem,
                    body_xml=body_xml,
                )
            )

    return EpubPackage(
        epub_path=epub_path,
        opf_path=opf_path,
        identifier=identifier,
        language=language,
        title=title,
        spine_documents=spine_documents,
    )


def parse_xhtml_document(body_xml: str) -> ET.Element:
    return ET.fromstring(body_xml)


def _read_rootfile_path(archive: ZipFile) -> str:
    container_root = ET.fromstring(archive.read("META-INF/container.xml"))
    rootfile = container_root.find("./container:rootfiles/container:rootfile", CONTAINER_NS)
    if rootfile is None or "full-path" not in rootfile.attrib:
        raise ValueError("EPUB container.xml missing rootfile full-path")
    return rootfile.attrib["full-path"]


def _find_text(root: ET.Element, path: str) -> str | None:
    node = root.find(path, OPF_NS)
    if node is None or node.text is None:
        return None
    return node.text.strip() or None


def _read_xhtml_document(archive: ZipFile, path_in_zip: str) -> tuple[str, str | None]:
    raw = archive.read(path_in_zip)
    root = ET.fromstring(raw)
    title_node = root.find("./xhtml:head/xhtml:title", XHTML_NS)
    body_node = root.find("./xhtml:body", XHTML_NS)
    if body_node is None:
        raise ValueError(f"XHTML body missing in {path_in_zip}")
    return ET.tostring(body_node, encoding="unicode"), title_node.text.strip() if title_node is not None and title_node.text else None


def _should_skip_document(href: str) -> bool:
    lowered = href.lower()
    return lowered in {"title.xhtml", "about.xhtml", "nav.xhtml"} or lowered.endswith("/title.xhtml") or lowered.endswith("/about.xhtml")
