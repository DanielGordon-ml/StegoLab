"""Extract a static README image from the exact, checked Archify HTML artifact."""

import hashlib
import json
import re
import xml.etree.ElementTree as element_tree
from pathlib import Path


def required_match(pattern: str, content: str) -> re.Match[str]:
    """Fail clearly when an Archify template change needs an exporter review."""
    match = re.search(pattern, content, re.DOTALL)
    if match is None:
        raise ValueError("The Archify template changed; review the SVG exporter.")
    return match


def export_svg(directory: Path) -> Path:
    """Copy authored geometry and light-theme styles without running viewer code."""
    artifact = (directory / "architecture.html").read_bytes()
    receipt = json.loads((directory / "architecture.delivery.json").read_text())
    if (
        not receipt["ok"]
        or receipt["artifact"]["sha256"] != hashlib.sha256(artifact).hexdigest()
    ):
        raise ValueError("Deliver and validate the current Archify HTML first.")
    specification = (directory / "architecture.architecture.json").read_bytes()
    if receipt["specification"]["sha256"] != hashlib.sha256(specification).hexdigest():
        raise ValueError("The specification changed; deliver the diagram again.")
    content = artifact.decode("utf-8")
    drawing = required_match(r"(<svg\b.*?</svg>)", content).group(1)
    root = element_tree.fromstring(drawing)
    theme = required_match(r'\[data-theme="light"\]\s*\{(.*?)\}', content).group(1)
    variables = dict(re.findall(r"(--[\w-]+):\s*([^;]+);", theme))
    fonts = required_match(r'<style id="archify-fonts">(.*?)</style>', content).group(1)
    # Only static semantic rules enter the image; no viewer controls or scripts.
    semantic = required_match(
        r"SVG SEMANTIC CLASSES(.*?)Stable semantic exploration hooks", content
    ).group(1)
    rules = re.findall(r"\.[ctam]-[\w-]+\s*\{[^{}]+\}", semantic)
    rules += re.findall(r"svg \.s-[\w-]+\s*\{[^{}]+\}", semantic)
    rules += re.findall(r"svg \.semantic-sigil[^{}]*\{[^{}]+\}", semantic)
    stylesheet = "\n".join(dict.fromkeys(rules))
    stylesheet = re.sub(
        r"var\((--[\w-]+)\)", lambda match: variables[match.group(1)], stylesheet
    )
    stylesheet += "\ntext { font-family: 'JetBrains Mono', monospace; }\n"
    root.set("xmlns", "http://www.w3.org/2000/svg")
    _, _, width, height = root.attrib["viewBox"].split()
    root.set("width", width)
    root.set("height", height)
    for node in root.iter():
        node.attrib.pop("tabindex", None)
        node.attrib.pop("aria-pressed", None)
        if node.attrib.get("role") == "button":
            node.attrib.pop("role")
    style = element_tree.Element("style")
    style.text = fonts + "\n" + stylesheet
    root.insert(0, style)
    background = element_tree.Element(
        "rect", {"width": "100%", "height": "100%", "fill": "#ffffff"}
    )
    root.insert(1, background)
    output = directory / "architecture.svg"
    element_tree.ElementTree(root).write(output, encoding="utf-8", xml_declaration=True)
    return output


if __name__ == "__main__":
    print(export_svg(Path(__file__).resolve().parents[1] / "docs"))
