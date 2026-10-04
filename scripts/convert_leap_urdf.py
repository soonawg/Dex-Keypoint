"""Convert the bundled LEAP Hand visual URDF to a MuJoCo kinematic model."""

from __future__ import annotations

from collections import defaultdict
import argparse
from pathlib import Path
import xml.etree.ElementTree as ET


ASSET_DIR = Path(__file__).resolve().parents[1] / "dex_teleop" / "assets" / "leap_hand"


def _numbers(value: str | None, default: tuple[float, ...]) -> tuple[float, ...]:
    if value is None:
        return default
    return tuple(float(number) for number in value.split())


def _format(values: tuple[float, ...]) -> str:
    return " ".join(f"{value:.10g}" for value in values)


def convert(urdf_path: Path, output_path: Path) -> None:
    robot = ET.parse(urdf_path).getroot()
    links = {link.attrib["name"]: link for link in robot.findall("link")}
    children: dict[str, list[ET.Element]] = defaultdict(list)
    child_links = set()
    mesh_names = set()

    for joint in robot.findall("joint"):
        if joint.attrib.get("type") != "revolute":
            raise ValueError(f"Only finite revolute joints are supported: {joint.attrib}")
        parent = joint.find("parent")
        child = joint.find("child")
        limit = joint.find("limit")
        if parent is None or child is None or limit is None:
            raise ValueError(f"Joint is missing parent, child, or limits: {joint.attrib}")
        children[parent.attrib["link"]].append(joint)
        child_links.add(child.attrib["link"])
        if float(limit.attrib["lower"]) >= float(limit.attrib["upper"]):
            raise ValueError(f"Joint has invalid limits: {joint.attrib['name']}")
    roots = set(links) - child_links
    if len(roots) != 1:
        raise ValueError(f"Expected one root link, found {sorted(roots)}")

    for link in links.values():
        for visual in link.findall("visual"):
            geometry = visual.find("geometry")
            mesh = geometry.find("mesh") if geometry is not None else None
            if mesh is None:
                raise ValueError(f"Only mesh visuals are supported for {link.attrib['name']}")
            mesh_name = Path(mesh.attrib["filename"]).stem
            mesh_path = urdf_path.parent / "meshes" / f"{mesh_name}.stl"
            if not mesh_path.is_file():
                mesh_path = urdf_path.parent / f"{mesh_name}.stl"
            if not mesh_path.is_file():
                raise FileNotFoundError(mesh_path)
            mesh_names.add(mesh_name)

    root = ET.Element("mujoco", {"model": "LEAP Hand"})
    ET.SubElement(
        root,
        "compiler",
        {"angle": "radian", "meshdir": "meshes", "autolimits": "true"},
    )
    default = ET.SubElement(root, "default")
    ET.SubElement(
        default,
        "joint",
        {"damping": "0.02", "armature": "0.001"},
    )
    ET.SubElement(default, "geom", {"contype": "0", "conaffinity": "0", "group": "2"})
    assets = ET.SubElement(root, "asset")
    for name in sorted(mesh_names):
        ET.SubElement(assets, "mesh", {"name": name, "file": f"{name}.stl"})
    worldbody = ET.SubElement(root, "worldbody")

    def add_link(
        parent_body: ET.Element,
        link_name: str,
        body_attributes: dict[str, str] | None = None,
        source_joint: ET.Element | None = None,
    ) -> None:
        link = links[link_name]
        attributes = {"name": link_name}
        if body_attributes:
            attributes.update(body_attributes)
        body = ET.SubElement(parent_body, "body", attributes)
        if source_joint is not None:
            limit = source_joint.find("limit")
            axis = source_joint.find("axis")
            ET.SubElement(
                body,
                "joint",
                {
                    "name": source_joint.attrib["name"],
                    "type": "hinge",
                    "axis": _format(_numbers(axis.attrib.get("xyz") if axis is not None else None, (1, 0, 0))),
                    "range": _format((float(limit.attrib["lower"]), float(limit.attrib["upper"]))),
                },
            )
        inertial = link.find("inertial")
        if inertial is None:
            raise ValueError(f"Link is missing inertia: {link_name}")
        origin = inertial.find("origin")
        inertial_origin = {} if origin is None else origin.attrib
        inertial_rpy = _numbers(inertial_origin.get("rpy"), (0.0, 0.0, 0.0))
        if any(abs(value) > 1e-10 for value in inertial_rpy):
            raise ValueError(f"Rotated inertial frames are not supported: {link_name}")
        inertial_values = inertial.find("inertia")
        mass = inertial.find("mass")
        if inertial_values is None or mass is None:
            raise ValueError(f"Link has incomplete inertia: {link_name}")
        ET.SubElement(
            body,
            "inertial",
            {
                "pos": _format(_numbers(inertial_origin.get("xyz"), (0.0, 0.0, 0.0))),
                "mass": mass.attrib["value"],
                "fullinertia": _format(
                    tuple(
                        float(inertial_values.attrib[key])
                        for key in ("ixx", "iyy", "izz", "ixy", "ixz", "iyz")
                    )
                ),
            },
        )
        for visual in link.findall("visual"):
            geometry = visual.find("geometry")
            mesh = geometry.find("mesh")
            name = Path(mesh.attrib["filename"]).stem
            origin = visual.find("origin")
            origin_values = {} if origin is None else origin.attrib
            rgba = "0.72 0.74 0.78 1"
            material = visual.find("material")
            color = material.find("color") if material is not None else None
            if color is not None:
                rgba = color.attrib["rgba"]
            ET.SubElement(
                body,
                "geom",
                {
                    "name": f"{link_name}_visual",
                    "type": "mesh",
                    "mesh": name,
                    "pos": _format(_numbers(origin_values.get("xyz"), (0.0, 0.0, 0.0))),
                    "euler": _format(_numbers(origin_values.get("rpy"), (0.0, 0.0, 0.0))),
                    "rgba": rgba,
                },
            )
        for joint in children[link_name]:
            joint_origin = joint.find("origin")
            origin_values = {} if joint_origin is None else joint_origin.attrib
            child = joint.find("child")
            add_link(
                body,
                child.attrib["link"],
                {
                    "pos": _format(_numbers(origin_values.get("xyz"), (0.0, 0.0, 0.0))),
                    "euler": _format(_numbers(origin_values.get("rpy"), (0.0, 0.0, 0.0))),
                },
                joint,
            )

    add_link(worldbody, roots.pop())
    tree = ET.ElementTree(root)
    ET.indent(tree, space="  ")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tree.write(output_path, encoding="unicode", xml_declaration=True)
    output_path.write_text(output_path.read_text(encoding="utf-8") + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--urdf", type=Path, default=ASSET_DIR / "robot.urdf")
    parser.add_argument("--output", type=Path, default=ASSET_DIR / "leap_hand.xml")
    args = parser.parse_args()
    convert(args.urdf, args.output)


if __name__ == "__main__":
    main()
