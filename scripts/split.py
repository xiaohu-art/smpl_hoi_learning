"""
Post-process MJCF XML to split multi-joint bodies into chains of single-joint bodies.

Isaac Sim's MJCF importer merges multiple hinge joints on the same body into a single
spherical/D6 joint, which PhysX cannot drive in articulation mode. This script splits
each multi-joint body into a chain of intermediate dummy bodies, each with a single
revolute joint. Only the final body in the chain keeps the original geom and children.

Usage:
    from split_mjcf_joints import split_compound_joints
    split_compound_joints("input.xml", "output.xml")

Or as standalone:
    python split_mjcf_joints.py input.xml output.xml
"""

import copy
import sys
from lxml import etree


def split_compound_joints(input_path: str, output_path: str):
    """
    Read a MuJoCo XML, split all bodies with multiple joints into
    single-joint body chains, and write the result.
    """
    parser = etree.XMLParser(remove_blank_text=True)
    tree = etree.parse(input_path, parser)
    root = tree.getroot()

    # Process all body elements (bottom-up to avoid tree mutation issues)
    all_bodies = list(root.iter("body"))
    for body in reversed(all_bodies):
        joints = body.findall("joint")
        if len(joints) <= 1:
            continue  # Nothing to split

        _split_body(body, joints)

    # Write output with nice indentation
    tree.write(output_path, pretty_print=True, xml_declaration=False)

    # Re-read and write with the mujoco header (lxml drops processing instructions)
    with open(output_path, "r") as f:
        content = f.read()
    with open(output_path, "w") as f:
        f.write(content)

    print(f"Split compound joints: {input_path} -> {output_path}")


def _split_body(body: etree._Element, joints: list):
    """
    Split a body with N joints into a chain of N bodies.

    Original:
        <body name="X" pos="...">
            <joint name="X_x" .../>
            <joint name="X_y" .../>
            <joint name="X_z" .../>
            <geom .../>
            <body name="child" ...> ... </body>
        </body>

    Becomes:
        <body name="X_x_link" pos="...">          ← keeps original pos (relative to parent)
            <joint name="X_x" .../>
            <inertial pos="0 0 0" mass="0.0001" diaginertia="1e-9 1e-9 1e-9"/>
            <body name="X_y_link" pos="0 0 0">    ← zero offset (same location)
                <joint name="X_y" .../>
                <inertial pos="0 0 0" mass="0.0001" diaginertia="1e-9 1e-9 1e-9"/>
                <body name="X" pos="0 0 0">       ← original name, keeps geom + children
                    <joint name="X_z" .../>
                    <geom .../>
                    <body name="child" ...> ... </body>
                </body>
            </body>
        </body>
    """
    body_name = body.get("name", "unnamed")
    original_pos = body.get("pos", "0 0 0")
    n_joints = len(joints)

    # Collect all non-joint children (geoms, child bodies, inertial, site, etc.)
    non_joint_children = [child for child in body if child.tag != "joint"]

    # Remove everything from the original body
    for child in list(body):
        body.remove(child)

    # Build chain from outermost to innermost
    # The outermost body replaces the original body element in the tree
    # We reuse `body` as the outermost link

    current = body
    for i, joint in enumerate(joints):
        is_last = (i == n_joints - 1)

        if is_last:
            # Last joint: this body keeps the original name, geom, and children
            if i == 0:
                # First AND last (shouldn't happen since we check len > 1, but safety)
                current.append(joint)
            else:
                # Create the final body with original name
                final_body = etree.SubElement(current, "body")
                final_body.set("name", body_name)
                final_body.set("pos", "0 0 0")
                final_body.append(joint)
                # Add all original non-joint children to final body
                for child in non_joint_children:
                    final_body.append(child)
                current = final_body
        else:
            if i == 0:
                # Reuse the original body element as the first link
                joint_name = joint.get("name", f"{body_name}_j{i}")
                current.set("name", f"{joint_name}_link")
                # pos stays as original_pos (already set)
                current.append(joint)
                # Add tiny inertial for dummy body
                inertial = etree.SubElement(current, "inertial")
                inertial.set("pos", "0 0 0")
                inertial.set("mass", "0.0001")
                inertial.set("diaginertia", "1e-9 1e-9 1e-9")
            else:
                # Create intermediate dummy body
                joint_name = joint.get("name", f"{body_name}_j{i}")
                intermediate = etree.SubElement(current, "body")
                intermediate.set("name", f"{joint_name}_link")
                intermediate.set("pos", "0 0 0")
                intermediate.append(joint)
                # Add tiny inertial
                inertial = etree.SubElement(intermediate, "inertial")
                inertial.set("pos", "0 0 0")
                inertial.set("mass", "0.0001")
                inertial.set("diaginertia", "1e-9 1e-9 1e-9")
                current = intermediate

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(f"Usage: python {sys.argv[0]} <input.xml> <output.xml>")
        sys.exit(1)
    split_compound_joints(sys.argv[1], sys.argv[2])