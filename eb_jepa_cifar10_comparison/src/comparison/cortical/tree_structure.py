"""Fixed-tree topology copied from ``Predictor/utils/tree_structure.py``.

Local adaptations: removed the unused ``pdb`` import and added the return type
of ``num_columns``. The tree construction algorithm is unchanged.
"""

from dataclasses import dataclass, field


def num_columns(n: int, L_max: int) -> int:
    """Return the number of nodes in a complete n-ary tree."""
    if n < 2:
        raise ValueError("n cannot be lower than 2")
    if L_max < 0:
        raise ValueError("L_max must be >= 0")
    return (n ** (L_max + 1) - 1) // (n - 1)


@dataclass
class ColumnNode:
    id: str
    level: int
    parent: "ColumnNode | None" = None
    children: list["ColumnNode"] = field(default_factory=list)


def build_tree_positions(n: int, L_max: int) -> list[ColumnNode]:
    """Build the complete n-ary tree in breadth-first order."""
    root = ColumnNode(id="root", level=0)
    result: list[ColumnNode] = []

    def _build_levels(current_level_nodes: list[ColumnNode]) -> None:
        if not current_level_nodes:
            return

        result.extend(current_level_nodes)
        next_level_nodes: list[ColumnNode] = []

        if current_level_nodes[0].level < L_max:
            for parent in current_level_nodes:
                for index in range(n):
                    child_id = (
                        f"{parent.id}_{index}"
                        if parent.id != "root"
                        else f"node_{index}"
                    )
                    child = ColumnNode(
                        id=child_id,
                        level=parent.level + 1,
                        parent=parent,
                    )
                    parent.children.append(child)
                    next_level_nodes.append(child)

        _build_levels(next_level_nodes)

    _build_levels([root])
    return result


def level_of(node: ColumnNode) -> int:
    return node.level


def children_of(node: ColumnNode) -> list[ColumnNode]:
    return node.children


def parent_of(node: ColumnNode) -> ColumnNode | None:
    return node.parent


def is_leaf(node: ColumnNode, L_max: int) -> bool:
    return node.level == L_max
