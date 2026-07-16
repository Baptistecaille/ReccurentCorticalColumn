from dataclasses import dataclass, field

# for debug
import pdb

def num_columns(n: int, L_max:int):
    """
    Args:
        n: facteur de branchement, n > 1
        L_max: profondeur maximale, L_max >= 0
    Returns:
        N(n, L_max) = (n^(L_max+1) - 1) / (n - 1)
    """
    
    if n < 2:
        raise ValueError("n cannot be lower than 2")
    elif L_max < 0:
        raise ValueError("L_max must be >= 0")
    else:
        return (n ** (L_max + 1) - 1) // (n - 1)
    


@dataclass
class ColumnNode:
    id: str
    level: int
    parent: "ColumnNode | None" = None
    children: list["ColumnNode"] = field(default_factory=list)


def build_tree_positions(n: int, L_max: int) -> list[ColumnNode]:

    """
    Construit l'arbre nœud par nœud, niveau par niveau (BFS). Chaque enfant
    reçoit son pointeur .parent au moment même de sa création (comme un graphe
    d'autodiff construit en avant, où chaque nœud enregistre son créateur), et
    s'ajoute à la liste .children de son parent.

    Returns:
        Liste de ColumnNode dans l'ordre BFS (racine d'abord, puis niveau 1,
        etc.). Longueur == num_columns(n, L_max).
    """

    root = ColumnNode(id="root", level=0)
    result = []
    
    def _build_levels(current_level_nodes: list[ColumnNode]) -> None:
        if not current_level_nodes:
            return
        
        result.extend(current_level_nodes)
        next_level_nodes = []
        
        if current_level_nodes[0].level < L_max:
            for parent in current_level_nodes:
                for i in range(n):
                    child_id = f"{parent.id}_{i}" if parent.id != "root" else f"node_{i}"
                    child = ColumnNode(
                        id=child_id,
                        level=parent.level + 1,
                        parent=parent
                    )
                    parent.children.append(child)
                    next_level_nodes.append(child)
                    
        _build_levels(next_level_nodes)

    _build_levels([root])
    return result

def level_of(node: ColumnNode) -> int:
    """
    root.level == 0
    root.children[0].level == 1
    """
    return node.level

def children_of(node: ColumnNode) -> list[ColumnNode]:
    """Retourne node.children directement (déjà rempli par build_tree_positions)."""
    return node.children

def parent_of(node: ColumnNode) -> ColumnNode | None:
    """Retourne node.parent directement (None pour la racine)."""
    return node.parent

def is_leaf(node: ColumnNode, L_max: int) -> bool:
    """True si level_of(node) == L_max."""
    return node.level == L_max
