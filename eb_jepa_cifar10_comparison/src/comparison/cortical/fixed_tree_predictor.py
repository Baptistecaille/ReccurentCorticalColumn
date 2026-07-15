"""Fixed tree predictor copied from ``Predictor/model/fixed_tree_predictor.py``.

Local adaptations: package-relative imports and removal of the executable demo
block. The model topology and forward equations are unchanged.
"""

import torch
import torch.nn as nn

from .column import CorticalColumn
from .decomposition import ChildDecomposition, FeedbackProjection
from .integration import LatentOutputProjection, LeafProjection, RecursiveIntegration
from .tree_structure import ColumnNode, build_tree_positions


class FixedTreePredictor(nn.Module):
    def __init__(
        self,
        n: int,
        L_max: int,
        dim_in: int,
        dim_hidden: int,
        dim_feedback: int,
        dim_U: int,
        dim_target: int,
        disable_lateral: bool = False,
        disable_feedback: bool = False,
    ) -> None:
        super().__init__()
        self.n = n
        self.L_max = L_max
        self.nodes = build_tree_positions(n, L_max)
        self.disable_lateral = disable_lateral
        self.disable_feedback = disable_feedback

        self.columns = nn.ModuleDict()
        self.decompositions = nn.ModuleDict()
        self.feedbacks = nn.ModuleDict()
        self.integrations = nn.ModuleDict()
        self.leaf_projs = nn.ModuleDict()

        for node in self.nodes:
            is_root = node.id == "root"
            current_dim_in = dim_in if is_root else dim_hidden
            self.columns[node.id] = CorticalColumn(
                dim_in=current_dim_in,
                dim_hidden=dim_hidden,
                dim_feedback=dim_feedback,
                is_root=is_root,
            )

            if node.level < L_max:
                self.decompositions[node.id] = ChildDecomposition(
                    dim_parent=dim_hidden,
                    dim_child=dim_hidden,
                    n=n,
                )
                self.feedbacks[node.id] = FeedbackProjection(
                    dim_parent=dim_hidden,
                    dim_feedback=dim_feedback,
                    n=n,
                )
                self.integrations[node.id] = RecursiveIntegration(
                    dim_hidden=dim_hidden,
                    dim_U=dim_U,
                    n=n,
                )
            else:
                self.leaf_projs[node.id] = LeafProjection(
                    dim_hidden=dim_hidden,
                    dim_U=dim_U,
                )

        self.out_proj = LatentOutputProjection(dim_U=dim_U, dim_target=dim_target)

    def forward(self, z_c: torch.Tensor) -> torch.Tensor:
        root_node = self.nodes[0]
        U_root = self._process_node(root_node, z_c, None)
        return self.out_proj(U_root)

    def _process_node(
        self,
        node: ColumnNode,
        X_v: torch.Tensor,
        R_v: torch.Tensor | None,
    ) -> torch.Tensor:
        H_v = self.columns[node.id].encode(X_v)
        B_v = self.columns[node.id](H_v, None, None)

        if node.level == self.L_max:
            return self.leaf_projs[node.id](B_v)

        X_children = self.decompositions[node.id](B_v)
        R_children = None if self.disable_feedback else self.feedbacks[node.id](B_v)
        U_children = self._process_siblings(node.children, X_children, R_children)
        return self.integrations[node.id](B_v, U_children)

    def _process_siblings(
        self,
        sibling_nodes: list[ColumnNode],
        X_list: list[torch.Tensor],
        R_list: list[torch.Tensor],
    ) -> list[torch.Tensor]:
        H_list = [
            self.columns[node.id].encode(inputs)
            for node, inputs in zip(sibling_nodes, X_list)
        ]

        if self.disable_lateral:
            A_list: list[torch.Tensor | None] = [None] * len(sibling_nodes)
        else:
            A_list = [
                self.columns[sibling_nodes[index].id].lateral_attention(
                    H_list[index],
                    H_list[:index] + H_list[index + 1 :],
                )
                for index in range(len(sibling_nodes))
            ]

        if self.disable_feedback:
            R_list = [None] * len(sibling_nodes)

        B_list = [
            self.columns[node.id](hidden, attention, feedback)
            for node, hidden, attention, feedback in zip(
                sibling_nodes,
                H_list,
                A_list,
                R_list,
            )
        ]

        U_list: list[torch.Tensor] = []
        for node, B_v in zip(sibling_nodes, B_list):
            if node.level == self.L_max:
                U_v = self.leaf_projs[node.id](B_v)
            else:
                X_children = self.decompositions[node.id](B_v)
                R_children = (
                    None
                    if self.disable_feedback
                    else self.feedbacks[node.id](B_v)
                )
                U_children = self._process_siblings(
                    node.children,
                    X_children,
                    R_children,
                )
                U_v = self.integrations[node.id](B_v, U_children)
            U_list.append(U_v)

        return U_list
