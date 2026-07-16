import torch
from Predictor.modules import CorticalColumn
from Predictor.model.fixed_tree_predictor import FixedTreePredictor
from Predictor.utils import build_tree_positions, num_columns


def make_column(is_root: bool = False) -> CorticalColumn:
    return CorticalColumn(dim_in=16, dim_hidden=16, dim_feedback=12, is_root=is_root)


def test_columns_do_not_share_lateral_weights():
    """P1: chaque colonne doit avoir son propre W_q (pas un module partagé)."""
    col_i, col_j = make_column(), make_column()
    assert not torch.allclose(col_i.W_q.weight, col_j.W_q.weight)


def test_lateral_attention_uses_receiving_column_weights():
    """P1: A_v dépend des projections DE LA COLONNE qui regarde, pas d'un
    triplet partagé — deux colonnes recevant les mêmes sœurs doivent
    produire des A_v différents."""
    col_i, col_j = make_column(), make_column()
    H_self = torch.randn(4, 16)
    H_others = [torch.randn(4, 16), torch.randn(4, 16)]
    A_i = col_i.lateral_attention(H_self, H_others)
    A_j = col_j.lateral_attention(H_self, H_others)
    assert not torch.allclose(A_i, A_j)


def test_lateral_attention_empty_siblings_is_zero():
    col = make_column()
    H_self = torch.randn(4, 16)
    A = col.lateral_attention(H_self, [])
    assert torch.allclose(A, torch.zeros_like(H_self))


def test_none_context_differs_from_zero_context():
    """P2: A_v=None doit sauter le terme de modulation entièrement, ce qui
    diffère de A_v=zeros (les biais de gamma/beta rendent FiLM(H, 0) != 0)."""
    col = make_column(is_root=False)
    H = col.encode(torch.randn(4, 16))
    R = torch.randn(4, 12)
    B_none = col(H, A_v=None, R_v=R)
    B_zero = col(H, A_v=torch.zeros(4, 16), R_v=R)
    assert not torch.allclose(B_none, B_zero, atol=1e-4)


def test_encode_called_once_per_column():
    """P3: encode() (E_v) ne doit être appelé qu'une fois par colonne, pas
    une fois au pré-calcul puis une seconde fois dans forward()."""
    col = make_column(is_root=False)
    calls = {"n": 0}
    original = col.E_v.forward

    def counting(x):
        calls["n"] += 1
        return original(x)

    col.E_v.forward = counting

    H = col.encode(torch.randn(4, 16))
    col.forward(H, A_v=None, R_v=torch.randn(4, 12))
    assert calls["n"] == 1


def test_total_encode_calls_matches_column_count():
    """P3, niveau modèle : le nombre total d'appels à E_v doit correspondre
    exactement au nombre de colonnes, pas au double."""
    n, L_max = 2, 2
    model = FixedTreePredictor(
        n=n, L_max=L_max, dim_in=16, dim_hidden=16, dim_feedback=16, dim_U=16, dim_target=16
    )
    call_counter = {"n": 0}
    for col in model.columns.values():
        original = col.E_v.forward

        def wrap(x, original=original):
            call_counter["n"] += 1
            return original(x)

        col.E_v.forward = wrap

    model(torch.randn(4, 16))
    assert call_counter["n"] == num_columns(n, L_max)


def test_root_only_tree():
    """P4: L_max=0 doit être accepté et produire un arbre à un seul nœud."""
    nodes = build_tree_positions(n=2, L_max=0)
    assert len(nodes) == 1
    assert nodes[0].parent is None
    assert nodes[0].children == []


def test_num_columns_accepts_l_max_zero():
    assert num_columns(2, 0) == 1
