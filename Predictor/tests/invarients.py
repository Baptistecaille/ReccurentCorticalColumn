import torch
from Predictor.model.fixed_tree_predictor import FixedTreePredictor
from Predictor.utils import num_columns
from Predictor.losses.jepa_losses import JEPAPredictorLoss

def test_column_count():
    """Invariant 1 : le nombre de colonnes exécutées == N(n, L_max)."""
    n, L_max = 2, 2
    model = FixedTreePredictor(
        n=n, L_max=L_max, dim_in=16, dim_hidden=16, dim_feedback=16, dim_U=16, dim_target=16
    )
    assert len(model.columns) == num_columns(n, L_max)


def test_independent_parameters():
    """Invariant 2 : deux positions distinctes ne partagent aucun paramètre."""
    n, L_max = 2, 2
    model = FixedTreePredictor(
        n=n, L_max=L_max, dim_in=16, dim_hidden=16, dim_feedback=16, dim_U=16, dim_target=16
    )
    params = [p.data_ptr() for p in model.parameters()]
    assert len(params) == len(set(params))
    
    ids = list(model.columns.keys())
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            assert not torch.allclose(model.columns[ids[i]].E_v.weight, model.columns[ids[j]].E_v.weight)


def test_lateral_permutation_invariance():
    """Invariant 3 : permuter le stockage des sœurs sans changer leur identité
    ne change pas le résultat de l'attention latérale."""
    n, L_max = 3, 2
    model = FixedTreePredictor(
        n=n, L_max=L_max, dim_in=16, dim_hidden=16, dim_feedback=16, dim_U=16, dim_target=16
    )
    root = model.nodes[0]
    siblings = root.children
    X_list = [torch.randn(4, 16) for _ in range(n)]
    R_list = [torch.randn(4, 16) for _ in range(n)]
    
    U_list_orig = model._process_siblings(siblings, X_list, R_list)
    
    perm = [1, 0, 2]
    siblings_perm = [siblings[i] for i in perm]
    X_list_perm = [X_list[i] for i in perm]
    R_list_perm = [R_list[i] for i in perm]
    
    U_list_perm = model._process_siblings(siblings_perm, X_list_perm, R_list_perm)
    
    assert torch.allclose(U_list_orig[0], U_list_perm[1], atol=1e-6)
    assert torch.allclose(U_list_orig[1], U_list_perm[0], atol=1e-6)
    assert torch.allclose(U_list_orig[2], U_list_perm[2], atol=1e-6)


def test_leaf_affects_root():
    """Invariant 4 : modifier un porteur de feuille modifie le porteur racine.
    Perturbe artificiellement une LeafProjection et vérifie que z_hat change."""
    n, L_max = 2, 2
    model = FixedTreePredictor(
        n=n, L_max=L_max, dim_in=16, dim_hidden=16, dim_feedback=16, dim_U=16, dim_target=16
    )
    z_c = torch.randn(4, 16)
    z_hat_orig = model(z_c).clone()
    
    leaf_id = list(model.leaf_projs.keys())[0]
    with torch.no_grad():
        model.leaf_projs[leaf_id].proj.weight.add_(1.0)
        
    z_hat_perturbed = model(z_c)
    assert not torch.allclose(z_hat_orig, z_hat_perturbed)


def test_gradient_reaches_every_column():
    """Invariant 5 : après un backward sur JEPAPredictorLoss, chaque
    CorticalColumn a un gradient non nul sur tous ses paramètres.

    n=3 (pas 2) : avec seulement 2 sœurs, l'attention latérale n'a qu'une
    seule clé candidate et softmax dégénère à 1.0 quel que soit W_q/W_k,
    donnant un gradient nul sur ces poids par construction mathématique
    (pas un bug) — indépendant du nombre de sœurs à partir de 3."""
    n, L_max = 3, 2
    model = FixedTreePredictor(
        n=n, L_max=L_max, dim_in=16, dim_hidden=16, dim_feedback=16, dim_U=16, dim_target=16
    )
    loss_fn = JEPAPredictorLoss("mse")
    
    z_c = torch.randn(4, 16)
    z_target = torch.randn(4, 16)
    
    z_hat = model(z_c)
    loss = loss_fn(z_hat, z_target)
    loss.backward()
    
    for name, column in model.columns.items():
        for p_name, p in column.named_parameters():
            assert p.grad is not None, f"Gradient manquant pour {name}.{p_name}"
            assert (p.grad != 0).any(), f"Gradient nul pour {name}.{p_name}"


def test_ablation_equivalence():
    """Invariant 6 : disable_lateral / disable_feedback doivent couper la
    voie EXACTEMENT (A_v/R_v forcés à None, terme de modulation sauté), pas
    seulement "changer le résultat" en passant des zéros — FiLM(H, 0) != 0 à
    cause des biais de gamma/beta, donc zéroter le contexte n'est pas
    équivalent à l'ablation. On le prouve en perturbant les poids qui ne
    devraient plus être utilisés du tout : si la sortie ne bouge pas, la
    voie est bien coupée, pas juste atténuée."""
    n, L_max = 2, 2

    model_no_lateral = FixedTreePredictor(
        n=n, L_max=L_max, dim_in=16, dim_hidden=16, dim_feedback=16, dim_U=16, dim_target=16,
        disable_lateral=True,
    )
    z_c = torch.randn(4, 16)
    z_hat_before = model_no_lateral(z_c).clone()
    with torch.no_grad():
        for column in model_no_lateral.columns.values():
            if not column.is_root:
                column.W_q.weight.add_(1.0)
                column.mod_lat.gamma.bias.add_(1.0)
    z_hat_after = model_no_lateral(z_c)
    assert torch.allclose(z_hat_before, z_hat_after), (
        "disable_lateral=True doit rendre le modèle insensible à mod_lat/W_q"
    )

    model_no_feedback = FixedTreePredictor(
        n=n, L_max=L_max, dim_in=16, dim_hidden=16, dim_feedback=16, dim_U=16, dim_target=16,
        disable_feedback=True,
    )
    z_hat_before = model_no_feedback(z_c).clone()
    with torch.no_grad():
        for column in model_no_feedback.columns.values():
            if not column.is_root:
                column.mod_fb.gamma.bias.add_(1.0)
    z_hat_after = model_no_feedback(z_c)
    assert torch.allclose(z_hat_before, z_hat_after), (
        "disable_feedback=True doit rendre le modèle insensible à mod_fb"
    )

    model_full = FixedTreePredictor(
        n=n, L_max=L_max, dim_in=16, dim_hidden=16, dim_feedback=16, dim_U=16, dim_target=16
    )
    z_hat_full = model_full(z_c)
    model_ablated = FixedTreePredictor(
        n=n, L_max=L_max, dim_in=16, dim_hidden=16, dim_feedback=16, dim_U=16, dim_target=16,
        disable_lateral=True, disable_feedback=True,
    )
    model_ablated.load_state_dict(model_full.state_dict())
    z_hat_ablated = model_ablated(z_c)
    assert not torch.allclose(z_hat_full, z_hat_ablated)


if __name__ == "__main__":
    test_column_count()
    test_independent_parameters()
    test_lateral_permutation_invariance()
    test_leaf_affects_root()
    test_gradient_reaches_every_column()
    test_ablation_equivalence()