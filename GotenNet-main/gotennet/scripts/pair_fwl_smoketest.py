import torch

from gotennet.models.representation.gotennet import (
    GotenNet,
    build_wedge_indices,
    split_to_components,
)
from gotennet.models.components.layers import CosineCutoff


def main() -> None:
    torch.manual_seed(0)
    num_nodes = 4
    pos = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    atomic_numbers = torch.tensor([1, 6, 8, 7])

    edges = [(i, j) for i in range(num_nodes) for j in range(num_nodes) if i != j]
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    src, dst = edge_index
    edge_vec = pos[dst] - pos[src]
    edge_diff = edge_vec.norm(dim=-1, keepdim=True)

    model = GotenNet(
        n_atom_basis=16,
        n_interactions=2,
        n_rbf=8,
        lmax=2,
        cutoff_fn=CosineCutoff(5.0),
    )
    h_e, X_e, t_e, r_components = model._initialize_edge_states(
        edge_index, edge_diff, edge_vec
    )
    assert h_e.shape == (edge_index.size(1), 16)
    assert len(X_e) == model.lmax
    for l, X_l in enumerate(X_e, start=1):
        assert X_l.shape == (edge_index.size(1), 2 * l + 1, 16)

    wedge_index = build_wedge_indices(edge_index, num_nodes=num_nodes)
    h_e_upd, X_e_upd, attn = model.pair_gata_list[0](
        h_e, X_e, t_e, r_components, wedge_index, return_attn=True
    )
    assert torch.isfinite(attn).all()
    assert torch.isfinite(h_e_upd).all()
    for X_l in X_e_upd:
        assert torch.isfinite(X_l).all()

    h_node, X_node = model(atomic_numbers, edge_index, edge_diff, edge_vec)
    assert h_node.shape == (num_nodes, 16)
    X_node_components = split_to_components(X_node, model.lmax, dim=1)
    for l, X_l in enumerate(X_node_components, start=1):
        assert X_l.shape == (num_nodes, 2 * l + 1, 16)
        assert torch.isfinite(X_l).all()


if __name__ == "__main__":
    main()
