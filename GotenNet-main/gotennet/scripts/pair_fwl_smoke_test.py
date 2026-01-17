import torch

from gotennet.models.components.layers import CosineCutoff
from gotennet.models.representation.gotennet import GotenNet


def build_complete_edge_index(num_nodes: int) -> torch.Tensor:
    nodes = torch.arange(num_nodes)
    pairs = torch.cartesian_prod(nodes, nodes)
    return pairs.t().contiguous()


def main() -> None:
    torch.manual_seed(0)
    num_nodes = 4
    hidden_dim = 32
    lmax = 2
    n_rbf = 8

    pos = torch.randn(num_nodes, 3)
    edge_index = build_complete_edge_index(num_nodes)
    edge_vec = pos[edge_index[0]] - pos[edge_index[1]]
    edge_diff = torch.norm(edge_vec, dim=-1, keepdim=True)
    atomic_numbers = torch.randint(1, 6, (num_nodes,))

    model = GotenNet(
        n_atom_basis=hidden_dim,
        n_interactions=2,
        lmax=lmax,
        n_rbf=n_rbf,
        cutoff_fn=CosineCutoff(5.0),
    )

    h_e, X_e, h_node, X_node = model.forward_pair_states(
        atomic_numbers, edge_index, edge_diff, edge_vec
    )

    num_edges = edge_index.size(1)
    assert h_e.shape == (num_edges, hidden_dim)
    assert len(X_e) == lmax
    for l, X_l in enumerate(X_e, start=1):
        assert X_l.shape == (num_edges, 2 * l + 1, hidden_dim)

    assert h_node.shape == (num_nodes, hidden_dim)
    assert len(X_node) == lmax
    for l, X_l in enumerate(X_node, start=1):
        assert X_l.shape == (num_nodes, 2 * l + 1, hidden_dim)

    for layer in model.pair_gata_list:
        if layer.last_attn is not None and layer.last_attn.numel() > 0:
            assert torch.isfinite(layer.last_attn).all()

    assert torch.isfinite(h_e).all()
    for X_l in X_e:
        assert torch.isfinite(X_l).all()
    assert torch.isfinite(h_node).all()
    for X_l in X_node:
        assert torch.isfinite(X_l).all()


if __name__ == "__main__":
    main()
