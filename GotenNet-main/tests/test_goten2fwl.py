import torch
from gotennet import GotenNet, GotenNetWrapper
from gotennet.models.components.layers import CosineCutoff
from gotennet.models.representation.gotennet import build_two_hop_edges


def build_toy_edges(pos, batch, r1=2.0):
    src, dst = [], []
    for i in range(pos.size(0)):
        for j in range(pos.size(0)):
            if i == j:
                continue
            if batch[i] != batch[j]:
                continue
            if torch.norm(pos[i] - pos[j]) <= r1:
                src.append(i)
                dst.append(j)
    edge_index = torch.tensor([src, dst], dtype=torch.long)
    edge_vec = pos[edge_index[0]] - pos[edge_index[1]]
    edge_diff = torch.norm(edge_vec, dim=-1)
    return edge_index, edge_diff, edge_vec


def main():
    torch.manual_seed(0)
    z = torch.randint(1, 10, (8,))
    pos = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
        ]
    )
    batch = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])

    cutoff_fn = CosineCutoff(5.0)
    model = GotenNet(
        n_atom_basis=32,
        n_interactions=2,
        n_rbf=16,
        lmax=2,
        num_heads=4,
        cutoff_fn=cutoff_fn,
        r2_cutoff=8.0,
    )

    edge_index1, edge_diff1, edge_vec1 = build_toy_edges(pos, batch, r1=1.1)
    out, _ = model(z, edge_index1, edge_diff1, edge_vec1, pos=pos, batch=batch)
    assert out is not None
    print("Forward pass OK, output shape:", out.shape)

    edge_index2, edge_type2, e1_to_e2 = build_two_hop_edges(edge_index1, z.numel(), batch=batch, pos=pos, r2=8.0)
    e2_only = torch.where(edge_type2 == 1)[0]
    assert e2_only.numel() > 0
    with torch.no_grad():
        h0 = model.A_na(z)
        d2 = torch.norm(pos[edge_index2[0]] - pos[edge_index2[1]], dim=-1)
        rbf2 = model.radial_basis(d2)
        t2 = (h0[edge_index2[0]] + h0[edge_index2[1]]) * model.W_erp_2hop(rbf2)
        vals = t2[e2_only].abs().sum(dim=-1)
        assert torch.unique(vals).numel() > 1, "E2-only init is degenerate"

    loss = out.sum()
    loss.backward()
    found_grad = False
    for name, param in model.named_parameters():
        if any(k in name for k in ["W_erp_2hop", "local2fwl"]):
            assert param.grad is not None and param.grad.abs().sum() > 0, f"No gradient flow to {name}"
            found_grad = True
            print(f"Gradient OK: {name}")
    assert found_grad

    d_test = torch.tensor([6.0])
    env1 = model.varphi1(d_test)
    env2 = model.varphi2(d_test)
    assert abs(env1.item()) < 1e-6
    assert env2.item() > 0.01
    print("Envelope check OK")

    _ = GotenNetWrapper(n_atom_basis=16, n_interactions=1, n_rbf=8, cutoff_fn=CosineCutoff(3.0), r2_cutoff=4.0)
    print("Import/instantiation OK")

    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
