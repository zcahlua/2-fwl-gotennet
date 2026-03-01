import re

import torch

from gotennet import GotenNet
from gotennet.models.components.layers import CosineCutoff
from gotennet.models.representation.gotennet import build_two_hop_edges


def make_radius_edges(pos, batch, r1):
    src, dst = [], []
    for i in range(pos.size(0)):
        for j in range(pos.size(0)):
            if i == j or batch[i] != batch[j]:
                continue
            if torch.norm(pos[i] - pos[j]) <= r1:
                src.append(i)
                dst.append(j)
    edge_index = torch.tensor([src, dst], dtype=torch.long)
    edge_vec = pos[edge_index[0]] - pos[edge_index[1]]
    edge_diff = torch.norm(edge_vec, dim=-1)
    return edge_index, edge_diff, edge_vec


def edge_set(edge_index):
    return {(int(u), int(v)) for u, v in edge_index.t().tolist()}


def enumerate_wedges(edge_index1, edge_index2, num_nodes):
    src1, dst1 = edge_index1
    src2, dst2 = edge_index2
    key_base = num_nodes
    e2_keys = src2 * key_base + dst2

    incoming = [[] for _ in range(num_nodes)]
    outgoing = [[] for _ in range(num_nodes)]
    in_eid = [[] for _ in range(num_nodes)]
    out_eid = [[] for _ in range(num_nodes)]

    for eid in range(src1.numel()):
        i, j = int(src1[eid]), int(dst1[eid])
        if i == j:
            continue
        outgoing[i].append(j)
        out_eid[i].append(eid)
        incoming[j].append(i)
        in_eid[j].append(eid)

    records = []
    for k in range(num_nodes):
        if not incoming[k] or not outgoing[k]:
            continue
        for in_local, i in enumerate(incoming[k]):
            for out_local, j in enumerate(outgoing[k]):
                if i == j:
                    continue
                pair_key = i * key_base + j
                eij = torch.searchsorted(e2_keys, torch.tensor([pair_key], dtype=torch.long)).item()
                if eij >= e2_keys.numel() or int(e2_keys[eij]) != pair_key:
                    continue
                records.append(
                    {
                        "i": i,
                        "k": k,
                        "j": j,
                        "eik": in_eid[k][in_local],
                        "ekj": out_eid[k][out_local],
                        "eij": eij,
                    }
                )
    return records


def test_bent_geometry_r2_pruning_toggle():
    pos = torch.tensor([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0], [3.0, 5.0, 0.0]])
    batch = torch.tensor([0, 0, 0])
    edge_index1 = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)

    strict_e2, _, _, _ = build_two_hop_edges(edge_index1, 3, batch=batch, pos=pos, r2=4.0, skip_r2_pruning=False)
    assert (0, 2) not in edge_set(strict_e2)

    relaxed_e2, _, _, _ = build_two_hop_edges(edge_index1, 3, batch=batch, pos=pos, r2=4.0, skip_r2_pruning=True)
    relaxed_set = edge_set(relaxed_e2)
    assert (0, 2) in relaxed_set
    assert (2, 0) in relaxed_set

    wedges = enumerate_wedges(edge_index1, relaxed_e2, 3)
    assert any(w["i"] == 0 and w["k"] == 1 and w["j"] == 2 for w in wedges)


def test_diamond_two_wedge_paths():
    pos = torch.tensor([[0.0, 0.0, 0.0], [2.0, 1.0, 0.0], [2.0, -1.0, 0.0], [4.0, 0.0, 0.0]])
    batch = torch.tensor([0, 0, 0, 0])
    edge_index1, _, _ = make_radius_edges(pos, batch, r1=3.5)
    edge_index2, _, _, _ = build_two_hop_edges(edge_index1, 4, batch=batch, pos=pos, r2=3.0, skip_r2_pruning=True)
    e2_set = edge_set(edge_index2)
    assert (0, 3) in e2_set and (3, 0) in e2_set

    wedges = enumerate_wedges(edge_index1, edge_index2, 4)
    ks = {w["k"] for w in wedges if w["i"] == 0 and w["j"] == 3}
    assert ks == {1, 2}


def test_batch_boundary_is_respected():
    pos = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
            [3.0, 5.0, 0.0],
            [10.0, 0.0, 0.0],
            [13.0, 0.0, 0.0],
            [13.0, 5.0, 0.0],
        ]
    )
    batch = torch.tensor([0, 0, 0, 1, 1, 1])
    edge_index1 = torch.tensor([[0, 1, 1, 2, 3, 4, 4, 5], [1, 0, 2, 1, 4, 3, 5, 4]], dtype=torch.long)
    edge_index2, _, _, _ = build_two_hop_edges(edge_index1, 6, batch=batch, pos=pos, r2=4.0, skip_r2_pruning=True)

    for u, v in edge_set(edge_index2):
        assert int(batch[u]) == int(batch[v])
    assert (0, 2) in edge_set(edge_index2)
    assert (3, 5) in edge_set(edge_index2)


def test_forward_smoke_and_gata_subset():
    torch.manual_seed(0)
    pos = torch.tensor([[0.0, 0.0, 0.0], [2.0, 1.0, 0.0], [2.0, -1.0, 0.0], [4.0, 0.0, 0.0]])
    batch = torch.zeros(4, dtype=torch.long)
    z = torch.tensor([6, 6, 8, 1], dtype=torch.long)
    edge_index1, edge_diff1, edge_vec1 = make_radius_edges(pos, batch, r1=3.5)

    model = GotenNet(
        n_atom_basis=32,
        n_interactions=1,
        n_rbf=16,
        lmax=2,
        num_heads=4,
        cutoff_fn=CosineCutoff(3.5),
        r2_cutoff=3.0,
        skip_r2_pruning=True,
        far_pair_mode="2fwl_only",
    )
    out, eq = model(z, edge_index1, edge_diff1, edge_vec1, pos=pos, batch=batch)
    assert out.shape == (4, 32)
    assert eq.shape[0] == 4

    edge_index2, _, _, far_mask = build_two_hop_edges(edge_index1, 4, batch=batch, pos=pos, r2=3.0, skip_r2_pruning=True)
    gata_edge_index = edge_index2[:, ~far_mask]
    assert gata_edge_index.size(1) < edge_index2.size(1)
    assert any(w["i"] == 0 and w["j"] == 3 for w in enumerate_wedges(edge_index1, edge_index2, 4))


def test_debug_counters_change_with_pruning_mode(capsys):
    pos = torch.tensor([[0.0, 0.0, 0.0], [2.0, 1.0, 0.0], [2.0, -1.0, 0.0], [4.0, 0.0, 0.0]])
    batch = torch.zeros(4, dtype=torch.long)
    z = torch.tensor([6, 6, 8, 1], dtype=torch.long)
    edge_index1, edge_diff1, edge_vec1 = make_radius_edges(pos, batch, r1=3.5)

    strict = GotenNet(n_atom_basis=16, n_interactions=1, n_rbf=8, cutoff_fn=CosineCutoff(3.5), r2_cutoff=3.0, debug_2hop=True)
    _ = strict(z, edge_index1, edge_diff1, edge_vec1, pos=pos, batch=batch)
    strict_out = capsys.readouterr().out

    relaxed = GotenNet(
        n_atom_basis=16,
        n_interactions=1,
        n_rbf=8,
        cutoff_fn=CosineCutoff(3.5),
        r2_cutoff=3.0,
        debug_2hop=True,
        skip_r2_pruning=True,
    )
    _ = relaxed(z, edge_index1, edge_diff1, edge_vec1, pos=pos, batch=batch)
    relaxed_out = capsys.readouterr().out

    strict_far = int(re.search(r"Far pairs \(d>r2\): (\d+)", strict_out).group(1))
    relaxed_far = int(re.search(r"Far pairs \(d>r2\): (\d+)", relaxed_out).group(1))
    strict_wedges_far = int(re.search(r"Wedges: \d+ total, (\d+) target far pairs", strict_out).group(1))
    relaxed_wedges_far = int(re.search(r"Wedges: \d+ total, (\d+) target far pairs", relaxed_out).group(1))
    strict_e2 = int(re.search(r"\|E2\|=(\d+)", strict_out).group(1))
    relaxed_e2 = int(re.search(r"\|E2\|=(\d+)", relaxed_out).group(1))

    assert strict_far == 0
    assert strict_wedges_far == 0
    assert relaxed_far > 0
    assert relaxed_wedges_far > 0
    assert relaxed_e2 > strict_e2


def test_e1_to_e2_mapping_used_for_e2_aligned_tensor_indexing():
    pos = torch.tensor([[0.0, 0.0, 0.0], [2.0, 1.0, 0.0], [2.0, -1.0, 0.0], [4.0, 0.0, 0.0]])
    batch = torch.zeros(4, dtype=torch.long)
    edge_index1, _, _ = make_radius_edges(pos, batch, r1=3.5)
    edge_index2, _, e1_to_e2, _ = build_two_hop_edges(edge_index1, 4, batch=batch, pos=pos, r2=3.0, skip_r2_pruning=True)

    t = (torch.arange(edge_index2.size(1), dtype=torch.float32) + 1.0) * 1000.0
    wedges = enumerate_wedges(edge_index1, edge_index2, 4)
    assert wedges, "Expected non-empty wedge set"

    saw_mismatch = False
    for w in wedges:
        eik, ekj, eij = w["eik"], w["ekj"], w["eij"]
        t_ik = t[e1_to_e2[eik]]
        t_kj = t[e1_to_e2[ekj]]
        t_ij = t[eij]
        assert t_ik.item() == (int(e1_to_e2[eik]) + 1) * 1000.0
        assert t_kj.item() == (int(e1_to_e2[ekj]) + 1) * 1000.0
        assert t_ij.item() == (eij + 1) * 1000.0
        saw_mismatch = saw_mismatch or int(e1_to_e2[eik]) != int(eik)

    assert saw_mismatch
