from __future__ import annotations

from typing import Mapping, Optional, Tuple, Union

import e3nn.o3
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.utils import scatter

from gotennet.models.components.layers import (
    CosineCutoff,
    Dense,
    Distance,
    EdgeInit,
    NodeInit,
    str2act,
    str2basis,
)


def build_two_hop_edges(
    edge_index1: Tensor,
    num_nodes: int,
    batch: Optional[Tensor] = None,
    pos: Optional[Tensor] = None,
    r2: Optional[float] = None,
    topk2: Optional[int] = None,
    skip_r2_pruning: bool = False,
    max_edges_e2: int = -1,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    src1, dst1 = edge_index1
    device = edge_index1.device
    key_base = num_nodes

    e1_keys = src1 * key_base + dst1
    extra_pairs = []

    incoming = [[] for _ in range(num_nodes)]
    outgoing = [[] for _ in range(num_nodes)]
    for e in range(src1.numel()):
        i = int(src1[e])
        j = int(dst1[e])
        if i == j:
            continue
        outgoing[i].append(j)
        incoming[j].append(i)

    for k in range(num_nodes):
        in_nodes = incoming[k]
        out_nodes = outgoing[k]
        if not in_nodes or not out_nodes:
            continue
        in_t = torch.tensor(in_nodes, device=device, dtype=torch.long)
        out_t = torch.tensor(out_nodes, device=device, dtype=torch.long)
        ii = in_t.repeat_interleave(out_t.numel())
        jj = out_t.repeat(in_t.numel())
        mask = ii != jj
        if mask.any():
            extra_pairs.append(torch.stack([ii[mask], jj[mask]], dim=0))

    if extra_pairs:
        extra = torch.cat(extra_pairs, dim=1)
        if skip_r2_pruning:
            extra = torch.cat([extra, extra.flip(0)], dim=1)
        e2 = torch.cat([edge_index1, extra], dim=1)
    else:
        e2 = edge_index1.clone()

    self_mask = e2[0] != e2[1]
    e2 = e2[:, self_mask]

    keys = e2[0] * key_base + e2[1]
    uniq_keys = torch.unique(keys, sorted=True)
    src2 = torch.div(uniq_keys, key_base, rounding_mode="floor")
    dst2 = uniq_keys - src2 * key_base
    edge_index2 = torch.stack([src2, dst2], dim=0)

    if batch is not None:
        same_graph = batch[edge_index2[0]] == batch[edge_index2[1]]
        edge_index2 = edge_index2[:, same_graph]
        uniq_keys = uniq_keys[same_graph]

    if pos is not None and (r2 is not None or topk2 is not None):
        d2 = torch.norm(pos[edge_index2[0]] - pos[edge_index2[1]], dim=-1)
        keep = torch.ones_like(d2, dtype=torch.bool)
        is_e1_edge = torch.isin(uniq_keys, e1_keys)
        if (not skip_r2_pruning) and (r2 is not None):
            keep &= (d2 <= r2) | is_e1_edge
        if topk2 is not None:
            keep_topk = torch.zeros_like(keep)
            for i in range(num_nodes):
                node_mask = edge_index2[0] == i
                if batch is not None:
                    node_mask &= batch[edge_index2[0]] == batch[i]
                idx = torch.where(node_mask)[0]
                if idx.numel() == 0:
                    continue
                d = d2[idx]
                k = min(topk2, idx.numel())
                top_idx = torch.topk(d, k=k, largest=False).indices
                keep_topk[idx[top_idx]] = True
            keep_topk |= is_e1_edge
            keep &= keep_topk
        edge_index2 = edge_index2[:, keep]
        uniq_keys = uniq_keys[keep]

    if skip_r2_pruning and max_edges_e2 > 0 and edge_index2.size(1) > max_edges_e2:
        if pos is None or r2 is None:
            raise RuntimeError("skip_r2_pruning fallback requires `pos` and `r2`.")
        d2 = torch.norm(pos[edge_index2[0]] - pos[edge_index2[1]], dim=-1)
        keep = d2 <= r2
        edge_index2 = edge_index2[:, keep]
        uniq_keys = uniq_keys[keep]
        print(f"[Goten2FWL warn] E2 exceeded max_edges_e2={max_edges_e2}; fell back to r2-pruned E2.")

    e1_keys_sorted = src1 * key_base + dst1
    pos_in_e2 = torch.searchsorted(uniq_keys, e1_keys_sorted)
    valid = pos_in_e2 < uniq_keys.numel()
    valid_match = torch.zeros_like(valid)
    valid_match[valid] = uniq_keys[pos_in_e2[valid]] == e1_keys_sorted[valid]
    valid = valid & valid_match
    e1_to_e2 = pos_in_e2
    if not bool(valid.all()):
        raise RuntimeError("Failed to align E1 edges in E2.")

    edge_type2 = torch.ones(edge_index2.size(1), device=device, dtype=torch.long)
    edge_type2[e1_to_e2] = 0
    if pos is not None and r2 is not None:
        d2 = torch.norm(pos[edge_index2[0]] - pos[edge_index2[1]], dim=-1)
        far_pair_mask = d2 > r2
    else:
        far_pair_mask = torch.zeros(edge_index2.size(1), device=device, dtype=torch.bool)
    return edge_index2, edge_type2, e1_to_e2, far_pair_mask


def count_wedges(edge_index1: Tensor, edge_index2: Tensor, far_pair_mask: Tensor, num_nodes: int) -> Tuple[int, int]:
    src1, dst1 = edge_index1
    src2, dst2 = edge_index2
    key_base = num_nodes
    e2_keys = src2 * key_base + dst2
    far_lookup = torch.zeros(edge_index2.size(1), dtype=torch.bool, device=edge_index2.device)
    far_lookup[:] = far_pair_mask

    incoming = [[] for _ in range(num_nodes)]
    outgoing = [[] for _ in range(num_nodes)]
    for eid in range(src1.numel()):
        i = int(src1[eid])
        j = int(dst1[eid])
        if i == j:
            continue
        outgoing[i].append(j)
        incoming[j].append(i)

    n_wedges = 0
    n_wedges_far = 0
    for k in range(num_nodes):
        in_nodes = incoming[k]
        out_nodes = outgoing[k]
        if not in_nodes or not out_nodes:
            continue
        for i in in_nodes:
            for j in out_nodes:
                if i == j:
                    continue
                pair_key = i * key_base + j
                eij = torch.searchsorted(e2_keys, torch.tensor([pair_key], device=edge_index2.device, dtype=torch.long))
                eij = int(eij.item())
                if eij < e2_keys.numel() and int(e2_keys[eij]) == pair_key:
                    n_wedges += 1
                    n_wedges_far += int(far_lookup[eij].item())
    return n_wedges, n_wedges_far


class Local2FWLRefine(nn.Module):
    def __init__(self, hidden_dim: int, n_rbf: int, lmax: int, wedge_use_high_degree: bool = False):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.lmax = lmax
        self.wedge_use_high_degree = wedge_use_high_degree
        cdim = lmax if wedge_use_high_degree else 1
        wedge_feat_dim = (3 * n_rbf) + cdim
        in_dim = hidden_dim * 6 + wedge_feat_dim
        self.rho = nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim))
        self.gamma_w = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Sigmoid())
        self.gamma_t = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Tanh())

    def forward(
        self,
        t_e2: Tensor,
        h: Tensor,
        edge_index1: Tensor,
        edge_index2: Tensor,
        e1_to_e2: Tensor,
        dist_e1: Tensor,
        dist_e2: Tensor,
        rbf_e1: Tensor,
        rbf_e2: Tensor,
        sph_e1: Tensor,
        num_nodes: int,
    ) -> Tensor:
        src1, dst1 = edge_index1
        src2, dst2 = edge_index2
        device = t_e2.device

        key_base = num_nodes
        e2_keys = src2 * key_base + dst2

        incoming = [[] for _ in range(num_nodes)]
        outgoing = [[] for _ in range(num_nodes)]
        in_eid = [[] for _ in range(num_nodes)]
        out_eid = [[] for _ in range(num_nodes)]
        for eid in range(src1.numel()):
            i = int(src1[eid])
            j = int(dst1[eid])
            if i == j:
                continue
            outgoing[i].append(j)
            out_eid[i].append(eid)
            incoming[j].append(i)
            in_eid[j].append(eid)

        wedges = []
        ik_edges = []
        kj_edges = []
        for k in range(num_nodes):
            in_nodes = incoming[k]
            out_nodes = outgoing[k]
            if not in_nodes or not out_nodes:
                continue
            in_t = torch.tensor(in_nodes, dtype=torch.long, device=device)
            out_t = torch.tensor(out_nodes, dtype=torch.long, device=device)
            i_rep = in_t.repeat_interleave(out_t.numel())
            j_rep = out_t.repeat(in_t.numel())
            mask = i_rep != j_rep
            if not mask.any():
                continue
            i_rep = i_rep[mask]
            j_rep = j_rep[mask]
            k_rep = torch.full_like(i_rep, k)
            wedges.append(torch.stack([i_rep, k_rep, j_rep], dim=0))

            in_ids = torch.tensor(in_eid[k], dtype=torch.long, device=device)
            out_ids = torch.tensor(out_eid[k], dtype=torch.long, device=device)
            ik = in_ids.repeat_interleave(out_ids.numel())[mask]
            kj = out_ids.repeat(in_ids.numel())[mask]
            ik_edges.append(ik)
            kj_edges.append(kj)

        if not wedges:
            return t_e2

        wedge = torch.cat(wedges, dim=1)
        eik = torch.cat(ik_edges)
        ekj = torch.cat(kj_edges)
        i, k, j = wedge
        pair_keys = i * key_base + j
        eij = torch.searchsorted(e2_keys, pair_keys)
        valid = (eij < e2_keys.numel()) & (e2_keys[eij] == pair_keys)
        if not valid.any():
            return t_e2

        i, k, j, eik, ekj, eij = i[valid], k[valid], j[valid], eik[valid], ekj[valid], eij[valid]

        if self.wedge_use_high_degree and sph_e1.size(-1) >= self.lmax:
            contractions = []
            for l in range(1, self.lmax + 1):
                contractions.append((sph_e1[eik, l] * sph_e1[ekj, l]).unsqueeze(-1))
            c_feat = torch.cat(contractions, dim=-1)
        else:
            c_feat = (sph_e1[eik, 1] * sph_e1[ekj, 1]).unsqueeze(-1)

        g = torch.cat([rbf_e1[eik], rbf_e1[ekj], rbf_e2[eij], c_feat], dim=-1)
        rho_in = torch.cat(
            [t_e2[e1_to_e2[eik]], t_e2[e1_to_e2[ekj]], t_e2[eij], h[i], h[k], h[j], g],
            dim=-1,
        )
        wedge_msg = self.rho(rho_in)
        m = scatter(wedge_msg, eij, dim=0, dim_size=t_e2.size(0), reduce="sum")
        return t_e2 + self.gamma_w(m) * self.gamma_t(t_e2)


class GATA(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int = 8, dropout: float = 0.0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.q = nn.Linear(hidden_dim, hidden_dim)
        self.k = nn.Linear(hidden_dim, hidden_dim)
        self.v = nn.Linear(hidden_dim, hidden_dim)
        self.edge_gate = nn.Linear(hidden_dim, num_heads)
        self.out = nn.Linear(hidden_dim, hidden_dim)
        self.edge_update = nn.Sequential(nn.Linear(hidden_dim * 3, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim))

    def forward(self, edge_index2: Tensor, h: Tensor, t_ij: Tensor) -> Tuple[Tensor, Tensor]:
        src, dst = edge_index2
        q = self.q(h[dst]).view(-1, self.num_heads, self.hidden_dim // self.num_heads)
        k = self.k(h[src]).view(-1, self.num_heads, self.hidden_dim // self.num_heads)
        v = self.v(h[src]).view(-1, self.num_heads, self.hidden_dim // self.num_heads)

        logits = (q * k).sum(-1) / (self.hidden_dim // self.num_heads) ** 0.5
        logits = logits + self.edge_gate(t_ij)
        alpha = torch.softmax(logits, dim=0)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        msg = (alpha.unsqueeze(-1) * v).reshape(-1, self.hidden_dim)
        h_update = scatter(msg, dst, dim=0, dim_size=h.size(0), reduce="sum")
        h = h + self.out(h_update)

        t_upd = self.edge_update(torch.cat([h[src], h[dst], t_ij], dim=-1))
        t_ij = t_ij + t_upd
        return h, t_ij


class EQFF(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.SiLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )

    def forward(self, h: Tensor, X: Tensor) -> Tuple[Tensor, Tensor]:
        dh = self.net(h)
        return h + dh, X


class EdgeHTR(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(hidden_dim * 3, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim))

    def forward(self, t_e2: Tensor, h: Tensor, edge_index1: Tensor, e1_to_e2: Tensor) -> Tensor:
        src1, dst1 = edge_index1
        sub_t = t_e2[e1_to_e2]
        delta = self.mlp(torch.cat([h[src1], h[dst1], sub_t], dim=-1))
        t_new = t_e2.clone()
        t_new[e1_to_e2] = sub_t + delta
        return t_new


class InteractionBlock(nn.Module):
    def __init__(self, hidden_dim: int, n_rbf: int, lmax: int, num_heads: int, dropout: float, wedge_use_high_degree: bool):
        super().__init__()
        self.local2fwl = Local2FWLRefine(hidden_dim, n_rbf, lmax, wedge_use_high_degree=wedge_use_high_degree)
        self.gata1 = GATA(hidden_dim, num_heads=num_heads, dropout=dropout)
        self.eqff1 = EQFF(hidden_dim)
        self.htr = EdgeHTR(hidden_dim)
        self.gata2 = GATA(hidden_dim, num_heads=num_heads, dropout=dropout)
        self.eqff2 = EQFF(hidden_dim)

    def forward(
        self,
        h: Tensor,
        X: Tensor,
        t_e2: Tensor,
        edge_index1: Tensor,
        edge_index2: Tensor,
        gata_edge_index: Tensor,
        gata_to_e2: Tensor,
        e1_to_e2: Tensor,
        dist_e1: Tensor,
        dist_e2: Tensor,
        rbf_e1: Tensor,
        rbf_e2: Tensor,
        sph_e1: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        t_e2 = self.local2fwl(t_e2, h, edge_index1, edge_index2, e1_to_e2, dist_e1, dist_e2, rbf_e1, rbf_e2, sph_e1, h.size(0))
        h, gata_t = self.gata1(gata_edge_index, h, t_e2[gata_to_e2])
        t_e2 = t_e2.clone()
        t_e2[gata_to_e2] = gata_t
        h, X = self.eqff1(h, X)
        t_e2 = self.htr(t_e2, h, edge_index1, e1_to_e2)
        h, gata_t = self.gata2(gata_edge_index, h, t_e2[gata_to_e2])
        t_e2 = t_e2.clone()
        t_e2[gata_to_e2] = gata_t
        h, X = self.eqff2(h, X)
        return h, X, t_e2


class GotenNet(nn.Module):
    def __init__(
        self,
        n_atom_basis: int = 128,
        n_interactions: int = 4,
        radial_basis: Union[str, callable] = "expnorm",
        n_rbf: int = 32,
        cutoff_fn: Optional[object] = None,
        activation: Optional[Union[str, callable]] = F.silu,
        max_z: int = 100,
        num_heads: int = 8,
        attn_dropout: float = 0.0,
        lmax: int = 2,
        r2_cutoff: Optional[float] = None,
        topk2: Optional[int] = None,
        init_use_e2: bool = False,
        wedge_use_high_degree: bool = False,
        skip_r2_pruning: bool = False,
        far_pair_mode: str = "2fwl_only",
        max_edges_e2: int = -1,
        debug_2hop: bool = False,
        r_rbf_max: Optional[float] = None,
        **kwargs,
    ):
        super().__init__()
        if isinstance(activation, str):
            activation = str2act(activation)
        self.n_atom_basis = n_atom_basis
        self.n_interactions = n_interactions
        self.lmax = lmax
        self.cutoff = cutoff_fn.cutoff if cutoff_fn is not None else 5.0
        self.r2_cutoff = r2_cutoff if r2_cutoff is not None else self.cutoff
        self.topk2 = topk2
        self.init_use_e2 = init_use_e2
        self.wedge_use_high_degree = wedge_use_high_degree
        self.skip_r2_pruning = skip_r2_pruning
        self.far_pair_mode = far_pair_mode
        if self.far_pair_mode not in {"2fwl_only", "no_envelope", "extended"}:
            raise ValueError("far_pair_mode must be one of ['2fwl_only', 'no_envelope', 'extended']")
        self.max_edges_e2 = max_edges_e2
        self.debug_2hop = debug_2hop
        self._debug_printed = False

        self.node_init = NodeInit([n_atom_basis, n_atom_basis], n_rbf, self.cutoff, max_z=max_z, activation=activation)
        self.edge_init = EdgeInit(n_rbf, n_atom_basis)
        self.W_erp_2hop = nn.Linear(n_rbf, n_atom_basis)

        rb = str2basis(radial_basis)
        self.radial_basis = rb(cutoff=max(self.cutoff, self.r2_cutoff), n_rbf=n_rbf)
        self.r_rbf_max = r_rbf_max if r_rbf_max is not None else max(2 * self.cutoff, self.r2_cutoff)
        self.radial_basis_e2 = rb(cutoff=max(self.r_rbf_max, self.r2_cutoff), n_rbf=n_rbf)
        self.varphi1 = CosineCutoff(self.cutoff)
        self.varphi2 = CosineCutoff(self.r2_cutoff)
        self.varphi3 = CosineCutoff(self.r_rbf_max)

        self.A_na = nn.Embedding(max_z, n_atom_basis, padding_idx=0)
        self.sh_irreps = e3nn.o3.Irreps.spherical_harmonics(lmax)
        self.sphere = e3nn.o3.SphericalHarmonics(self.sh_irreps, normalize=False, normalization="norm")

        self.blocks = nn.ModuleList([
            InteractionBlock(n_atom_basis, n_rbf, lmax, num_heads, attn_dropout, wedge_use_high_degree)
            for _ in range(n_interactions)
        ])

    def forward(
        self,
        atomic_numbers: Tensor,
        edge_index: Tensor,
        edge_diff: Tensor,
        edge_vec: Tensor,
        pos: Optional[Tensor] = None,
        batch: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        num_nodes = atomic_numbers.size(0)
        h0 = self.A_na(atomic_numbers)
        rbf_e1 = self.radial_basis(edge_diff)
        h = self.node_init(atomic_numbers, h0, edge_index, edge_diff, rbf_e1)

        edge_index2, edge_type2, e1_to_e2, far_pair_mask = build_two_hop_edges(
            edge_index,
            num_nodes=num_nodes,
            batch=batch,
            pos=pos,
            r2=self.r2_cutoff,
            topk2=self.topk2,
            skip_r2_pruning=self.skip_r2_pruning,
            max_edges_e2=self.max_edges_e2,
        )

        if pos is not None:
            edge_vec2 = pos[edge_index2[0]] - pos[edge_index2[1]]
            dist_e2 = torch.norm(edge_vec2, dim=-1)
            safe = dist_e2 > 0
            edge_vec2 = edge_vec2.clone()
            edge_vec2[safe] = edge_vec2[safe] / dist_e2[safe].unsqueeze(-1)
        else:
            dist_e2 = edge_diff[e1_to_e2]
            edge_vec2 = edge_vec[e1_to_e2]

        rbf_e2 = self.radial_basis_e2(dist_e2) if self.skip_r2_pruning else self.radial_basis(dist_e2)
        env = torch.where(edge_type2 == 0, self.varphi1(dist_e2), self.varphi2(dist_e2))
        if self.far_pair_mode == "no_envelope":
            env = torch.where(far_pair_mask, torch.ones_like(env), env)
        elif self.far_pair_mode == "extended":
            env = torch.where(far_pair_mask, self.varphi3(dist_e2), env)
        env = env.unsqueeze(-1)

        gata_keep = ~far_pair_mask if self.far_pair_mode == "2fwl_only" else torch.ones_like(far_pair_mask)
        gata_edge_index = edge_index2[:, gata_keep]
        gata_to_e2 = torch.where(gata_keep)[0]

        t1 = (h[edge_index2[0]] + h[edge_index2[1]]) * self.edge_init.W_erp(rbf_e2)
        t2 = (h[edge_index2[0]] + h[edge_index2[1]]) * self.W_erp_2hop(rbf_e2)
        t_e2 = torch.where(edge_type2.unsqueeze(-1) == 0, t1, t2) * env

        sph_e1 = self.sphere(edge_vec)[:, 1:]
        sph_e2 = self.sphere(edge_vec2)[:, 1:]

        eq_dim = ((self.lmax + 1) ** 2) - 1
        X = torch.zeros((num_nodes, eq_dim, self.n_atom_basis), device=h.device)
        init_edges = edge_index2 if self.init_use_e2 else edge_index
        init_sph = sph_e2 if self.init_use_e2 else sph_e1
        msg = init_sph.unsqueeze(-1) * h[init_edges[1]].unsqueeze(1)
        X = X + scatter(msg, init_edges[0], dim=0, dim_size=num_nodes, reduce="sum")

        for block in self.blocks:
            h, X, t_e2 = block(
                h,
                X,
                t_e2,
                edge_index,
                edge_index2,
                gata_edge_index,
                gata_to_e2,
                e1_to_e2,
                edge_diff,
                dist_e2,
                rbf_e1,
                rbf_e2,
                sph_e1,
            )

        if self.debug_2hop and not self._debug_printed:
            n_e1 = edge_index.size(1)
            n_e2 = edge_index2.size(1)
            n_e2_only = int((edge_type2 == 1).sum().item())
            n_far = int((far_pair_mask & (edge_type2 == 1)).sum().item())
            print(f"[Goten2FWL debug] |E1|={n_e1}, |E2|={n_e2} ({(n_e2 / max(n_e1, 1)):.1f}x), |E2_only|={n_e2_only}")
            print(f"[Goten2FWL debug] Far pairs (d>r2): {n_far} of {n_e2_only} E2-only edges")
            n_wedges, n_wedges_far = count_wedges(edge_index, edge_index2, far_pair_mask, num_nodes)
            print(f"[Goten2FWL debug] Wedges: {n_wedges} total, {n_wedges_far} target far pairs")
            included_or_excluded = "excluded" if self.far_pair_mode == "2fwl_only" else "included"
            print(f"[Goten2FWL debug] far_pair_mode={self.far_pair_mode}: {n_far} pairs {included_or_excluded} from GATA")
            self._debug_printed = True

        return h, X


class GotenNetWrapper(GotenNet):
    def __init__(self, *args, max_num_neighbors: int = 32, **kwargs):
        super().__init__(*args, **kwargs)
        self.distance = Distance(self.cutoff, max_num_neighbors=max_num_neighbors, loop=True)

    def forward(self, inputs: Mapping[str, Tensor]) -> Tuple[Tensor, Tensor]:
        atomic_numbers, pos, batch = inputs.z, inputs.pos, inputs.batch
        edge_index, edge_diff, edge_vec = self.distance(pos, batch)
        return super().forward(atomic_numbers, edge_index, edge_diff, edge_vec, pos=pos, batch=batch)
