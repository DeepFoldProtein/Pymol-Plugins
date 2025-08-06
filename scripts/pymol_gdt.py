import numpy as np
from pymol import cmd


@cmd.extend
def gdt_seq(sel_mov, sel_ref, quiet=1, _self=cmd):
    """
    sel_mov, sel_ref의 sequence alignment → 대응 CA atom mapping → GDT_TS 계산
    Usage: gdt_seq sel_movable, sel_reference
    """
    _self.align(f"{sel_mov} and name CA", f"{sel_ref} and name CA", object="aln")

    raw = _self.get_raw_alignment("aln")
    if not raw:
        print("Error: alignment object에 매핑 정보가 없습니다.")
        return

    coords_mov, coords_ref = [], []
    for column in raw:
        a1, a2 = column
        if a1[0] == sel_mov:
            mov_idx, ref_idx = a1[1], a2[1]
        else:
            mov_idx, ref_idx = a2[1], a1[1]
        c_mov = np.array(
            _self.get_model(f"{sel_mov} and index {mov_idx}").atom[0].coord
        )
        c_ref = np.array(
            _self.get_model(f"{sel_ref} and index {ref_idx}").atom[0].coord
        )
        coords_mov.append(c_mov)
        coords_ref.append(c_ref)

    coords_mov = np.vstack(coords_mov)
    coords_ref = np.vstack(coords_ref)

    scores = []
    for cutoff in (1.0, 2.0, 4.0, 8.0):
        dists = np.linalg.norm(coords_mov - coords_ref, axis=1)
        scores.append((dists <= cutoff).mean() * 100)
    gdt_ts = np.mean(scores)

    if not int(quiet):
        print(f"GDT_TS = {gdt_ts:.2f}")
    return gdt_ts
