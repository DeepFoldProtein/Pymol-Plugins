from uuid import uuid4

import numpy as np
from pymol import cmd


@cmd.extend
def gdt_seq(sel_mov, sel_ref, quiet=0):
    if not int(quiet):
        print(f"Starting GDT_TS calculation between '{sel_mov}' and '{sel_ref}'")

    aln_obj = f"_aln_{uuid4().hex[:8]}"
    cmd.align(
        f"{sel_mov} and name CA",
        f"{sel_ref} and name CA",
        object=aln_obj,
        quiet=int(quiet),
    )

    raw = cmd.get_raw_alignment(aln_obj)
    cmd.delete(aln_obj)

    if not raw:
        print("Error: alignment object에 매핑 정보가 없습니다.")
        return None

    coords_mov, coords_ref = [], []
    for a1, a2 in raw:
        if a1[1] is None or a2[1] is None:
            continue
        mov_idx, ref_idx = (a1[1], a2[1]) if a1[0] == sel_mov else (a2[1], a1[1])
        coords_mov.append(cmd.get_model(f"{sel_mov} and index {mov_idx}").atom[0].coord)
        coords_ref.append(cmd.get_model(f"{sel_ref} and index {ref_idx}").atom[0].coord)

    if not coords_mov:
        print("Error: 유효한 좌표 쌍이 없습니다.")
        return None

    coords_mov, coords_ref = np.vstack(coords_mov), np.vstack(coords_ref)
    cutoffs = (1.0, 2.0, 4.0, 8.0)
    scores = [
        (np.linalg.norm(coords_mov - coords_ref, axis=1) <= c).mean() * 100
        for c in cutoffs
    ]
    gdt_ts = np.mean(scores)

    if not int(quiet):
        print(f"Scores at cutoffs {cutoffs}: {[f'{s:.2f}%' for s in scores]}")
        print(f"GDT_TS = {gdt_ts:.2f}%")
    return gdt_ts


cmd.auto_arg[0]["gdt_seq"] = cmd.auto_arg[0]["align"]
cmd.auto_arg[1]["gdt_seq"] = cmd.auto_arg[1]["align"]
