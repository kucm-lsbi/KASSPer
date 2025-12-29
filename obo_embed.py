import os, re, torch, numpy as np
from transformers import AutoTokenizer, AutoModel

# ---------------- 공통 유틸 ---------------- #
def sanitize_id(raw: str) -> str:
    """ID 를 파일·딕셔너리 key 로 안전하게 변환"""
    return re.sub(r"[^A-Za-z0-9._-]", "_", raw.split()[0].lstrip(">")) or "unnamed"

# --------- SMI / FASTA 파서 --------- #
def load_smi(path, smiles_col: int = 0, id_col: int = 1):
    """
    SMI → {id: smiles}
    - 스페이스/탭 등 모든 공백을 구분자로 split()
    - 헤더/주석(#)·빈 줄 무시
    """
    d = {}
    with open(path, encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln or ln.startswith("#"):
                continue
            parts = ln.split()  # 공백 전부가 구분자
            if len(parts) <= max(smiles_col, id_col):
                print(f"[!] Skipped malformed line: {ln}")
                continue
            d[sanitize_id(parts[id_col])] = parts[smiles_col].strip()
    return d

def load_fasta(path):
    """FASTA → {id: sequence}"""
    d, hdr, seq = {}, None, []
    with open(path, encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            if ln.startswith(">"):
                if hdr:
                    d[sanitize_id(hdr)] = "".join(seq).upper()
                hdr, seq = ln, []
            else:
                seq.append(ln)
        if hdr:
            d[sanitize_id(hdr)] = "".join(seq).upper()
    return d

# --------- 메인 함수 --------- #
def embed_paired(
    fasta_path, smi_path, out_root,
    prot_model="facebook/esm2_t36_3B_UR50D",
    chem_model="seyonec/ChemBERTa-zinc-base-v1",
    bs_prot=1, bs_chem=1,
):
    os.makedirs(out_root, exist_ok=True)
    out_fa, out_sm = (os.path.join(out_root, "prot"),
                      os.path.join(out_root, "lig"))
    os.makedirs(out_fa, exist_ok=True)
    os.makedirs(out_sm, exist_ok=True)

    fasta  = load_fasta(fasta_path)
    smiles = load_smi(smi_path)
    common = sorted(set(fasta) & set(smiles))

    if not common:
        print("공통 ID가 없습니다.")
        return

    print(f"SMI:{len(smiles)}  FASTA:{len(fasta)}  ➜  공통:{len(common)}")

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ===== 단백질 임베딩 =====
    tok_p = AutoTokenizer.from_pretrained(prot_model)
    mdl_p = AutoModel.from_pretrained(prot_model).to(dev).eval()

    for i in range(0, len(common), bs_prot):
        ids  = common[i:i + bs_prot]
        seqs = [fasta[x] for x in ids]

        inp = tok_p(seqs, return_tensors="pt", padding=True).to(dev)
        with torch.no_grad():
            h = mdl_p(**inp, output_hidden_states=True).hidden_states[-1]

        for j, v in enumerate(ids):
            emb = h[j, inp["attention_mask"][j].bool()].mean(0).cpu().numpy()
            np.savez_compressed(os.path.join(out_fa, f"{v}.npz"), embedding=emb)

    # ===== SMILES 임베딩 =====
    tok_s = AutoTokenizer.from_pretrained(chem_model)
    mdl_s = AutoModel.from_pretrained(chem_model).to(dev).eval()

    for i in range(0, len(common), bs_chem):
        ids = common[i:i + bs_chem]
        sms = [smiles[x] for x in ids]

        inp = tok_s(sms, return_tensors="pt", padding=True, truncation=True).to(dev)
        with torch.no_grad():
            pooled = mdl_s(**inp).last_hidden_state.mean(1).cpu().numpy()

        for v, emb in zip(ids, pooled):
            np.savez_compressed(os.path.join(out_sm, f"{v}.npz"), embedding=emb)

    print(f"완료! {len(common)} 쌍 임베딩 저장")


# ----------- 사용 예시 ----------- #
# embed_paired(
#     fasta_path="./sample/sample.fasta",
#     smi_path="./sample/sample.smi",
#     out_root="./sample/sample_embeddings"
# )
