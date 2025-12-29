import os, json, numpy as np, pandas as pd, joblib
from tqdm import tqdm
import sys
import contextlib
from obo_embed import embed_paired
def load_npz_vec(path):
    return np.load(path)["embedding"]

# --- 2) 예측 함수 -------------------------------------------------------
def predict_all_pairs(embed_root: str, output_csv: str):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir  = os.path.join(script_dir, "model")
    stacking_model = os.path.join(model_dir, "stack_lgb_rf_dihedral_sc.joblib")
    label_map_json = os.path.join(model_dir, "label_mapping.json")

    prot_dir = os.path.join(embed_root, "prot")
    lig_dir  = os.path.join(embed_root, "lig")

    prot_ids = {os.path.splitext(f)[0] for f in os.listdir(prot_dir) if f.endswith(".npz")}
    lig_ids  = {os.path.splitext(f)[0] for f in os.listdir(lig_dir)  if f.endswith(".npz")}
    common   = sorted(prot_ids & lig_ids)
    if not common:
        print("❌ 공통 ID가 없습니다. 임베딩 생성 결과를 확인하세요."); return
    print(f"공통 페어 수: {len(common)}개")


    model = joblib.load(stacking_model)
    label_list = json.load(open(label_map_json))["dihedral"]
    id2label = {i: lab for i, lab in enumerate(label_list)}

    rows = []
    for pid in tqdm(common, desc="Predicting"):
        p_vec = load_npz_vec(os.path.join(prot_dir, f"{pid}.npz"))
        l_vec = load_npz_vec(os.path.join(lig_dir,  f"{pid}.npz"))
        feats = np.concatenate([l_vec, p_vec])[None, :]
        pred = model.predict(feats)[0]
        rows.append({
            "ID": pid,
            "Prediction": id2label.get(pred, pred)
        })

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    pd.DataFrame(rows).to_csv(output_csv, index=False)
    print(f"✅ 결과 저장: {output_csv}")

# --- 3) 메인 ------------------------------------------------------------
def main(
    fasta_path: str,
    smi_path: str,
    output_path: str = "./work",
    force_remake: bool = False
):
    """
    output_path/
      └─ embeddings/ {prot,lig}/*.npz
      └─ results.csv
    """
    embed_root = os.path.join(output_path, "embeddings")
    result_csv = os.path.join(output_path, "results.csv")

    # 1) 임베딩 (이미 있으면 건너뜀)
    if force_remake or not (os.path.isdir(os.path.join(embed_root, "prot"))
                            and os.path.isdir(os.path.join(embed_root, "lig"))):
        print("🔄 임베딩 생성 시작 …")
        embed_paired(fasta_path, smi_path, embed_root)
    else:
        print("⏩ 임베딩 폴더가 존재하여 건너뜀")

    # 2) 예측
    predict_all_pairs(embed_root, result_csv)


# ---------------- CLI 예시 ----------------
if __name__ == "__main__":
    main(
        fasta_path = "./sample/sample.fasta",
        smi_path   = "./sample/sample.smi",
        output_path  = "./sample/obo/run_output",
        force_remake = True          # 이미 임베딩이 있으면 True 로 강제 재생성
    )
