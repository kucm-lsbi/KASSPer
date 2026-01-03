#!/usr/bin/env python3
import argparse
import os
import sys


def _dir_nonempty(path: str) -> bool:
    return os.path.isdir(path) and any(os.scandir(path))


def parse_args():
    p = argparse.ArgumentParser(
        prog="main.py",
        description="Unified CLI for OBO/OBN embedding + prediction",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument("--method", choices=["obo", "obn"], required=True, help="Choose pipeline method")
    p.add_argument("--fasta", required=True, help="Input FASTA path")
    p.add_argument("--smi", required=True, help="Input SMI path")
    p.add_argument("--output", required=True, help="Output directory (will be created if needed)")

    p.add_argument("--force-remake", action="store_true", help="Force re-generate embeddings even if exist")

    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--embed-only", action="store_true", help="Run embedding only")
    mode.add_argument("--predict-only", action="store_true", help="Run prediction only (requires existing embeddings)")

    p.add_argument("--result-name", default="results.csv", help="Result CSV filename under output dir")

    # Embedding options (passed through to embed_paired)
    p.add_argument("--prot-model", default="facebook/esm2_t36_3B_UR50D", help="Protein embedding model")
    p.add_argument("--chem-model", default="seyonec/ChemBERTa-zinc-base-v1", help="Chemical embedding model")
    p.add_argument("--bs-prot", type=int, default=1, help="Protein embedding batch size")
    p.add_argument("--bs-chem", type=int, default=1, help="Chemical embedding batch size")

    # Device control (best-effort without touching embed code)
    p.add_argument("--cpu", action="store_true", help="Disable CUDA via CUDA_VISIBLE_DEVICES='' (best-effort)")

    return p.parse_args()


def main():
    args = parse_args()

    if args.cpu:
        # Best-effort: make torch.cuda.is_available() false in most setups
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    # Resolve paths early
    fasta_path = os.path.abspath(args.fasta)
    smi_path = os.path.abspath(args.smi)
    out_dir = os.path.abspath(args.output)
    embed_root = os.path.join(out_dir, "embeddings")
    result_csv = os.path.join(out_dir, args.result_name)

    os.makedirs(out_dir, exist_ok=True)

    # ---------------- method routing ----------------
    if args.method == "obo":
        from obo_embed import embed_paired as embed_paired_fn
        from obo_pred import predict_all_pairs as predict_fn

        # OBO expects embeddings/prot + embeddings/lig
        prot_dir = os.path.join(embed_root, "prot")
        lig_dir = os.path.join(embed_root, "lig")

        def embeddings_exist() -> bool:
            return _dir_nonempty(prot_dir) and _dir_nonempty(lig_dir)

    elif args.method == "obn":
        from obn_embed import embed_paired as embed_paired_fn
        from obn_pred import predict_all_pairs as predict_fn

        # OBN expects embeddings/obn_prot + embeddings/obn_lig
        prot_dir = os.path.join(embed_root, "obn_prot")
        lig_dir = os.path.join(embed_root, "obn_lig")

        def embeddings_exist() -> bool:
            return _dir_nonempty(prot_dir) and _dir_nonempty(lig_dir)

    else:
        print("Unknown method:", args.method)
        return 1

    # ---------------- run embedding ----------------
    if not args.predict_only:
        if args.force_remake or not embeddings_exist():
            print("Embedding: start")
            embed_paired_fn(
                fasta_path=fasta_path,
                smi_path=smi_path,
                out_root=embed_root,
                prot_model=args.prot_model,
                chem_model=args.chem_model,
                bs_prot=args.bs_prot,
                bs_chem=args.bs_chem,
            )
        else:
            print("Embedding: skipped (existing embeddings found)")

        if args.embed_only:
            print(" Done (embed-only).")
            return 0

    # ---------------- run prediction ----------------
    if not embeddings_exist():
        print("Prediction requested but embeddings are missing/empty:")
        print("   -", prot_dir)
        print("   -", lig_dir)
        print("   Run without --predict-only or use --force-remake.")
        return 2

    print("Prediction: start")
    predict_fn(embed_root=embed_root, output_csv=result_csv)
    print(" Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
