from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .config import DEMO_EDF_PATH, DEMO_MARKER_PATH, DEMO_MODEL_PATH
from .edf_trial_processor import process_trial_edf
from .stage1_model_adapter import Stage1DiffEAdapter, validate_checkpoint
from .stage2_decoder_adapter import Stage2DecoderAdapter


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description="Run the bundled Khayal demo inference end-to-end.")
    parser.add_argument("--model_path", default=str(DEMO_MODEL_PATH))
    parser.add_argument("--edf_path", default=str(DEMO_EDF_PATH))
    parser.add_argument("--marker_csv", default=str(DEMO_MARKER_PATH))
    parser.add_argument("--stage2_mode", default="qwen", choices=["qwen"])
    parser.add_argument("--top_k_words", type=int, default=8)
    parser.add_argument("--retrieval_topk", type=int, default=5)
    args = parser.parse_args()

    metadata = validate_checkpoint(args.model_path)
    print("model loaded successfully")
    print("checkpoint keys detected:", sorted(metadata["checkpoint"].keys()))
    print("arch values:", metadata["checkpoint"]["arch"])
    print("device:", metadata["device"])

    processed = process_trial_edf(args.edf_path, args.marker_csv)
    print("EDF loaded successfully:", Path(processed.edf_path).name)
    print("marker CSV detected:", Path(processed.marker_csv_path).name)
    print("3 imagine phases found:", len(processed.imagine_markers))
    print("slot tensor shapes:", processed.slot_tensor_shapes)
    if processed.warnings:
        print("preprocessing warnings:")
        for warning in processed.warnings:
            print("  -", warning)

    adapter = Stage1DiffEAdapter(args.model_path)
    stage1_slots = adapter.predict_slots(processed.slot_tensors, top_k=args.top_k_words)
    print("posterior vector shape for each slot:")
    for slot in stage1_slots:
        print(f"  slot {slot['slot']}: {len(slot['probabilities'])}")
        print(f"  top-k words for slot {slot['slot']}:")
        for item in slot["top_k"]:
            print(
                f"    - {item['word']} | {item['arabic']} | {item['probability']:.4f}"
            )

    decoder = Stage2DecoderAdapter()
    stage2_output = decoder.decode(stage1_slots, retrieval_topk=args.retrieval_topk, stage2_mode=args.stage2_mode)
    print("Stage 2 device:", stage2_output["device"])
    print("Stage 2 candidate sentences:")
    for candidate in stage2_output["candidate_sentences"]:
        print(
            f"  {candidate['rank']}. {candidate['sentence_id']} | {candidate['romanized']} | "
            f"{candidate['arabic']} | score={candidate['retrieval_score']:.4f}"
        )

    print("final prediction:")
    print("  Sentence ID:", stage2_output["prediction"]["sentence_id"])
    print("  Arabic:", stage2_output["prediction"]["arabic"])
    print("  Romanized:", stage2_output["prediction"]["romanized"])
    print("fallback status:", "used" if stage2_output["used_fallback"] else "not used")
    if stage2_output["warnings"]:
        print("stage2 warnings:")
        for warning in stage2_output["warnings"]:
            print("  -", warning)


if __name__ == "__main__":
    main()
