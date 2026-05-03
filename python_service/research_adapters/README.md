# Official Stage 2 LLM-RAG Method

This folder contains an isolated copy of the precomputed Stage 2 LLM-RAG
decoder.

Contents:

- `data/`: copied minimal metadata from `preprocessed_output`
- `precomputed_diffe/`: copied `classifier_A_results.json` and
  `classifier_B_results.json` for each subject
- `run_stage2_llm_rag_all_precomputed.py`: standalone runner
- `stage2_llm_rag.py`: retrieval + LLM reranker implementation
- `stage2_precomputed_common.py`: minimal local corpus/session/precomputed loader

Run from this folder:

```bash
python3 run_stage2_llm_rag_all_precomputed.py
```

Top-k sweep over EEG words:

```bash
python3 run_stage2_llm_rag_topk_sweep.py --k_min 1 --k_max 13
```

Default outputs land under:

```text
outputs/stage2_llm_rag_all_precomputed
```

Useful arguments:

- `--retrieval_mode`: `cosine`, `eeg`, or `hybrid`
- `--retrieval_topk`: how many retrieved candidates are sent to the LLM
- `--topk_words`: how many EEG top words per slot are shown in the prompt
- `--hybrid_alpha`: cosine / EEG interpolation weight in hybrid mode
- `--prompt_language`: `english` or `arabic`
- `--surface_form`: `romanized` or `arabic`
- `--embedding_model`: retrieval embedding model
- `--llm_model`: causal LLM used for shortlist selection
- `--embedding_local_only`: force local Hugging Face loading for the retriever
- `--llm_local_only`: force local Hugging Face loading for the LLM

Example:

```bash
python3 run_stage2_llm_rag_all_precomputed.py \
  --retrieval_mode hybrid \
  --hybrid_alpha 0.1 \
  --retrieval_topk 12 \
  --topk_words 8 \
  --prompt_language arabic \
  --surface_form arabic
```

To build an HTML report plus LaTeX tables from an existing run, point the root
report script at the run's `json/` directory. This generates both retrieval-only
and post-LLM subject/sentence tables.

```bash
python3 ..\..\display_stage2_llm_rag.py \
  --path outputs\stage2_llm_rag_all_precomputed\json \
  --out outputs\stage2_llm_rag_all_precomputed\html\stage2_llm_rag_report.html
```

The Arabic prompt mode reads the following table to recover Arabic surface forms
for the 25 vocabulary items.

| Romanized | Arabic | English |
|-----------|--------|---------|
| almareed | المريض | the patient |
| altabeeb | الطبيب | the doctor |
| almumarid | الممرض | the nurse |
| alsaydalee | الصيدلي | the pharmacist |
| al3amil | العامل | the worker |
| almustashfa | المستشفى | the hospital |
| alsareer | السرير | the bed |
| aldawa2 | الدواء | the medicine |
| alta3am | الطعام | the food |
| alma2 | الماء | the water |
| alhatif | الهاتف | the phone |
| alard | الأرض | the floor |
| yash3ur | يشعر | feels |
| yu7dar | يُحضر | brings |
| yughadir | يغادر | leaves |
| yasif | يصف | describes |
| yujeeb | يجيب | answers |
| yamsa7 | يمسح | wipes |
| yuratib | يرتّب | arranges |
| yu3id | يُعيد | returns |
| biljoo3 | بالجوع | with hunger |
| bilalam | بالألم | with pain |
| gheir | غير | not |
| muree7 | مريح | comfortable |
| latheeth | لذيذ | delicious |
