# Manual Test Checklist

1. Create a new profile in Setup and verify it becomes active.
2. Confirm model readiness is `Not Ready` before uploading a checkpoint.
3. Open Catalog and verify all 12 sentences are visible.
4. Open Catalog and verify all 25 words are visible.
5. Upload a Stage 1 checkpoint from the Session page.
6. Upload an EDF recording from the Session page.
7. Run Session inference and verify final Arabic sentence output.
8. Force Stage 2 invalid index by disabling Ollama and verify fallback to the first retrieval candidate.
9. Open Recent Sessions and verify Session records.
10. Restart the app and verify profile/session persistence from local JSON files.
