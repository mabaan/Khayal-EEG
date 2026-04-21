# Manual Test Checklist

1. Create a new profile in Setup and verify it becomes active.
2. Confirm model readiness is `Not Ready` before training.
3. Upload non-EDF in calibration and verify rejection.
4. Upload EDF and verify saved raw file path in response.
5. Start training and verify a user checkpoint is created.
6. Verify profile transitions to `Ready` after successful training.
7. Run Session inference and verify final Arabic sentence output.
8. Force Stage 2 invalid index (disable ollama) and verify fallback to first retrieval candidate.
9. Open Recent Sessions and verify calibration/training/inference records.
10. Restart app and verify profile/session persistence from local JSON files.
