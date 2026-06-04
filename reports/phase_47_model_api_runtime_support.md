# Phase 47 Model API Runtime Support

Status: complete
Generated at: 2026-06-04T17:04:06Z

## Checks
- PASS phase25RollbackArtifactVerifies
- PASS phase46ArtifactVerifies
- PASS phase46EmbeddingDeclared
- PASS phase25EmbeddingDeclared
- PASS prefixPolicyPreserved
- PASS artifactRootSwitchesModelPath
- PASS performanceSmokeRecorded
- PASS responseContractStable
- PASS backendOwnedFieldsDoNotLeak

## Runtime Metadata
- Phase 25 embedding: intfloat/e5-base-v2
- Phase 46 embedding: intfloat/multilingual-e5-small
- Phase 46 model path: artifacts/phase_46_calibration_model_card_manifest_handoff_refresh/export/selected_jobfit_tf_phase46_multilingual_e5_small.keras

## Performance Smoke
- Startup artifact verify: 1.751 ms
- First inference: 0.451 ms
- Warm inference: 0.328 ms
- Max RSS: 62.766 MB
