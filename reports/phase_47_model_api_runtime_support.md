# Phase 47 Model API Runtime Support

Status: complete
Generated at: 2026-06-04T09:12:13Z

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

- Startup artifact verify: 2.009 ms
- First inference: 0.779 ms
- Warm inference: 0.324 ms
- Max RSS: 27.375 MB
