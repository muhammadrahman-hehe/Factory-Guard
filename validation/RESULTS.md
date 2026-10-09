# Validation — 2026-10-09

- Seven automated rule tests passed: zone footpoints, episode transitions, operator candidate geometry, proximity boundary, track association/expiry, event gap bridging, ambiguous/missing PPE association.
- Real-model Streamlit AppTest passed with all three detectors enabled. Included image: four people, one in zone; downloadable report and image generated.
- Warehouse clip `13751478_3840_2160_25fps.mp4`: sampled seconds 0 through 4 at 640px inference. One forklift detected in all five samples. Two people detected at second 0; only one detected at seconds 1–4. Thus the seated driver is still missed intermittently.
- Visually checked annotated frames. Second 0 now shows one forklift and a possible operator, without the earlier duplicate boxes or shirt-as-vest label. This setting adjustment is not independent accuracy validation.
- Detailed sample results are in `video-report.json`; annotated frames are alongside it.
- No precision/recall, calibrated distance, safety certification, or industrial deployment readiness is claimed. Further footage must include lighting changes, occlusions and examples without hazards. PPE recall remains unmeasured; unknown is deliberately not converted to a violation.
- Local Streamlit health endpoint returned `ok` after startup.
