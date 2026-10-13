# FactoryGuard AI

Local industrial-safety review application. The interface is **upload -> annotated image/video -> download**. All configuration and reports stay in the backend.

<!-- Demo video: on github.com click the pencil icon on this file and drag the video here. -->

## Run

Double-click `start.bat`. Open http://127.0.0.1:8501. Python, dependencies, FFmpeg and model weights are installed on this computer. On a fresh computer install Python 3.11+, FFmpeg on PATH, then use start.bat to install dependencies and download weights.

Uploads: JPG/PNG or MP4/AVI/MOV, up to 100 MB. Video processing covers the first 60 seconds by default. Output is silent H.264 MP4 sampled at `samples_per_second` (default 2; 10 gives smoother demos) at the source time scale. Multiple models make CPU analysis slower than playback. Settings are in `settings.json`.

## Implemented

- YOLO26n people, specialist forklifts, and a custom fine-tuned YOLO26n helmet and safety-vest detector.
- Explicit missing-hardhat and missing-vest detector candidates, with person association. No detection or ambiguous association means unknown.
- Person/forklift tracking and approximate proximity review alerts.
- Possible operator labels; these do not establish authorization or suppress alerts.
- Fire/smoke candidates with temporal persistence escalation.
- Fallen-person posture candidates with temporal persistence escalation. Lying posture is not proof of a fall transition.
- Restricted zones and camera-specific machine danger polygons.
- Machine activity using supplied source-time intervals; alternatively visual motion flags explicitly remain review evidence, not verified machine state.
- Optional calibrated ground-point distance estimates using four measured floor correspondences.
- Per-run risk indicator, repeated same-track events, detailed JSON/CSV, local SQLite run history including camera IDs when configured.
- Annotated image/video export with detection labels, review flags and score directly on the media.

## Evaluation (PPE model)

Test set: 37 frames I labeled by hand from 4 free stock videos. None of these frames were used in training.
Metric: mAP50, image size 960.

| Class | First model | After retraining |
|---|---|---|
| Hardhat | 0.11 | 0.34 |
| NO-Hardhat | 0.08 | 0.15 |
| Safety Vest | 0.79 | 0.77 |
| Person | 0.63 | 0.67 |
| All classes | 0.40 | 0.46 |

The first model was fine-tuned from YOLO26n on the Roboflow Universe dataset odmodeling/construction-site-safety. Retraining continued from it on that dataset plus 9 of my own labeled frames (repeated 3x) taken from clips different from the test clips.

Limitations:

- Bare-head detection is weak (recall 0.12), so a missing helmet is only a review candidate. Unknown PPE is never treated as a violation.
- The test set is small, so differences of a few points are noise.
- The first model scored 0.77 mAP50 on the dataset's own test split, which overstates real-world accuracy because that dataset contains near-duplicate images.
- Caps can be mistaken for hardhats.
- Most of my extra training frames show helmeted workers, so bare heads remain under-represented. Next step: more wide-angle and overhead training data.

## Camera-specific features

A random upload contains no information about authorized zones, actual machine state or physical scale. These features require a real camera profile. `camera-profile.example.json` explains the backend structure; its coordinates are illustrative and must be replaced before enabling it.

In `settings.json`, set `camera_id`, `machines`, and `calibration` for the actual camera. Machine `activity_source` accepts `intervals`, `visual_motion`, or `unknown`. Activity intervals use seconds from the start of the uploaded clip. No live PLC integration is included. Visual motion can be caused by people, lighting or camera movement.

Ground calibration requires four normalized image points and corresponding measured floor positions in meters, ordered around the quadrilateral. Distance uses box bottom centers, not body/vehicle extents. Ground-plane assumptions and seated/occluded subjects limit accuracy. Without calibration, proximity is a footpoint-to-forklift-box distance normalized by forklift width and is not measured in meters.

## Events and score

Per-clip IDs are temporary and can switch after occlusion. Repeated events count episodes for those temporary IDs; they do not establish permanent worker identity or misconduct. Brief gaps are bridged for up to 1.5 seconds by default. Fire/fallen-posture signals escalate only after multiple samples spanning one second. Single-image signals remain candidates.

Score is an illustrative priority indicator: each subject contributes its highest severity weight (15 medium, 30 high, 50 critical), summed and capped at 100. It is not an accident probability or validated risk assessment. Fire/smoke tracks are separate subjects. Peak scores are saved per run; camera IDs are only attached when explicitly configured.

## Outputs and privacy

`outputs/<run-id>/` holds `result.mp4` or `result.jpg`, `report.json`, and `events.csv`. `outputs/history.sqlite3` stores run summaries. Original uploads use temporary files and are removed after processing. Annotated media and reports persist locally until deleted. Re-rendering the same upload reuses its result.

## Models and attribution

The app uses YOLO26n for people. Except for the PPE model, which I fine-tuned myself, the specialists are public weights and are not claimed to be YOLO26 or trained by me.

- Forklift: https://huggingface.co/keremberke/yolov8s-forklift-detection (YOLOv8s; upstream model license not specified in metadata).
- Full PPE: own fine-tune of YOLO26n (`models/ppe_full.pt`), trained on the Roboflow Universe dataset odmodeling/construction-site-safety (CC BY 4.0) plus my own labeled frames from free stock footage. See Evaluation above.
- Fire/smoke: https://huggingface.co/rabahdev/fire-smoke-yolov8n (AGPL-3.0 per model card).
- Fall: https://huggingface.co/melihuzunoglu/human-fall-detection (AGPL-3.0 per model card).
- Optional older/light PPE: https://huggingface.co/melihuzunoglu/ppe-detection (AGPL-3.0; no no-vest class).

Exact revisions, URLs and SHA256 hashes are in `models/manifest.json`. Download/check via `python setup_models.py`. Review upstream code, weight and dataset license terms before distributing the product.

## Testing

```
.venv\Scripts\python.exe -m unittest discover -s tests
.venv\Scripts\python.exe smoke_test.py
```

Tests cover rule boundaries, temporal signals, missing PPE association, configured machine activity, calibration math, event routing, report/history persistence, invalid input cleanup and the upload-only UI. Real-model execution tests are not accuracy benchmarks. PPE accuracy is measured separately on hand-labeled real footage (see Evaluation); the other specialists have not been benchmarked.

## Deployment boundary

This is a completed local review application, not a certified safety system. It does not replace physical safety controls or fire alarms. Automated detection of arbitrary machinery types, live PLC connections, permanent worker identity and validated site accuracy are outside this local upload application. Fixed machinery is represented by camera-configured regions. The software supports calibration, but actual physical measurements cannot be supplied without access to the site.

## License and credits

- Dataset: odmodeling, "construction-site-safety", Roboflow Universe, CC BY 4.0.
- Test and demo videos: free stock footage from Pexels.
- Ultralytics YOLO and several of the models above are AGPL-3.0. Review the licenses before distributing.
