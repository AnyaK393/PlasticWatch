# PlasticWatch PS-08 Demo Video

Target length: 90 seconds

## 0:00-0:08 - Opening

Screen: Dashboard header and map.

Say: “PlasticWatch is an AI-GIS decision-support tool for PS-08. It turns image reports into verified plastic-waste cleanup priorities.”

## 0:08-0:20 - Transparency

Screen: Point to the `SIMULATION REPLAY` label.

Say: “This prototype uses clearly labelled synthetic reports. The same workflow later accepts citizen images, ward-worker reports, or street-camera frames. We do not present demo data as live evidence.”

## 0:20-0:38 - Hotspot map

Screen: Stay on **Hotspot map**. Select `PH-01`.

Say: “The map merges nearby reports into one hotspot. PH-01 has three reports, an AI confidence of 85 percent, and lies 35 metres from a drain. Those inputs produce a priority score of 99.1 out of 100.”

## 0:38-0:53 - Explainable evidence

Screen: Open **Evidence & AI** and select `PH-01`.

Say: “The system explains why it ranked this location: confidence, severity, recurrence, and drain exposure. AI confidence is evidence, not final truth. A human team still verifies the material before action.”

## 0:53-1:08 - Field operations

Screen: Open **Verification queue**.

Say: “Field teams receive a ranked queue. Critical locations near drains are inspected first. The verification result records whether the material is plastic, algae, foam, or another type of waste. This prevents wasted cleanup trips and unsupported responsibility claims.”

## 1:08-1:22 - TACO readiness

Screen: Return to **Evidence & AI** and point to the TACO status card.

Say: “We added TACO, a labelled litter dataset, as training data. It contains 1,500 images and 4,784 labelled objects. TACO does not perform detection by itself. Our next step is to fine-tune a four-class detector for bottles, bags and wrappers, mixed plastic, and uncertain waste.”

## 1:22-1:30 - Close

Screen: Return to the map with PH-01 selected.

Say: “PlasticWatch gives local teams an auditable path from report to verification to cleanup, while protecting against false positives and unsafe automated decisions.”

## Recording checklist

- Run `./venv/bin/streamlit run dashboard/app.py` before recording.
- Keep the browser at 100% zoom and use a 16:9 screen capture.
- Do not show terminal windows, credentials, or local paths.
- Keep the `SIMULATION REPLAY` label visible when describing demo data.
- Record the narration in one take after practising the transitions once.
