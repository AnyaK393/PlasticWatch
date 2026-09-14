import fs from "node:fs/promises";
import path from "node:path";
import { pathToFileURL } from "node:url";
const skillDir = "/Users/Prasad/.codex/plugins/cache/openai-primary-runtime/presentations/26.909.12148/skills/presentations";
const workspaceDir = "/Users/Prasad/Desktop/College_notes /4th SEM/EDI/AetherSea-II";
const tmpDir = path.join(workspaceDir, ".codex-build", "plasticwatch-pitch");
const finalPath = path.join(workspaceDir, "output", "pptx", "PlasticWatch_PS08_Pitch.pptx");
const { finalizePresentation } = await import(pathToFileURL(path.join(skillDir, "container_tools", "artifact_tool_utils.mjs")).href);
await fs.mkdir(path.dirname(finalPath), { recursive: true });
const result = await finalizePresentation({
  workspaceDir,
  candidatePath: path.join(tmpDir, "plasticwatch-ps08-pitch-draft.pptx"),
  finalPath,
  pythonExecutable: "/Users/Prasad/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3",
  integrityValidatorPath: path.join(skillDir, "container_tools", "inspect_presentation_package_integrity.py"),
  layoutValidatorPath: path.join(skillDir, "container_tools", "inspect_presentation_layout_geometry.py"),
  layoutArgs: ["--expected-slide-size-emu", "12192000,6858000", "--validate-bullet-geometry", "--validate-heading-fit"],
  explicitTotalSlideCount: 9,
  requiredNativeTableOwnerSlides: [],
  requiredNativeChartOwnerSlides: [],
  fontPolicy: { basis: "design", families: ["Helvetica Neue"] },
  verifyArtifactToolImport: true,
  receiptPath: path.join(tmpDir, "PlasticWatch_PS08_Pitch.validation.json"),
});
console.log(JSON.stringify(result));
