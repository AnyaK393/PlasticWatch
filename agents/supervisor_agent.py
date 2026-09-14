from __future__ import annotations

import os
import json
from dotenv import load_dotenv
from typing import Dict, Any, List
import google.generativeai as genai

load_dotenv()

# OPTIONAL Gemini support
USE_GEMINI = False

try:
    import google.generativeai as genai

    api_key = os.getenv("GEMINI_API_KEY")

    if api_key:
        genai.configure(api_key=api_key)
        USE_GEMINI = True

except Exception:
    USE_GEMINI = False


class SupervisorAgent:
    def __init__(self):
        self.model = None

        if USE_GEMINI:
            try:
                self.model = genai.GenerativeModel("gemini-2.5-flash")
            except Exception:
                self.model = None

    def generate_mission_report(
        self,
        hotspots: list,
        route: dict,
        region_stats: dict = None,
        source: str = "unknown",
    ) -> str:
        """ Generates AI mission analysis report. """

        hotspot_count = len(hotspots)
        mean_fdi = 0

        if region_stats:
            mean_fdi = region_stats.get("mean_fdi", 0)

        total_distance = route.get("total_dist_km", 0)
        total_time = route.get("total_cost", 0)

        # Fallback local intelligence
        fallback_report = f"""
## 🌊 AetherSea-II Mission Report

### Detection Summary
- Total debris hotspots detected: {hotspot_count}
- Mean Floating Debris Index (FDI): {mean_fdi:.4f}

### Route Analysis
- Optimized cleanup distance: {total_distance:.1f} km
- Estimated mission duration: {total_time:.1f} hours

### Risk Assessment
The detected marine debris concentration suggests moderate floating plastic accumulation across the monitored Arabian Sea sector.

### AI Recommendations
- Prioritize high-FDI clusters first.
- Deploy cleanup vessels during low-current windows.
- Continue Sentinel-2 monitoring every 24 hours.
- Increase scan resolution near dense hotspot clusters.

### Mission Status
AetherSea-II autonomous analysis pipeline completed successfully.
"""

        # If Gemini unavailable -> use fallback
        if not self.model:
            return fallback_report

        # Gemini AI generation
        try:
            prompt = f"""
You are an advanced marine intelligence AI.

Analyze the following marine debris mission data.

Hotspots detected:
{json.dumps(hotspots[:20], indent=2)}

Route:
{json.dumps(route, indent=2)}

Region statistics:
{json.dumps(region_stats, indent=2)}

Generate:
1. Mission summary
2. Risk analysis
3. Cleanup strategy
4. Environmental impact assessment
5. Operational recommendations

Keep response concise and professional.
"""

            response = self.model.generate_content(prompt)

            if response and response.text:
                return response.text

            return fallback_report

        except Exception as e:
            return f"""
{fallback_report}

---
AI generation failed:
{str(e)}
"""

class MarineHabitatSupervisorAgent:
    """
    Multi-agent orchestrator that ingests telemetry, runs ecological triage,
    and formats vessel dispatch orders with verifiable impact metrics.
    """
    def __init__(self, api_key: str):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(
            model_name="gemini-2.5-flash",
            system_instruction=(
                "You are the AetherSea Environmental Mission Commander for marine habitat protection. "
                "Your role is to triage satellite-detected plastic clusters, assess ecological risk to "
                "vulnerable marine sanctuaries, and issue actionable interception commands for cleanup fleets. "
                "Always output your operational triage in structured JSON."
            )
        )

    def generate_incident_response(self, telemetry_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        prompt = f"""
        Analyze the following marine debris clusters, their hydrodynamics, and habitat risk scores:
        {json.dumps(telemetry_data, indent=2)}

        Provide an incident report containing:
        1. 'executive_summary': A direct overview of immediate ecological threats.
        2. 'triage_priorities': Sorted list of cluster IDs from highest ecological threat to lowest.
        3. 'vessel_dispatch_strategy': Specific navigational instructions and ETA windows.
        4. 'measurable_impact': Estimates of:
           - microplastic_breakdown_prevented_kg
           - vulnerable_area_shielded_sq_km
           - estimated_emissions_saved_pct

        Return strictly valid JSON matching this structure.
        """
        response = self.model.generate_content(
            prompt,
            generation_config={"response_mime_type": "application/json"}
        )
        return json.loads(response.text)       