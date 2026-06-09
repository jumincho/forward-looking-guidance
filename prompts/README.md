# Prompts

System prompts for the STRIDE pipeline modules.

| File | Role |
|------|------|
| `meta_plan/meta_plan.txt` | Meta-Planner — builds the general strategy + concrete plan |
| `supervisor/default.txt` | Supervisor — schedules `retrieve` / `rewrite` / `answer` per sub-question |
| `extractor/default.txt` | Extractor — distills atomic facts from retrieved documents |

The Reasoner and Fallback Reasoner system prompts are defined in `methods/stride.py`.
Forward-Looking Guidance suffixes are added to the Extractor / Reasoner inputs at run time.
