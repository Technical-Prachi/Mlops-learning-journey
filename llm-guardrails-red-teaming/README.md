# LLM Guardrails: Red Teaming and Governance for the Fine-Tuned IRIS Model

## Governing the Fine-Tuned LLM Guardrails on the IRIS Pipeline

**Model:** Fine-tuned Gemini on Vertex AI  
**Task:** Evaluate and govern the fine-tuned Gemini IRIS pipeline (see `llmops-gemini-finetuning/`) against prompt injection and prompt leakage.

---

# 1. Assignment Overview

This module extends the fine-tuned Gemini IRIS classification pipeline (`llmops-gemini-finetuning/`) with LLM-specific governance controls.

The implementation covers:

- Prompt Injection Red Teaming
- Prompt Leakage Red Teaming
- Input Guardrails
- Output Guardrails
- Guardrail Effectiveness Metrics
- Audit Logging
- Guarded inference pipeline
- V1 and V2 model evaluation

The guarded pipeline follows:

```text
User Input
    ↓
Input Guardrail
    ↓
Fine-Tuned Gemini on Vertex AI
    ↓
Output Guardrail
    ↓
Safe Classification Response
