---
name: develop-ml-systems
description: "Designs, implements, tests, operates, and reviews ML, LLM, RAG, inference, and data-processing systems using the team's internal libraries and production conventions. Use for model services, LiteLLM integrations, prompts and RAG pipelines, training or inference code, evaluations, notebooks, Prefect or Airflow workflows, S3 data flows, model telemetry, and ML production-readiness work."
---

# Develop ML Systems

Treat code, data, model behavior, evaluation, and operations as one production system.

## Workflow

1. Define the task, user-visible contract, data boundary, model/provider, quality target, latency and cost constraints, and deployment context.
2. Read [the ML platform reference](references/ml-platform.md).
3. Choose internal libraries documented by the current repository or platform baseline, then establish versioned inputs, outputs, model configuration, and artifacts.
4. Implement the end-to-end path: validation, transformations, model call, output validation, persistence, retries, and fallback.
5. Create representative evaluations and regression tests before relying on aggregate quality claims.
6. Add traceable model/service metadata and bounded operational telemetry.
7. Verify failure behavior for empty input, schema drift, timeout, partial batches, provider outage, and retries.

## Rules

- Prevent train/test leakage and train/serve skew.
- Make model, prompt, threshold, dataset, and service versions recoverable from results.
- Validate structured model output before downstream use.
- Make side effects idempotent before adding retries.
- Keep internal data on providers explicitly allowed by repository policy. If no such policy is available, do not send data to an external provider.
- During review-only tasks, separate confirmed defects from risks that require data or runtime evidence.

## Adjacent skills

Hand off when the skill is installed; otherwise apply this skill's own rules and continue: $develop-python (implementation conventions), $secure-applications (secrets and untrusted input), $build-prefect-flows (Prefect), $instrument-observability (telemetry).
