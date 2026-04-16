"""Stepper HTML component for rendering pipeline progress as numbered circles.

Produces a self-contained HTML fragment suitable for embedding in a Gradio
``gr.HTML`` component. Each step is drawn as a circle connected by lines;
completed steps are filled, the active step pulses, and future steps are dimmed.
"""

from __future__ import annotations

_STEPPER_CSS = """\
<style>
  .pipeline-stepper {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 0;
    padding: 16px 0 12px;
    font-family: 'Segoe UI', Arial, sans-serif;
  }
  .pipeline-step {
    display: flex;
    flex-direction: column;
    align-items: center;
    position: relative;
  }
  .pipeline-circle {
    width: 36px; height: 36px;
    border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: 14px; font-weight: 600;
    border: 2px solid #555;
    background: #2a2a3a;
    color: #888;
    transition: all 0.3s ease;
  }
  .pipeline-label {
    margin-top: 6px;
    font-size: 11px;
    color: #888;
    white-space: nowrap;
    transition: color 0.3s ease;
  }
  .pipeline-line {
    width: 40px; height: 2px;
    background: #555;
    margin: 0 2px;
    align-self: center;
    margin-bottom: 22px;
    transition: background 0.3s ease;
  }
  .pipeline-step.completed .pipeline-circle {
    background: #4361ee;
    border-color: #4361ee;
    color: #fff;
  }
  .pipeline-step.completed .pipeline-label {
    color: #e0e0e0;
  }
  .pipeline-step.active .pipeline-circle {
    background: #4361ee;
    border-color: #4361ee;
    color: #fff;
    animation: pulse 1.5s ease-in-out infinite;
  }
  .pipeline-step.active .pipeline-label {
    color: #4361ee;
    font-weight: 600;
  }
  .pipeline-line.completed {
    background: #4361ee;
  }
  @keyframes pulse {
    0%, 100% { box-shadow: 0 0 0 0 rgba(67, 97, 238, 0.5); }
    50% { box-shadow: 0 0 0 8px rgba(67, 97, 238, 0); }
  }
</style>"""

_STEP_TEMPLATE = """\
<div class="pipeline-step {state}">
  <div class="pipeline-circle">{number}</div>
  <div class="pipeline-label">{label}</div>
</div>"""

_LINE_TEMPLATE = '<div class="pipeline-line {state}"></div>'


def render_stepper(steps: list[str], active_index: int = -1) -> str:
    """Render a horizontal stepper as an HTML fragment.

    Args:
        steps:       Labels for each pipeline step, in order.
        active_index: 0-based index of the currently active step.
                      Steps at indices < *active_index* are marked completed.
                      -1 means no step is active (all pending).

    Returns:
        An HTML string (with embedded CSS) suitable for ``gr.HTML``.
    """
    parts = [_STEPPER_CSS, '<div class="pipeline-stepper">']
    for i, label in enumerate(steps):
        if i > 0:
            line_state = "completed" if i <= active_index else ""
            parts.append(_LINE_TEMPLATE.format(state=line_state))
        step_state = (
            "completed" if i < active_index else "active" if i == active_index else ""
        )
        parts.append(_STEP_TEMPLATE.format(state=step_state, number=i + 1, label=label))
    parts.append("</div>")
    return "\n".join(parts)
