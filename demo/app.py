"""Interactive Gradio chatbot demo for the German Legal QA Assistant.

Features:
  - Conversational interface in German
  - Source citations displayed alongside answers
  - Toggle RAG on/off for comparison
  - Retrieval latency display
  - Legal disclaimer
"""

from __future__ import annotations

import time

import gradio as gr

# ---------------------------------------------------------------------------
# Pipeline (lazy-loaded)
# ---------------------------------------------------------------------------

_pipeline = None


def get_pipeline():
    """Lazy-load the QA pipeline on first query."""
    global _pipeline
    if _pipeline is None:
        from glqa.rag.pipeline import LegalQAPipeline

        _pipeline = LegalQAPipeline()
    return _pipeline


# ---------------------------------------------------------------------------
# Chat logic
# ---------------------------------------------------------------------------

DISCLAIMER = (
    "⚠️ **Hinweis:** Dies ist ein experimentelles KI-System zu Forschungszwecken. "
    "Die Antworten stellen **keine Rechtsberatung** dar. Für verbindliche Auskünfte "
    "wenden Sie sich bitte an eine qualifizierte Rechtsanwältin / einen qualifizierten Rechtsanwalt."
)

EXAMPLES = [
    "Was besagt §823 BGB?",
    "Kann ich einen Vertrag innerhalb von 14 Tagen widerrufen?",
    "Was ist der Unterschied zwischen Mord und Totschlag im StGB?",
    "Welche Kündigungsfristen gelten nach §622 BGB?",
    "Was regelt das Allgemeine Gleichbehandlungsgesetz (AGG)?",
    "Habe ich Anspruch auf Mutterschutz nach dem MuSchG?",
]


def respond(
    message: str,
    history: list[dict],
    use_rag: bool,
    show_sources: bool,
) -> tuple[list[dict], str]:
    """Process a user message and return the assistant response.

    Args:
        message: User's legal question.
        history: Chat history in messages format.
        use_rag: Whether to use retrieval-augmented generation.
        show_sources: Whether to display source citations.

    Returns:
        Updated history and sources text.
    """
    if not message.strip():
        return history, ""

    pipeline = get_pipeline()

    # Query the pipeline
    result = pipeline.ask(message, use_rag=use_rag)

    # Format answer
    answer = result.answer

    # Add timing info
    timing = ""
    if use_rag:
        timing = (
            f"\n\n---\n*Retrieval: {result.retrieval_time_ms:.0f}ms | "
            f"Generation: {result.generation_time_ms:.0f}ms*"
        )

    full_response = answer + timing

    # Format sources
    sources_text = ""
    if show_sources and result.sources:
        sources_text = "**Quellen:**\n\n"
        for i, src in enumerate(result.sources, 1):
            sources_text += f"{i}. {src}\n"

    # Update history
    history.append({"role": "user", "content": message})
    history.append({"role": "assistant", "content": full_response})

    return history, sources_text


def clear_chat() -> tuple[list, str]:
    """Clear the chat history."""
    return [], ""


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------


def create_demo() -> gr.Blocks:
    """Build the Gradio Blocks interface."""
    with gr.Blocks(
        title="German Legal QA – Deutsches Recht fragen",
    ) as demo:
        gr.Markdown(
            """
            # ⚖️ German Legal QA Assistant
            ### Fragen Sie deutsches Recht – powered by RAG + Fine-tuned LLM
            """
        )

        gr.Markdown(DISCLAIMER)

        with gr.Row():
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(
                    height=500,
                    label="Chat",
                )

                with gr.Row():
                    msg = gr.Textbox(
                        placeholder="z.B. Was besagt §823 BGB?",
                        label="Ihre Frage",
                        scale=4,
                        lines=1,
                    )
                    submit_btn = gr.Button("Fragen", variant="primary", scale=1)

                with gr.Row():
                    clear_btn = gr.Button("Chat leeren", size="sm")

                gr.Examples(
                    examples=EXAMPLES,
                    inputs=msg,
                    label="Beispiel-Fragen",
                )

            with gr.Column(scale=1):
                gr.Markdown("### Einstellungen")

                use_rag = gr.Checkbox(
                    value=True,
                    label="RAG verwenden",
                    info="Rechtstexte abrufen und als Kontext nutzen",
                )
                show_sources = gr.Checkbox(
                    value=True,
                    label="Quellen anzeigen",
                    info="Zitierte Rechtsquellen anzeigen",
                )

                sources_display = gr.Markdown(
                    value="",
                    label="Quellen",
                )

                gr.Markdown(
                    """
                    ---
                    ### Über dieses System

                    Dieses System nutzt:
                    - **Retrieval**: FAISS-Index über deutsche Gesetze & Urteile
                    - **Generation**: Fine-tuned Llama 3.2 / Qwen2.5
                    - **Daten**: Gesetze im Internet, Open Legal Data, GerLayQA

                    [GitHub Repository](https://github.com/) | 
                    [Forschungsprojekt](https://example.com)
                    """
                )

        # Event handlers
        submit_btn.click(
            fn=respond,
            inputs=[msg, chatbot, use_rag, show_sources],
            outputs=[chatbot, sources_display],
        ).then(fn=lambda: "", outputs=msg)

        msg.submit(
            fn=respond,
            inputs=[msg, chatbot, use_rag, show_sources],
            outputs=[chatbot, sources_display],
        ).then(fn=lambda: "", outputs=msg)

        clear_btn.click(fn=clear_chat, outputs=[chatbot, sources_display])

    return demo


# ---------------------------------------------------------------------------
# Launch
# ---------------------------------------------------------------------------


def launch_demo(port: int = 7860, share: bool = False) -> None:
    """Launch the Gradio demo server."""
    demo = create_demo()
    demo.launch(
        server_port=port,
        share=share,
        show_error=True,
    )


if __name__ == "__main__":
    launch_demo()
