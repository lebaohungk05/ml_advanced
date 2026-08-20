"""STUB — Streamlit demo UI (Sprint 5).

Layout reference only: ``reference_repos/Multimodal-Image-Search-Engine/app.py``
shows the shape we want (modality radio -> text box or image uploader, a results
count slider, a grid of result images). That file is Gradio and we ship
Streamlit per the proposal, so read it for the layout, do not port the code.

Talk to ``app/api.py`` over HTTP (``POST /search``, ``POST /search/image``) rather
than importing the retriever — the API already owns config loading and indexing,
and keeping one boundary means the latency numbers in the report describe the
same path users hit.
"""

from __future__ import annotations

API_BASE_URL = "http://localhost:8000"


def main() -> None:
    # TODO(Sprint 5, Hiệp): st.title, modality radio, text_input / file_uploader,
    # top_k slider, then st.columns grid over response["hits"].
    raise NotImplementedError("TODO(Sprint 5, Hiệp): Streamlit demo UI")


if __name__ == "__main__":
    main()
