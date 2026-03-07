from __future__ import annotations

import os
import shutil

import matplotlib.pyplot as plt
import scienceplots  # noqa: F401  # Registers "science" styles.


def apply_plot_style() -> None:
    plt.style.use(["science", "grid"])
    plt.rcParams["font.size"] = 10.5

    use_tex_raw = os.getenv("PRESSUREPROCESS_USE_TEX", "auto").strip().lower()
    if use_tex_raw == "auto":
        use_tex = shutil.which("latex") is not None
    elif use_tex_raw in {"1", "true", "yes", "on"}:
        use_tex = True
    elif use_tex_raw in {"0", "false", "no", "off"}:
        use_tex = False
    else:
        raise ValueError(
            "PRESSUREPROCESS_USE_TEX must be one of: auto, true, false"
        )

    plt.rc("text", usetex=use_tex)
    if use_tex:
        plt.rc("text.latex", preamble=r"\usepackage{mathpazo}")
