# AI-Augmented Observability System - Publication Resources

This document outlines the resources used to generate the IEEE Conference Paper for the project **"AI-Augmented Observability System for ISP Networks using LSTM Autoencoders"**.

## 📂 Core Files

| File | Type | Description |
| :--- | :--- | :--- |
| **[anomaly_detection_pipeline.ipynb](anomaly_detection_pipeline.ipynb)** | 🐍 Jupyter Notebook | **The Ground Truth.** Contains the full Python implementation of the LSTM Autoencoder, including data loading, log-normalization preprocessing, model training loop, and evaluation metrics. The technical specs in the paper are derived directly from this code. |
| **[IEEE_Paper.tex](IEEE_Paper.tex)** | 📄 LaTeX Source | **The Academic Paper.** A fully formatted IEEE Conference paper source file. It includes: <ul><li>**TikZ Flow Diagrams** for System Architecture and Model Design.</li><li>**Literature Review Table** comparing 10+ references.</li><li>**Technical Methodology** matching the notebook.</li></ul> |
| **[references.bib](references.bib)** | 📚 Bibliography | **BibTeX Database.** Contains 15+ researched citations used in the paper, ranging from foundational statistical methods (Lakhina 2004) to modern Deep Learning surveys (Bhuyan 2013, Vinayakumar 2019). |

## 🔗 Relationship Between Files

1.  **Code to Paper:** The *Methodology* section of `IEEE_Paper.tex` is a direct translation of the code in `anomaly_detection_pipeline.ipynb`.
    *   *Example:* The paper mentions a window size of $T=20$ and 64 hidden units because those are the variables set in the notebook.
2.  **Context to Paper:** The *Literature Review* section was built by analyzing external PDFs (`gu.pdf`, `IEEE CST published version 2013.pdf`) and standard academic history (Lakhina, Chandola) to create a realistic "Drawbacks vs. Solutions" narrative.

## 🛠️ How to Compile the Paper

To generate the final PDF from these files, you can use any LaTeX distribution (like TeX Live, MiKTeX) or an online editor like Overleaf.

### Using Overleaf (Recommended)
1.  Create a `New Project`.
2.  Upload `IEEE_Paper.tex` and `references.bib`.
3.  Upload any images if referenced (currently diagrams are drawn in code via TikZ, so no external images are strictly needed unless you add screenshots).
4.  Hit **Recompile**.

### Using Command Line (locally)
```bash
pdflatex IEEE_Paper.tex
bibtex IEEE_Paper
pdflatex IEEE_Paper.tex
pdflatex IEEE_Paper.tex
```
*(Note: Running pdflatex multiple times is necessary to resolve cross-references and the bibliography.)*
