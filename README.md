# Income Classification: A Comparative Study of Predictive Models

> Can machine learning predict income brackets from demographic data, and does added model complexity actually justify its cost in a regulated business setting?
> My Break Through Tech AI capstone works through the full ML lifecycle on U.S. census data and closes with a business grounded model recommendation.

🔗 **[View the interactive presentation](https://sal-fakoya.github.io/Income-Classification-A-Comparative-Study-of-Predictive-Models/)**. Choose a business or technical audience, switch themes, and download a PDF.

## Highlights

| | |
|---|---|
| **Task** | Binary classification: income ≤ $50K vs > $50K (32,561 census records, 14 features) |
| **Recommended model** | Logistic Regression, 0.807 accuracy, 0.679 F1, **0.905 AUC** |
| **Challenger** | Keras neural network (32 to 16 ReLU), 0.810 accuracy, 0.687 F1 |
| **Verdict** | Deploy the simpler model. A 0.3 point gain does not justify losing interpretability in a credit context |

The result I lead with is the process, not the AUC: catching a proxy feature with Cramer's V (dropped `relationship`, a near proxy for sex at V=0.65), correcting class imbalance without contaminating the test distribution, validating model stability across three splits, and documenting the fairness limits of 1990s census data before modeling rather than after.

## Repository contents

`
├── index.html # Interactive presentation site (GitHub Pages)
├── PROJECT.md # Full written project report
├── Capstone.ipynb # The notebook: EDA, prep, models, neural network
└── README.md # You are here
`
