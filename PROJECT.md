# Income Classification: A Comparative Study of Predictive Models

**Sal Fakoya · Break Through Tech AI Capstone**

This project evaluates whether machine learning can accurately predict income brackets from demographic data, and, more importantly, whether complex models deliver tangible business value over simpler, interpretable alternatives. Using 32,561 census records, I benchmark multiple approaches to determine the optimal balance between predictive power and operational transparency.

**Live presentation:** [sal-fakoya.github.io/Capstone-Project](https://sal-fakoya.github.io/Capstone-Project/) · **Notebook:** [Capstone.ipynb](Capstone.ipynb)

---

## Business context and problem framing

Accurate income estimation is a cornerstone of responsible lending, credit risk assessment, and financial inclusion. Institutions rely on these estimates daily to underwrite loans, set credit limits, and comply with fair lending regulations. A robust predictive model can streamline these decisions; a biased or opaque one introduces regulatory exposure and reputational risk. This project directly addresses that tension by prioritizing both accuracy and interpretability.

**Objective:** predict the binary label `income_binary` (≤ $50K vs > $50K) from 14 demographic and employment features. The downstream use case is credit risk adjacent, since organizations extending credit require reliable income proxies.

**Data:** UCI Adult Census dataset, 32,561 observations, 76/24 class imbalance, evaluated on accuracy, F1 score (positive class), and AUC ROC over a stratified 80/20 train/test split.

## Data profile

The label is imbalanced: 75.9 percent of records fall below $50K and 24.1 percent above. A naive classifier predicting the majority class would reach 76 percent accuracy while providing no real value, so accuracy alone was never the deciding metric. Model selection is driven by F1 and AUC.

Profiling the 14 features surfaced redundancy and risk that needed to be resolved before modeling:

**Cramer's V analysis** across categorical variables flagged `relationship` and `sex_selfID` at 0.65. Given values like Husband and Wife, that column operates as a near proxy for sex. It also overlapped with `marital-status` at 0.49. I dropped `relationship` to close that proxy pathway.

**Redundancy check:** `education` and `education-num` encode the same information, once categorical and once ordinal. I retained the numeric version and excluded the text one.

**Multicollinearity:** Variance Inflation Factor scores for the numeric features were near 1.0 across the board, so no concerning collinearity among continuous predictors.

**Missing data:** `age`, `hours-per-week`, `workclass`, `occupation`, and `native-country` contained missing values, addressed via imputation rather than listwise deletion to preserve sample size.

## Responsible AI and bias considerations

The dataset reflects 1990s U.S. census demographics, with inherent skews in race, nationality, and gender representation. A model trained on this historical data risks perpetuating or amplifying those skews if deployed without oversight. Subgroups with thin representation will likely show higher error rates, and the fixed $50K threshold is not meaningful today without inflation adjustment. The binary sex field also excludes nonbinary identities entirely. While the data cannot be retroactively balanced, the risk can be managed through feature selection, subgroup evaluation, and explicit deployment guardrails. My position: a model like this is appropriate as a decision support tool with human review, never as the final authority on whether a person receives credit.

## Preprocessing and pipeline configuration

Categorical variables were transformed into binary indicators using `OneHotEncoder(drop="first")` inside a `ColumnTransformer`, with standard scaling applied to the numeric remainder, producing a 77 dimension feature space. I used an 80/20 stratified train/test split (random_state=42), then addressed class imbalance by downsampling the majority class in the training set only, leaving the test set at its natural distribution to preserve evaluation rigor.

## Model benchmarking and baseline comparison

Three algorithms competed on identical data:

| Model | Accuracy | F1 Score | AUC |
|---|---|---|---|
| **Logistic Regression** | **0.807** | **0.679** | **0.905** |
| KNN | 0.785 | 0.658 | 0.880 |
| Decision Tree | 0.771 | 0.621 | 0.774 |

Logistic Regression led across every metric, and the ranking held when I reran the comparison at 20, 30, and 40 percent test splits. GridSearchCV across regularization strength and penalty type (C in 0.01 to 100, l1/l2, liblinear solver) landed on C=1 with l2, essentially the defaults, at a cross validated AUC of 0.904.

The Decision Tree was left unpruned (default `max_depth`), so its clear underperformance likely reflects overfitting on its part rather than a purely algorithmic gap between linear and tree based models.

### Performance deep dive and feature influence

The row normalized confusion matrix shows the trade off created by the balanced training set:

| | Predicted ≤50K | Predicted >50K |
|---|---|---|
| **Actual ≤50K** | 79.6% | 20.4% |
| **Actual >50K** | 15.5% | 84.5% |

Recall on high earners is strong at 84.5 percent, at the cost of a 20.4 percent false positive rate. That balance can be adjusted through threshold tuning against a specific business cost function.

The largest coefficients align with economic intuition: marital status married civ spouse (+2.21), capital gain (+0.87), and executive or managerial occupation (+0.79) push predictions up, while private household service (−1.61) and farming or fishing (−1.27) push them down. Several native country coefficients carry large magnitudes but rest on very few observations, so I treat them as unstable and as a reminder of the representation issue noted above.

## Advanced modeling: neural network exploration

Built with Keras: input (77 features) to Dense(32, ReLU) to Dense(16, ReLU) to Dense(1, sigmoid), trained with SGD (learning rate 0.01) and binary cross entropy over 50 epochs with a 20 percent validation split, in about 55 seconds. Validation loss plateaued between epochs 15 and 30 with a slight upward drift afterward, indicating mild overfitting and diminishing returns rather than meaningful additional lift.

## Recommendation and business implications

| Metric | Logistic Regression (tuned) | Neural Network |
|---|---|---|
| Accuracy | 0.807 | 0.813 |
| F1 Score | 0.679 | 0.687 |
| AUC | 0.905 | 0.906 |

The neural network edged out logistic regression by roughly 0.6 percentage points in accuracy, based on a single run, while the logistic regression results were shown stable across three splits. AUC tells a tighter story: it measures how well a model ranks high earners above low earners across every possible decision threshold rather than just the one fixed cutoff accuracy and F1 depend on, and here it lands within 0.001 for both models. That gap is an order of magnitude smaller than the accuracy or F1 gap, meaning the two models discriminate between income groups almost equally well; the neural network's edge is mostly about where it happens to draw its decision line, not a genuinely better read of the data. Given that, and the regulatory requirement for explainability in financial services, logistic regression is the recommended production model. Revisit deep architectures only if the feature space expands significantly or a specific performance threshold requires it. The broader takeaway: model complexity does not inherently equate to business value.

This also functions as an informal check against memorization: the cross validated AUC used to pick the logistic regression's hyperparameters (0.904) and its held out test AUC (0.905) land within 0.001 of each other. If the model had simply memorized the training data, those two numbers would typically pull apart rather than agree this closely.

## Key learnings and future directions

The work that demanded the most judgment was not the modeling, it was the EDA and feature triage: choosing visualizations across 14 mixed type features, defining defensible thresholds for dropping correlated variables, and addressing historical data skew. Given more time I would run per subgroup error analysis across sex and race for fairness auditing, tune the decision threshold against an explicit cost function, assess calibration curves, and test gradient boosting as a middle ground between the linear model and the neural network.

On tooling, I leveraged Claude to accelerate implementation and debugging while retaining full ownership of the strategic decisions, data transformations, and final model selection. Every step was reviewed for business relevance and technical soundness.

---

**Sal Fakoya** · [GitHub](https://github.com/Sal-Fakoya) · [LinkedIn](https://www.linkedin.com/in/salamot-fakoya-650325224/)