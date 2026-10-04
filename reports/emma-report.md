# Income Insight: Individual Report
**Emma Rogoveanu — CST-435 Topic 2**

## Project and Personal Contribution

Income Insight is a deep-learning application that predicts whether a census record belongs to the annual income category above $50,000. The project uses the UCI Adult Income dataset, containing 48,842 records extracted from 1994 census data. Its three-cloud architecture connects a Streamlit interface to a FastAPI service on Render and a Supabase Postgres database. The application supports individual and CSV predictions, model evaluation, explanations of neural-network concepts, and a fairness audit.

My primary responsibilities included repository setup, database design and loading, preprocessing, configurable neural-network training, controlled experiments, calibration, evaluation, API serving, and prediction logging. I also coordinated the final interface corrections, updated technical documentation, verified the deployed product, and added an automated GitHub Actions test workflow. I used AI assistance during implementation and review, with responsibility for checking the resulting behavior and documenting that assistance. Komal Khan contributed the initial six-tab Streamlit frontend and associated frontend work.

## Model Design and Activation Choice

The dataset was divided into fixed training, validation, and test splits of 34,188, 7,327, and 7,327 records. The split was stratified by income and sex. Numeric features receive median imputation and standard scaling, while categorical features receive missing-value replacement and one-hot encoding. Preprocessing is fitted only on training data to prevent information leakage. Ten original features become 84 model inputs.

The selected PyTorch multilayer perceptron has two hidden layers, with 64 and 32 units, followed by one output logit. It uses GELU activations, dropout of 0.1, AdamW optimization, and binary cross-entropy loss. Training retains the checkpoint with the lowest validation loss.

Three configurations were compared using the same split, seed, learning rate, batch size, weight decay, epoch budget, and early-stopping policy. The baseline used ReLU with hidden sizes [64, 32]. A deeper ReLU model used [128, 64, 32]. The GELU model retained the baseline architecture and changed only the activation.

GELU achieved the lowest validation loss: 0.3156, compared with 0.3158 for the deeper model and 0.3163 for the baseline. I therefore selected GELU under the established validation-loss criterion, although the baseline had slightly higher validation accuracy. GELU smoothly weights inputs rather than setting every negative input to zero. However, the small differences and single seed provide limited evidence of general superiority. My choice is justified for this experiment, rather than as a claim that GELU always outperforms ReLU.

## Performance and the Harder Class

The selected model achieved test accuracy of 0.8540 and ROC-AUC of 0.9060. For the >50K class, precision was 0.7322, recall was 0.6144, and F1 was 0.6681. The confusion matrix contained 5,180 true negatives, 394 false positives, 676 false negatives, and 1,077 true positives.

The >50K category was the harder class. The model missed 676 of the 1,753 actual positive records, producing a false-negative rate of approximately 38.6%. In comparison, recall for the ≤50K class was 0.9293. The minority status of >50K records helps explain why overall accuracy can appear strong while positive-class recall remains substantially lower.

Temperature scaling was fitted on validation data, producing T ≈ 1.0238. Test expected calibration error changed from approximately 0.0100 to 0.0099. This was a small adjustment because the original probabilities were already reasonably calibrated overall. Calibration does not guarantee accuracy for an individual or equal calibration across demographic groups.

## Feature Importance and Fairness

Permutation importance measured the decrease in test ROC-AUC after shuffling each original feature. Marital status had the largest mean decrease, approximately 0.0549, followed by capital gain at 0.0370 and age at 0.0335. These results identify features the model relies on; they do not establish that those features cause income differences.

The fairness audit uses SQL counts from labeled test predictions. Female records had an FPR of 0.0268 and FNR of 0.4151, while Male records had an FPR of 0.0985 and FNR of 0.3804. Thus, women earning >50K were missed more frequently, while men earning ≤50K were more frequently classified above the threshold.

Excluding sex and race from model inputs did not eliminate disparities. Relationship and marital status can act as proxies, and group income base rates differ. Potential mitigations include reweighting training examples, examining proxy features, and evaluating fairness-aware objectives. These approaches have not been implemented and would require validation-based tuning and careful evaluation. Repeated experiments and confidence intervals would also strengthen the evidence.

## Christian Worldview and Responsible Use

Deuteronomy 1:17 calls for impartial judgment and equal attention to people regardless of status. Applied to this project, that principle challenges me to examine unequal errors instead of relying only on aggregate accuracy. Each record represents a person whose dignity cannot be reduced to a predicted category.

Responsible stewardship also requires honesty about limitations. This model reflects historical data, an outdated income threshold, and observed disparities. I would not use it to determine employment, credit, housing, or benefits. Its appropriate purpose is education: demonstrating how a functioning machine-learning product can be evaluated transparently while acknowledging the human consequences of its errors.