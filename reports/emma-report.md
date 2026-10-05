# Income Insight: Individual Report
**Emma Rogoveanu — CST-435 Topic 2**

## Project and Personal Contribution

Income Insight predicts whether a census record earns above $50,000 a year, using the 48,842-record UCI Adult Income dataset from the 1994 census. A Streamlit interface calls a FastAPI service on Render, which persists to Supabase Postgres. The application supports individual and CSV predictions, model evaluation, neural-network concept explanations, and a fairness audit.

My primary responsibilities included repository setup, database design and loading, preprocessing, configurable neural-network training, controlled experiments, calibration, evaluation, API serving, and prediction logging. I also coordinated the final interface corrections, updated technical documentation, verified the deployed product, and added an automated GitHub Actions test workflow. I used AI assistance throughout, and I was responsible for checking and documenting it. Komal Khan contributed the initial six-tab Streamlit frontend and associated frontend work.

## Model Design and Activation Choice

The dataset was divided into fixed training, validation, and test splits of 34,188, 7,327, and 7,327 records. The split was stratified by income and sex. Numeric features are median-imputed and standardized; categorical features are one-hot encoded. Preprocessing is fitted on training data only, preventing leakage. Ten original features become 84 model inputs.

The selected PyTorch multilayer perceptron has two hidden layers, with 64 and 32 units, followed by one output logit. It uses GELU activations, dropout of 0.1, AdamW optimization, and binary cross-entropy loss. Training retains the checkpoint with the lowest validation loss.

Three configurations were compared using the same split, seed, learning rate, batch size, weight decay, epoch budget, and early-stopping policy. They were a ReLU baseline [64, 32], a deeper ReLU model [128, 64, 32], and a GELU model that changed only the baseline's activation.

GELU achieved the lowest validation loss: 0.3156, compared with 0.3158 for the deeper model and 0.3163 for the baseline. I therefore selected GELU under the established validation-loss criterion, although the baseline had slightly higher validation accuracy. GELU weights negative inputs smoothly instead of zeroing them. The differences are small and come from one seed, so they justify the choice for this experiment, not a claim that GELU generally outperforms ReLU.

## Performance and the Harder Class

The selected model achieved test accuracy of 0.8540 and ROC-AUC of 0.9060. For the >50K class, precision was 0.7322, recall was 0.6144, and F1 was 0.6681. The confusion matrix contained 5,180 true negatives, 394 false positives, 676 false negatives, and 1,077 true positives.

The >50K category was the harder class. The model missed 676 of the 1,753 actual positive records, producing a false-negative rate of approximately 38.6%. In comparison, recall for the ≤50K class was 0.9293. Only 23.9% of test records are >50K, and the loss weighted every record equally. This imbalance is consistent with, though it does not prove, the lower positive-class recall behind a strong overall accuracy.

Temperature scaling was fitted on validation data, producing T ≈ 1.0238. Test expected calibration error changed from approximately 0.0100 to 0.0099. The original probabilities were already well calibrated overall. Calibration does not guarantee accuracy for an individual or equal calibration across demographic groups.

## Feature Importance and Fairness

Permutation importance measured the decrease in test ROC-AUC after shuffling each original feature. Marital status had the largest mean decrease, approximately 0.0549, followed by capital gain at 0.0370 and age at 0.0335. These results show what the model relies on, not what causes income.

The fairness audit uses SQL counts from labeled test predictions. Female records have the worse false-negative rate: 0.4151, compared with 0.3804 for Male records. Male records have the worse false-positive rate: 0.0985, compared with 0.0268. Neither group fares worse on every metric. Women earning >50K were missed more often, while men earning ≤50K were more often placed above the threshold.

Consider an illustrative workforce-services screen, which I am not recommending, that uses a >50K prediction to shortlist people for higher-salary job referrals. A false negative there means a qualified person is never referred. At the audited rates, about 42 of every 100 high-earning women would be passed over, compared with 38 of every 100 such men. A false positive means a man earning ≤50K is referred to roles that may not match his circumstances, which also spends referrals that others needed. Overall accuracy would hide both harms.

Excluding sex and race from model inputs did not eliminate disparities. Relationship and marital status can act as proxies, and base rates differ (10.9% of Female versus 30.4% of Male test records earn >50K). Two mitigations I would test first are reweighting each sex-and-income combination during training, and group-specific thresholds chosen on validation data to narrow the false-negative gap. A third option is retraining without relationship. None of these is implemented. Responsible evaluation would tune each mitigation on validation data only, then check it on the test split once. It would report FPR and FNR with confidence intervals across several seeds, and confirm that accuracy and calibration did not degrade unacceptably.

## Christian Worldview and Responsible Use

Deuteronomy 1:17 commands, "You shall not be partial in judgment. You shall hear the small and the great alike." Good stewardship of a model means taking responsibility for its disparate impact instead of hiding behind aggregate accuracy. Before any deployment, the group harmed by an error is owed four things:

- transparent disclosure of the measured gap
- investigation of why it occurs
- validation that a mitigation actually narrows it without creating new harm
- safeguards so that no automated prediction alone decides a person's opportunity

In the illustrative referral screen, women whose qualifications the model tends to miss should not carry the cost of its blind spot. That would require human review and a way to appeal.

Responsible stewardship also requires honesty about limitations. This model reflects historical data, an outdated income threshold, and observed disparities. I would not use it to determine employment, credit, housing, or benefits. Its appropriate purpose is education: demonstrating how a functioning machine-learning product can be evaluated transparently while acknowledging the human consequences of its errors.