## Notes

The data describes whether an audio clip contains anomalies or not. Hence, I framed the project as an audio classification problem. I considered four scenarios:
1. Normal vs abnormal: A binary classification problem.
2. Wheeze vs crackle vs both: A multi-label classification problem.
3. None vs weeze vs crackle vs both: A multi-label classification problem.
4. Diagnosis: A multi-class classification problem.

I decided to focus on scenario 3. It combines both tasks 1 and 2 and can provide some explainability insights, which can help address task 4 in the future. I trained a single model to predict two binary labels: the presence or absence of crackles and wheezes. I could have used hierarchical models (one for normal vs abnormal and then another for wheeze vs crackle vs both) or separate binary classifiers for crackles and wheezes. But I chose a multi-label solution because it often works better, as shown in other domains like [voice type classification](https://arxiv.org/abs/2005.12656).

---

1. As an exploration, I analyzed the dataset using scripts `s00_audio_summary.py` and `s01_plot_spectrogram.ipynb`. Among other things, I notices that: 
   - The sampling rate of recordings varies from 4 kHz to 44.1 kHz.  
   - Some samples (especially 100% of Litt3200 device samples) had blank regions above the 2000 Hz frequency range.
   - The dataset contains 920 recordings from 126 patients, totaling 5.5 hours. Each breathing cycle is labeled by an expert into one of four classes: normal, crackle, wheeze, or both (crackle and wheeze). Cycle durations vary from 0.2s to 16.2s (average: 2.7s). The dataset is imbalanced, with 3642 normal, 1864 crackle, 886 wheeze, and 506 both-class cycles.
2. Preprocessing: 
   - I re-sampled all recordings to 16 kHz (which is the sampling rate of the encoder I used in my model).  
   - I applied a 4th-order Butterworth low-pass filter to reduce the impact of blank regions above 2000 Hz. (Notably: this preprocessing improved the model’s performance during exploratory tests).
   - I segmented audio clips based on the ground truth in the metadata. Each segment corresponds to one respiration cycle.
3. Train-dev-test splitting:
  - I performed a train-dev-test split (Train: 60%, Dev: 20%, Test: 20%) while stratifying by chest location to ensure proportional data distribution. Notably, I considered other stratifications but skipped them due to the limited data size and time constraints.
  - Each participant appears in only one subset to prevent the model from recognizing participants instead of detecting anomalies.  
4. Model training:
  - I used an AST encoder (`MIT/ast-finetuned-audioset-14-14-0.443`) with this simple head (mean pooling, relu, dropout, and linear classifier).
  - I froze the encoder completely. Partial freezing caused overfitting, probably due to the small dataset and limited variability.
  - I evaluated AUDIOMAE as an alternative encoder but saw no significant performance difference, so I kept the AST encoder for simplicity. AST processes mel-spectrograms, which mimic the human auditory system (logarithmic frequency bands). Since this is not proper speech (no words), I wanted a model focusing on spectral variations rather than language features.
  - As loss function, I used `nn.BCEWithLogitsLoss()`, which is good for multi-label classification because it handles independent binary predictions for each label.
  - I applied different types of data augmentation (both showed some improvements in the performance and robustness of the model):
    -  Signal processing-based: gain adjustment and time shifting.
    -  Concatenation-based: I concatenated cycles to increase data for positive wheeze labels.
  -  I didn’t optimize hyperparameters (e.g., learning rate) due to time constraints. I would use Optuna to minimize validation loss in future experiments.
  -  I used early stopping to avoid overfitting and tracked training loss with W&B.

<img width="1631" alt="Screenshot 2025-01-27 at 9 13 45 AM" src="https://github.com/user-attachments/assets/0507df5a-b793-4c56-a424-fdb555a8a8b2" />

5. Evaluation
  - I measured performance using F1-score, precision, recall, area under the curve (AUC), and evaluated each label separately.
  - I also aggregated results using macro and micro averages.
  - The best model is available [here](https://huggingface.co/fabiocat/icbhi_classification/).
  - You can use `s05_inference.py` for inference. The script outputs scores for crackle and wheeze labels.

---

## Performance:

```json
{
  "label_specific": {
    "label_Crackle": {
      "f1_score": 0.6756,
      "precision": 0.6147,
      "recall": 0.75,
      "roc_auc": 0.7033
    },
    "label_Wheeze": {
      "f1_score": 0.4853,
      "precision": 0.6565,
      "recall": 0.3849,
      "roc_auc": 0.8031
    }
  },
  "aggregated": {
    "f1_macro": 0.5805,
    "precision_macro": 0.6356,
    "recall_macro": 0.5674,
    "f1_micro": 0.6282,
    "precision_micro": 0.6223,
    "recall_micro": 0.6342,
    "auc_macro": 0.7532,
    "auc_micro": 0.7875
  }
}
```

![confusion_matrices](https://github.com/user-attachments/assets/5c887d98-39df-47cc-8d6a-1439ea0d6149)

---

## Future work
- Benchmarking different architectures: Exploring other pre-trained encoders (including multimodal models like CLIP) could help identify a more optimal model. 
- Exploring different augmentations: More comprehensive augmentation strategies, such as frequency masking, may further improve robustness.
- Error analysis: Analyzing the confusion matrices could reveal patterns in misclassification (for instance, are most false positives associated with specific types of background noise - which may depend on the location of the microphone?)
- Hyperparams optimization: Leveraging tools like Optuna for systematic hyperparameter tuning could refine training parameters, including learning rate, batch size, and dropout rates.
- Expanding evaluation metrics: Evaluating temporal consistency of predictions across cycles (e.g., a moving average of predictions).
- Exploring contextual info: Combining consecutive cycles or analyzing the temporal evolution of crackles and wheezes.
- ... (this list can be endless)
