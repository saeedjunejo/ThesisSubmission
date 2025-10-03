# ThesisSubmission
<img width="597" height="607" alt="Screenshot 2025-10-03 at 14 42 21" src="https://github.com/user-attachments/assets/d4395865-4932-4b21-94e8-b743dd7c536d" />
<img width="603" height="655" alt="Screenshot 2025-10-03 at 14 42 43" src="https://github.com/user-attachments/assets/fafd4acc-2a59-4022-b176-ec979e10f322" />

Code Segment counts 
 True Positive (TP)=608, 
 False Positive (FP)=172, 
 True Negative (TN)=363, 
 False Negative (FN)=119

AI Model (Logistic Regression) Results
Classification Report:
                 precision    recall  f1-score   support

      Safe (0)       1.00      0.93      0.96       111
Vulnerable (1)       0.95      1.00      0.97       142

      accuracy                           0.97       253
     macro avg       0.97      0.96      0.97       253
  weighted avg       0.97      0.97      0.97       253

Training + Inference Runtime: 0.0245 sec for 253 test samples

Time Complexity Analysis
Theoretical: O(m × n), where m = number of samples, n = average snippet length.
Empirical runtime: 0.0194 seconds for 1262 samples.
Average time per sample: 0.000015 seconds.
