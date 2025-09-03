Overall framework:

- raw gait data (16 features -> 16 sensor channels across the time domain)
- 1DCNN model -> predict the severity score of the patient ("Healthy","Stage 2","Stage 2.5","Stage 3")
- XAI methods: LRP and GradCAM for time series signal -> compute the relevance score of each data point of each sensor
- Discrepancy: absolute difference
- Chatbox: template prompt (for system 'role'), Patient data e.g. prediction, gait metrics, XAI plots (for user 'role')
