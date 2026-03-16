# ADHD-Detection
A Streamlit-based ADHD assessment platform that integrates EEG brainwave monitoring, behavioral questionnaires, facial emotion detection, and cognitive activity analysis to provide a comprehensive ADHD evaluation dashboard.
The system captures live EEG signals via serial port, analyzes behavioral responses, monitors facial expressions during assessments, and generates patient progress reports and ADHD risk interpretations.

This platform is designed for research, academic projects, and cognitive assessment experiments.

Key Features
1. Live EEG Brainwave Dashboard

Real-time EEG signal monitoring

Serial device integration via COM6

Displays brainwave signals:

Attention

Meditation

Delta

Theta

Alpha

Beta

Gamma

Live interactive charts using Plotly

Signal quality indicator

2. ADHD Questionnaire Assessment

Structured behavioral screening questionnaire

Likert-scale responses

ADHD risk scoring

Behavioral interpretation

3. Facial Emotion Monitoring

Real-time webcam emotion detection

Uses OpenCV + DeepFace

Detects emotions:

Happy

Neutral

Sad

Angry

Fear

Surprise

Used to measure engagement and distraction during testing.

4. Cognitive Activity Builder

Interactive tasks to evaluate cognitive performance:

Memory sequence test

Attention tracking

Reaction time tests

Pattern recognition

Performance metrics recorded:

Accuracy

Completion time

Error rate

5. Patient Progress Dashboard

Tracks historical patient data including:

EEG signal trends

Questionnaire results

Emotion patterns

Cognitive task performance

6. Admin Dashboard

Admin can:

View all patient assessments

Manage patient records

Monitor system usage

Export reports

7. ADHD Risk Interpretation

The system combines:

EEG brainwave analysis

Questionnaire results

Emotion monitoring

Cognitive activity performance

To generate a cumulative ADHD risk report.

Risk levels:

Low ADHD Risk

Moderate ADHD Risk

High ADHD Risk
