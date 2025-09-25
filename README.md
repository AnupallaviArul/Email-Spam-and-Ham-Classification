# Email Spam Classification Project

## Project Description

This project aims to build a machine learning model to classify emails as either "spam" or "ham" (not spam). Effective spam classification is important for improving user experience, protecting against security threats, and ensuring efficient resource utilization.

## Data

The dataset used in this project is an email dataset containing email text and a label indicating whether the email is spam or ham.

**Initial Data Inspection:**

*   The dataset contains 5572 emails.
*   Initial inspection revealed that the dataset has three columns ('Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4') with a large number of missing values, which were later handled by concatenating their content with the main text column ('v2') and dropping the original columns. There were no missing values in the 'v1' and 'v2' columns.

**Class Distribution:**

*   The dataset is imbalanced, with 4825 ham emails and 747 spam emails. This represents approximately 86.6% ham and 13.4% spam.

![Class Distribution Plot](https://github.com/AnupallaviArul/Email-Spam-and-Ham-Classification/blob/main/class_dist.png)

**Email Length Analysis:**

*   Spam emails tend to be longer than ham emails on average (142.15 characters for spam vs 74.63 characters for ham).

![Email Length Distribution Plots](email_length_distribution.png)

## Methodology

The project followed a standard machine learning workflow:

1.  **Data Loading and Inspection:** Loading the dataset and performing initial checks for data quality.
2.  **Exploratory Data Analysis (EDA):** Analyzing the class distribution, email length, and common words in spam and ham emails.
3.  **Text Preprocessing:** Cleaning the text data by converting to lowercase, removing punctuation, tokenizing, and stemming.
4.  **Feature Extraction:** Transforming the text data into numerical features using TF-IDF vectorization.
5.  **Model Training:** Training Multinomial Naive Bayes and Logistic Regression classifiers on the prepared data.
6.  **Model Evaluation:** Evaluating the trained models using metrics such as accuracy, precision, recall, F1-score, and confusion matrices.
7.  **Key Feature Analysis:** Identifying the words most indicative of spam and ham according to the models.
8.  **Error Analysis:** Examining examples of misclassified emails to understand model limitations.

## Results

Two models, Multinomial Naive Bayes and Logistic Regression, were trained and evaluated.

**Model Performance on Test Set:**

*   **Multinomial Naive Bayes:**
    *   Accuracy: 0.9713
    *   Precision (Spam): 0.9916
    *   Recall (Spam): 0.7919
    *   F1-score (Spam): 0.8806

*   **Logistic Regression:**
    *   Accuracy: 0.9695
    *   Precision (Spam): 0.9915
    *   Recall (Spam): 0.7785
    *   F1-score (Spam): 0.8722

Both models achieved high precision for spam detection, crucial for minimizing false positives. The Multinomial Naive Bayes model showed slightly better overall performance (F1-score) for the spam class.

**Confusion Matrices:**

![Multinomial Naive Bayes Confusion Matrix](mnb_confusion_matrix.png)

![Logistic Regression Confusion Matrix](lr_confusion_matrix.png)

**Key Feature Analysis:**

The models identified distinct words as indicators of spam and ham. According to the Multinomial Naive Bayes model, top spam indicators include 'claim', 'prize', 'won', while top ham indicators include 'ill', 'ltgt', 'my'.

![Top Spam/Ham Words Bar Charts](top_words_bar_chart.png)

## Visualizations

Here are some key visualizations from the project:

### Class Distribution

![Class Distribution Plot](class_distribution.png)

### Email Length Distribution

![Email Length Distribution Plots](email_length_distribution.png)

### Top Indicative Words (Multinomial Naive Bayes)

![Top Spam/Ham Words Bar Charts](top_words_bar_chart.png)

## Conclusion and Potential Improvements

Our analysis and modeling efforts for email spam classification have yielded promising results. Both Multinomial Naive Bayes and Logistic Regression models demonstrated high accuracy and very high precision in identifying spam emails, which is crucial for minimizing false positives in a real-world spam filter. The Multinomial Naive Bayes model was slightly better in terms of recall and F1-score for the spam class.

Given the high precision, these models would be highly useful where minimizing false positives is a top priority. While some spam might still get through (lower recall), the low rate of legitimate emails being incorrectly flagged makes these models suitable for many applications, with the acceptable trade-off depending on specific business needs.

To further enhance the performance, future work could involve exploring advanced text representation techniques (like N-grams or word embeddings), evaluating different classification algorithms (SVMs, deep learning models), systematic hyperparameter tuning, and incorporating strategies to handle evolving spam tactics through periodic retraining on new data. Additional feature engineering, such as analyzing URL patterns or email structure, could also be explored. Deploying such a model can lead to cleaner inboxes, improved security, and resource savings.
