# 📺 Sentiment Analysis of Urdu Drama Transcripts 🇵🇰

This project aims to perform **sentiment analysis** on Urdu drama transcripts collected from YouTube. Due to the lack of labeled Urdu sentiment datasets, we used **translation** and **transfer learning** techniques to generate a high-quality labeled dataset and trained a machine learning model to classify sentiments as **positive** or **negative**.

---

## 📌 Project Highlights

- 🗂 Collected 500+ English-Urdu drama transcripts from YouTube
- 🌍 Translated content using `facebook/m2m100_418M` multilingual model
- 💬 Labeled English lines using `distilbert-base-uncased-finetuned-sst-2-english`
- 🔄 Mapped sentiment labels to corresponding Urdu translations
- 🧹 Preprocessed Urdu text: normalization, stopword removal, stemming, tokenization
- 🧠 Trained a **Logistic Regression** model on a **balanced dataset of 40,000 samples (20K positive + 20K negative)**
- 📈 Achieved **75% accuracy** on test data

---

## 🧠 Methodology

1. **Data Collection**  
   - Gathered English subtitle files from YouTube dramas  
   - Used YouTube auto-translations and replaced them with high-quality multilingual translations using `facebook/m2m100_418M`

2. **Sentiment Labeling**  
   - Applied a pre-trained English sentiment classifier (`distilbert-sst2`)  
   - Mapped labels to Urdu translations via aligned sentences

3. **Preprocessing Urdu Text**  
   - Normalization (removal of diacritics, character unification)  
   - Stopword removal (custom Urdu list)  
   - Tokenization & stemming

4. **Model Training**  
   - Used `TF-IDF` vectorization  
   - Trained a **Logistic Regression** model  
   - Balanced the dataset: 20K positive + 20K negative samples

---

## 📊 Results

| Metric        | Value |
|---------------|-------|
| Accuracy       | 75.0% |
| Precision      | 75.0% |
| Recall         | 75.0% |
| F1 Score       | 75.0% |

The model performed well on both sentiment classes, slightly better at recognizing **negative** sentiments.

---

## 💻 How to Run

```bash
# 1. Clone the repository
git clone https://github.com/zohaibsaeed117/urdu-drama-sentiment-analysis.git
cd urdu-drama-sentiment-analysis

# 2. Create a virtual environment and activate it
python -m venv venv
source venv/bin/activate      # On Windows: venv\Scripts\activate

# 3. Install required packages
pip install -r requirements.txt

# 4. Launch Jupyter Notebook
jupyter notebook
````

---

## 📚 Dependencies

* Python 3.8+
* scikit-learn
* pandas
* nltk
* transformers
* sentencepiece
* tqdm

(Full list in `requirements.txt`)

---

## 📂 Dataset

There are 2 dataset files.

* `Master-Labeled-Bert-Model.xlsx`: Master Urdu Labeled dataset with almost 160000 instances
* `Balanced-Labeled-Bert-Model.xlsx`: 40,000 labeled Urdu sentences (balanced)

Each row contains:

* `Urdu Sentence`: The sentence
* `Sentiment`: `Positive` or `Negative`

---

## 🤝 Contributing

Contributions are welcome! Feel free to fork this repo, open issues, or submit pull requests. Suggestions for:

* Better Urdu tokenization
* Improved translation strategies
* Use of contextual models like BERT for Urdu

are especially welcome.

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

---

## 👤 Author

**Zohaib Saeed**
BS Computer Science | UET Lahore
GitHub: [@zohaibsaeed117](https://github.com/zohaibsaeed117)
LinkedIn: [Zohaib Saeed](https://linkedin.com/in/zohaibsaeed117)

---

## 🌟 Acknowledgements

* Instructor: **Maam Qurat-ul-Ain** (NLP Course - UET)
* Hugging Face 🤗 for amazing models
* YouTube channels providing rich Urdu drama content
