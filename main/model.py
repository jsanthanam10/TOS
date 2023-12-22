import pandas as pd
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from transformers import BertTokenizer, BertModel
import torch
import numpy as np
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

class ToSDataset(Dataset):
        def __init__(self, texts, labels, tokenizer, max_len=512):
            self.texts = texts
            self.labels = labels
            self.tokenizer = tokenizer
            self.max_len = max_len

        def __len__(self):
            return len(self.texts)

        def __getitem__(self, item):
            text = str(self.texts[item])
            label = self.labels[item]

            encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
            truncation=True
            )

            return {
            'text': text,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
            }

def main():
    claudette = pd.read_csv('../combined.csv')
    ToSDR = pd.read_csv('../tosdr_data.csv')

    claudette = claudette[["clause", "severity"]]
    claudette = claudette[(claudette.severity == 1) | (claudette.severity == 2) | (claudette.severity == 3)]
    #rename columns
    claudette = claudette.rename(columns={"clause": "title", "severity": "classification"})
    ToSDR = ToSDR[["title", "classification"]]

    #in ToSDR, rename good, bad, and neutral to 1, 3, and 2 respectively drop the rest

    ToSDR["classification"] = ToSDR["classification"].replace("good", 1)
    ToSDR["classification"] = ToSDR["classification"].replace("bad", 3)
    ToSDR["classification"] = ToSDR["classification"].replace("neutral", 2)
    ToSDR = ToSDR[(ToSDR.classification == 1) | (ToSDR.classification == 2) | (ToSDR.classification == 3)]

    # concat both dataframes

    frames = [claudette, ToSDR]
    result = pd.concat(frames)
    result.reset_index(drop=True, inplace=True)

    X = result["title"]
    y = result["classification"]

    # make model to predict classification based on title

    

    # Convert labels to integers
    y = y.astype(int)

    # split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # create pipeline
    text_clf = Pipeline([('tfidf', TfidfVectorizer()),
                        ('clf', SGDClassifier(loss='hinge', penalty='l2',
                                            alpha=1e-3, random_state=42,
                                            max_iter=5, tol=None))])

    # train model
    text_clf.fit(X_train, y_train)

    # predict
    predictions = text_clf.predict(X_test)

    # evaluate
    print("Accuracy: ", accuracy_score(y_test, predictions))

    print(confusion_matrix(y_test, predictions))
    print(classification_report(y_test, predictions))

    # cross validation
    from sklearn.model_selection import cross_val_score

    scores = cross_val_score(text_clf, X, y, cv=5)
    print("Cross Validation Scores: ", scores)
    print("Mean Cross Validation Score: ", scores.mean())

    

    # Assuming X and y are your dataset's features and labels
    # X: List of text clauses, y: Corresponding labels (1, 2, 3)

    # Load a pre-trained BERT model and tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    bert_model = BertModel.from_pretrained('bert-base-uncased')

    
    def create_data_loader(df, tokenizer, max_len, batch_size):
        ds = ToSDataset(
            texts=df.X.to_numpy(),
            labels=df.y.to_numpy(),
            tokenizer=tokenizer,
            max_len=max_len
        )

        return DataLoader(
            ds,
            batch_size=batch_size,
            num_workers=4
        )

    # Data preparation
    df = pd.DataFrame({'X': X, 'y': y})
    train_df, test_df = train_test_split(df, test_size=0.2)

    BATCH_SIZE = 16
    MAX_LEN = 128
    train_data_loader = create_data_loader(train_df, tokenizer, MAX_LEN, BATCH_SIZE)
    test_data_loader = create_data_loader(test_df, tokenizer, MAX_LEN, BATCH_SIZE)

    # BERT feature extraction
    def extract_features(data_loader, model, device):
        model.eval()
        features = []
        labels = []

        with torch.no_grad():
            for d in data_loader:
                input_ids = d["input_ids"].to(device)
                attention_mask = d["attention_mask"].to(device)
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                features.extend(outputs.last_hidden_state[:, 0, :].detach().cpu().numpy())
                labels.extend(d['labels'].detach().cpu().numpy())

        return np.array(features), np.array(labels)


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bert_model.to(device)

    train_features, train_labels = extract_features(train_data_loader, bert_model, device)
    test_features, test_labels = extract_features(test_data_loader, bert_model, device)

    # Train a simple classifier on the extracted features
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    clf.fit(train_features, train_labels)

    # Predict and evaluate
    predictions = clf.predict(test_features)
    print("Accuracy: ", accuracy_score(test_labels, predictions))
    print(confusion_matrix(test_labels, predictions))
    print(classification_report(test_labels, predictions))
    
if __name__ == "__main__":
    main()