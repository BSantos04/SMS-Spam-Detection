import pandas as pd
import os
import argparse
import chardet as cd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score
from sklearn.feature_extraction.text import TfidfVectorizer

def load_dataset(infile, verbose):
    """
    Summary:
        Load the SPAM dataset and convert it into a Pandas dataframe.
    
    Parameters:
        infile: Path to the .csv file containing the dataset.
    
    Returns:
        df: Pandas dataframe of the SPAM dataset.
    """
    try:
        # Open the .csv file and detect the type of encoding of the dataset
        with open(infile, "rb") as file:
            rawdata = file.read()
            result = cd.detect(rawdata)
            encoding = result["encoding"]

        # Convert the .csv file into a Pandas dataframe specifying the detected encoding
        df = pd.read_csv(infile, encoding=encoding)

        # Display the first 5 rows of the raw dataframe
        if verbose:
            print("-"*211)
            print("Give a glimpse of the raw dataset:\n")
            print(df.head().to_string(index=False))

        return df
    except Exception as e:
        raise e

def pre_processing(df, verbose, label="v1", text="v2"):
    """
    Summary:
        Pre-process the previous Pandas dataframe: clean and normalize data conform it might be needed.
    
    Parameters:
        df: Pandas dataframe containing the data to be analyzed.
    
    Returns:
        df: Pre-process Pandas dataframe.
    """
    try:
        # Check if the labl and text columns are actually in the dataset
        if label not in df.columns or text not in df.columns:
            raise ValueError(f"Dataset does not contain at least one of the required columns: {label}, {text}")
        
        # Restrain the dataset to only the first 2 columns, since the other are just filling columns and those are the only that have any relevant data 
        df = df.loc[:, [label, text]].copy()

        # Remove duplicate row
        df.drop_duplicates(inplace=True)

        # Since it's crucial that a row has data on both columns, I chose to remove any row with NaN values
        df.dropna(axis=0, how="any", inplace=True)

        # Rename the columns for a more intuitive work from now on
        df.rename(columns={label: "SPAM/HAM", text: "SMS"}, inplace=True)

        # Remove additional empty spaces and put the text of every column content to lowercase
        df["SPAM/HAM"] = df["SPAM/HAM"].str.lower().str.strip()
        df["SMS"] = df["SMS"].str.lower().str.strip()

        # Ordinal codification for 'SPAM/HAM' column
        ord_spam = {"spam": 0, "ham": 1}
        df["SPAM/HAM"] = df["SPAM/HAM"].map(ord_spam)

        # Display the first 5 rows of the now pre-processed dataset
        if verbose:
            print("-"*211)
            print("Give a glimpse of the pre-processed dataset:\n")
            print(df.head().to_string(index=False))

        return df
    except Exception as e:
        raise e

def logistic_regression(df, sms, verbose):
    """
    Summary:
        Train and test a logistic regression model using the pre-processed dataframe and applies the now trained model to a new SMS to check if it's SMAP or not.

    Parameters:
        df: Pre-processed Pandas dataframe.
        sms: SMS message that's going to be verified by the model.
    """
    try:
        # Define the dependent and idenpendent variables (being X the independent variable and y the dependent variable)
        X = df["SMS"]
        y = df["SPAM/HAM"]

        # Split the data into training data and test data, setting the test data as 30% of the dataset and training data 70%, with a random state of 42
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        # Vectorize SMS data to be used in the logistic regression model
        vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_df=0.9, stop_words=None, min_df=2)
        X_train_vec = vectorizer.fit_transform(X_train)
        X_test_vec = vectorizer.transform(X_test)

        # Get an instance of the model, and train and test it with the selected data
        model = LogisticRegression(class_weight="balanced", max_iter=1000)
        model.fit(X_train_vec, y_train)
        
        # Get the predictions and probability based on the test datasets
        y_pred = model.predict(X_test_vec)
        y_proba = model.predict_proba(X_test_vec)[:, 1]

        # Evaluate the model using methods such as Confusion Matrix, Rating Report (focusing on F1-Score) and AUC-ROC Score
        conf_matrix = confusion_matrix(y_test, y_pred)
        class_report = classification_report(y_test, y_pred)
        auc_roc_score = roc_auc_score(y_test, y_proba)

        # Display the results of the evaluations
        if verbose:
            print("-"*211)
            print("Confusion Matrix:\n", conf_matrix)
            print("\n\nClassification Report:\n", class_report)
            print(f"\n\nAUC-ROC Score: {auc_roc_score}")

        # Create a new Pandas dataframe with the input message
        new_data = pd.DataFrame({"SMS": [sms]})

        # Vectorize the message it will be used in the logistic regression model
        new_data_vec = vectorizer.transform(new_data["SMS"])

        # Predict if the message is SPAM or not
        pred = model.predict(new_data_vec)

        # Display the results of the prediction
        if verbose:
            print("-"*211)
            print(f"The message: {sms.strip().lower()}")
            print(f"\nThe verdict: {'SPAM' if pred==0 else 'HAM'}!!!")
            print("-"*211)
        else:
            print(f"The message: {sms.strip().lower()}")
            print(f"\nThe verdict: {'SPAM' if pred==0 else 'HAM'}!!!")
    except Exception as e:
        raise e

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("sms", type=str, help="Message to be analyzed (ex.: 'Hello World!!!').")
    parser.add_argument("--dataset", type=str, help="Name of the CSV dataset to be used as training (make sure to include it into the datasets folder).", required=False)
    parser.add_argument("--label", type=str, help="Name of the column containing SPAM/HAM labeling")
    parser.add_argument("--text", type=str, help="Name of the column containing the SMSs")
    parser.add_argument("--verbose", action="store_true", help="Will give you a more informative output.")
    args = parser.parse_args()

    # Make sure the label and SMSs column names are specified when giving a different dataset for training
    if args.dataset and not args.label and not args.text:
        parser.error("Could you give me the name of the label and SMSs columns?\nUse the --label and --text flags")

    # Get the path of the training dataset (you better not touch the path of the file)
    if args.dataset:
        dataset = os.path.abspath(f"datasets/{args.dataset}")
    else:
        dataset = os.path.abspath("datasets/spam.csv")

    # Load the raw dataset
    df = load_dataset(dataset, args.verbose)

    # Pre-process the dataset
    if  args.label and  args.text:
        df_cleaned = pre_processing(df, args.verbose, args.label, args.text)
    else:
        df_cleaned = pre_processing(df, args.verbose)

    # Apply logistic regression to check if the input message is SPAM or not
    logistic_regression(df_cleaned, args.sms, args.verbose)