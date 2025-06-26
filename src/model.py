
from transformers import BertForSequenceClassification

def get_model(model_name="bert-base-uncased", num_labels=5):
    """
    Initializes a BERT model for multi-label emotion classification.

    Args:
        model_name (str): The name of the pre-trained model from Hugging Face.
                          Default is 'bert-base-uncased'.
        num_labels (int): The number of emotion labels we want to predict.
                          For this project, we are using 5 emotions.

    Returns:
        model (BertForSequenceClassification): A model with a classification
                                               head suitable for multi-label tasks.
    """

    # Load a pre-trained BERT model and attach a classification head to it.
    # We're using Hugging Face's 'BertForSequenceClassification' which is designed for classification tasks.

    model = BertForSequenceClassification.from_pretrained(
        model_name,               # Base transformer model
        num_labels=num_labels,    # Number of output neurons (one per emotion)

        # This is the key setting: we specify that it's a multi-label task.
        # This tells the model to use sigmoid (not softmax) during inference.
        problem_type="multi_label_classification"
    )

    return model


if __name__ == "__main__":
    model = get_model()
    print(model.config)
