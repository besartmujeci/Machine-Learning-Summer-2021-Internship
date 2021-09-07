import streamlit as st
import tarfile
from simpletransformers.classification import ClassificationModel
import recommender as rc


def unpack_model(model_name=''):
    tar = tarfile.open(f"{model_name}.tar.gz", "r:gz")
    tar.extractall()
    tar.close()


unpack_model('tapas-categ-class')

train_args = {"reprocess_input_data": True,
              "overwrite_output_dir": True,
              "fp16": False,
              "num_train_epochs": 4}

model = ClassificationModel(
    "roberta", "roberta-base",
    num_labels=9,
    args=train_args,
    use_cuda=False
)


class_list = ["Art | Comics",
              "Collaborations",
              "Events | Challenges",
              "Off Topic",
              "Promotions",
              "Questions",
              "Reviews | Feedback",
              "Tech Support | Site Feedback",
              "Writing | Novels"]


def predict(post):
    placeholder = st.empty()
    placeholder.markdown("![Alt Text](https://media2.giphy.com/"
                         "media/tXL4FHPSnVJ0A/source.gif)")
    predictions, raw_outputs = model.predict([rc.Recommender.
                                             clean_new_text(post)])
    placeholder.empty()
    return class_list[predictions[0]]