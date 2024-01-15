import re
import string
import tensorflow as tf
from transformers import AutoTokenizer, TFAutoModel
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import euclidean
from scipy.spatial.distance import cityblock
from scipy.stats import pearsonr
import numpy as np

class BERTCorrection:

    def load_model(self, model):
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.model = TFAutoModel.from_pretrained(model)

    def __cleaning(self, text: str):
        # clear punctuations
        text = text.translate(str.maketrans('', '', string.punctuation))

        # clear multiple spaces
        text = re.sub(r'/s+', ' ', text).strip()

        return text

    def __process(self, first_token: str, second_token: str):
        inputs = self.tokenizer([first_token, second_token],
                                max_length=self.max_length,
                                truncation=self.truncation,
                                padding=self.padding,
                                return_tensors='tf')

        attention = tf.cast(inputs['attention_mask'], tf.float32)

        outputs = self.model(**inputs)

        # get the weights from the last layer as embeddings
        embeddings = outputs[0]  # when used in older transformers version
        # embeddings = outputs.last_hidden_state # when used in newer one

        # add more dimension then expand tensor
        # to match embeddings shape by duplicating its values by rows
        mask = tf.expand_dims(attention, -1)
        masked_embeddings = embeddings * mask

        # MEAN POOLING FOR 2ND DIMENSION
        # first, get sums by 2nd dimension
        # second, get counts of 2nd dimension
        # third, calculate the mean, i.e. sums/counts
        summed = tf.reduce_sum(masked_embeddings, axis=1)
        counts = tf.reduce_sum(attention, axis=1, keepdims=True)
        counts = tf.maximum(counts, 1e-9)  # Ensure counts are not zero
        mean_pooled = summed / counts

        # return mean pooling as numpy array
        return mean_pooled.numpy()

    def predict(self, first_token: str, second_token: str,
                return_as_embeddings: bool = False, max_length: int = 64,
                truncation: bool = True, padding: str = "max_length"):
        self.max_length = max_length
        self.truncation = truncation
        self.padding = padding

        first_token = self.__cleaning(first_token)
        second_token = self.__cleaning(second_token)

        mean_pooled_arr = self.__process(first_token, second_token)

        if return_as_embeddings:
            return mean_pooled_arr

        # calculate similarity
        cosineSimilarity = cosine_similarity([mean_pooled_arr[0]], [mean_pooled_arr[1]])
        # print('cosine similiarity',cosineSimilarity)
        # euclidian = 1 / (1 + euclidean(mean_pooled_arr[0], mean_pooled_arr[1]))
        # print('euclidian',euclidian)
        # manhattan = 1 / (1 + cityblock(mean_pooled_arr[0], mean_pooled_arr[1]))
        # print('manhattan',manhattan)
        # pearsonSimilarity, _ = pearsonr(mean_pooled_arr[0], mean_pooled_arr[1])
        # print('pearson',pearsonSimilarity)

        return cosineSimilarity
    
    def processUniqueness(self,token):
        lenToken = len(token.split())

        tokenTable = {}

        for t in token.split():
            if t in tokenTable.keys():
                tokenTable[t] += 1
            else:
                tokenTable[t] = 1

        lenTable = len(tokenTable.keys())

        aggregation = (lenToken + lenTable) // 2

        return aggregation / (10 * len(str(aggregation)))

