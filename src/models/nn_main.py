import keras


class NN_main:

    def metrics(self, key: str) -> keras.metrics:
        metrics_dict = {
            "true_positive": keras.metrics.TruePositives(name='tp'),
            "false_positive": keras.metrics.FalsePositives(name='fp'),
            "true_positive": keras.metrics.TrueNegatives(name='tn'),
            "false_negative": keras.metrics.FalseNegatives(name='fn'),
            "accuracy": keras.metrics.BinaryAccuracy(name='accuracy'),
            "precision": keras.metrics.Precision(name='precision'),
            "recall": keras.metrics.Recall(name='recall'),
            "roc_auc": keras.metrics.AUC(name='auc')
        }
        return metrics_dict.get(key)
