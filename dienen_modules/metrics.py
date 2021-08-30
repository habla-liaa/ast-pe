import numpy as np
from IPython import embed
from sklearn.metrics import roc_auc_score, average_precision_score
import scipy.stats
from tensorflow.keras.losses import binary_crossentropy
import tensorflow as tf

class OutputAsMetric():
    def __init__(self, name='loss'):
        self.name = name
    def __call__(self,y_true,y_pred):
        return {'loss': np.mean(y_pred)}

class DPrime():
    def __init__(self, name='dprime', balanced=False):
        self.balanced = balanced

    def __call__(self, y_true, y_pred):
        aucs = roc_auc_score(y_true,y_pred,average=None)
        dprimes = np.sqrt(2)*scipy.stats.norm.ppf(aucs)

        return {'dprime': dprimes,
                'weighted_average_dprime': np.nanmean(dprimes)}
        #if self.balanced:
        #    return np.nanmean(dprimes)
        #else:
        #    raise Exception('Not implemented DPrime unbalanced average')

class lwlrap():
    def __init__(self):
        pass

    def _one_sample_positive_class_precisions(self,scores,truth):
        """Calculate precisions for each true class for a single sample.

        Args:
        scores: np.array of (num_classes,) giving the individual classifier scores.
        truth: np.array of (num_classes,) bools indicating which classes are true.

        Returns:
        pos_class_indices: np.array of indices of the true classes for this sample.
        pos_class_precisions: np.array of precisions corresponding to each of those
            classes.
        """
        num_classes = scores.shape[0]
        pos_class_indices = np.flatnonzero(truth > 0)
        # Only calculate precisions if there are some true classes.
        if not len(pos_class_indices):
            return pos_class_indices, np.zeros(0)
        # Retrieval list of classes for this sample.
        retrieved_classes = np.argsort(scores)[::-1]
        # class_rankings[top_scoring_class_index] == 0 etc.
        class_rankings = np.zeros(num_classes, dtype=np.int)
        class_rankings[retrieved_classes] = range(num_classes)
        # Which of these is a true label?
        retrieved_class_true = np.zeros(num_classes, dtype=np.bool)
        retrieved_class_true[class_rankings[pos_class_indices]] = True
        # Num hits for every truncated retrieval list.
        retrieved_cumulative_hits = np.cumsum(retrieved_class_true)
        # Precision of retrieval list truncated at each hit, in order of pos_labels.
        precision_at_hits = (
                retrieved_cumulative_hits[class_rankings[pos_class_indices]] /
                (1 + class_rankings[pos_class_indices].astype(np.float)))
        return pos_class_indices, precision_at_hits


    def calculate_per_class_lwlrap(self,truth, scores):
        """Calculate label-weighted label-ranking average precision.

        Arguments:
        truth: np.array of (num_samples, num_classes) giving boolean ground-truth
            of presence of that class in that sample.
        scores: np.array of (num_samples, num_classes) giving the classifier-under-
            test's real-valued score for each class for each sample.

        Returns:
        per_class_lwlrap: np.array of (num_classes,) giving the lwlrap for each
            class.
        weight_per_class: np.array of (num_classes,) giving the prior of each
            class within the truth labels.  Then the overall unbalanced lwlrap is
            simply np.sum(per_class_lwlrap * weight_per_class)
        """
        assert truth.shape == scores.shape
        num_samples, num_classes = scores.shape
        # Space to store a distinct precision value for each class on each sample.
        # Only the classes that are true for each sample will be filled in.
        precisions_for_samples_by_classes = np.zeros((num_samples, num_classes))
        for sample_num in range(num_samples):
            pos_class_indices, precision_at_hits = (
                self._one_sample_positive_class_precisions(scores[sample_num, :],
                                                    truth[sample_num, :]))
            precisions_for_samples_by_classes[sample_num, pos_class_indices] = (
                precision_at_hits)
        labels_per_class = np.sum(truth > 0, axis=0)
        weight_per_class = labels_per_class / float(np.sum(labels_per_class))
        # Form average of each column, i.e. all the precisions assigned to labels in
        # a particular class.
        per_class_lwlrap = (np.sum(precisions_for_samples_by_classes, axis=0) /
                            np.maximum(1, labels_per_class))
        # overall_lwlrap = simple average of all the actual per-class, per-sample precisions
        #                = np.sum(precisions_for_samples_by_classes) / np.sum(precisions_for_samples_by_classes > 0)
        #           also = weighted mean of per-class lwlraps, weighted by class label prior across samples
        #                = np.sum(per_class_lwlrap * weight_per_class)

        return per_class_lwlrap, weight_per_class

    def __call__(self,y_true,y_pred):
        lwlrap_per_class, weights_per_class = self.calculate_per_class_lwlrap(y_true,y_pred)

        return {'label-ranking precision': lwlrap_per_class,
                'lwlrap': np.mean(lwlrap_per_class*weights_per_class)}

class CustomPrecision():
    def __init__(self, average='micro'):
        self.average = average

    def __call__(self,y_true,y_pred):
        return {'precision': average_precision_score(y_true,y_pred,average=None),
                'micro_avg_precision': average_precision_score(y_true,y_pred,average='micro'),
                'macro_avg_precision': average_precision_score(y_true,y_pred,average='macro'),
                'weighted_avg_precision': average_precision_score(y_true,y_pred,average='weighted')}

class BinaryCrossEntropy():
    def __init__(self):
        pass

    def __call__(self,y_true,y_pred):
        loss = binary_crossentropy(y_true,y_pred)
        return {'mean_binary_crossentropy': np.mean(loss.numpy())}

class CodebookStatistics():
    def __init__(self, codebook_layer = None):
        self.codebook_layer = codebook_layer

    def __call__(self,y_true,y_pred):
        codebook_w = [w for v, w in zip(self.codebook_layers[self.codebook_layer].variables, self.codebook_layers[self.codebook_layer].get_weights()) if 'codebook' in v.name.split('/')[-1]][0]
        if codebook_w.ndim == 3:
            codebook_sim = np.stack([np.matmul(codebook_wi,codebook_wi.T)/(np.linalg.norm(codebook_wi,axis=1)*np.linalg.norm(codebook_wi.T,axis=0)) for codebook_wi in codebook_w])
        elif codebook_w.ndim == 2:
            codebook_sim = np.matmul(codebook_w,codebook_w.T)/(np.linalg.norm(codebook_w,axis=1)*np.linalg.norm(codebook_w.T,axis=0))
        
        pred_fn = tf.keras.backend.function(inputs=self.model.inputs, outputs=self.codebook_layers[self.codebook_layer].input)
        codebook_ins = np.concatenate([pred_fn(x) for x,y in self.validation_data])
        bs = self.validation_data.batch_size
        codebook_idxs = [self.codebook_layers[self.codebook_layer].get_codebook_indices(codebook_ins[i:i+bs]) for i in range(0,len(codebook_ins)-bs,bs)]
        
        if type(self.codebook_layers[self.codebook_layer]).__name__ == 'GumbelSoftmaxVQ':
            codebook_idxs = np.concatenate(codebook_idxs)
            codebook_idxs = np.reshape(codebook_idxs, (-1,codebook_idxs.shape[-2],codebook_idxs.shape[-1]))
            codebook_accesses = np.sum(codebook_idxs,axis=0)
            codebook_usage = np.sum(codebook_accesses>0,axis=-1)/codebook_accesses.shape[-1]
        elif type(self.codebook_layers[self.codebook_layer]).__name__ == 'VQLayer':
            codebook_idxs = np.concatenate(codebook_idxs)
            codebook_idxs = np.reshape(codebook_idxs,(-1,codebook_idxs.shape[-1]))
            codebook_usage = np.array([len(np.unique(codebook_idxs[:,i]))/(self.codebook_layers[self.codebook_layer].k) for i in range(codebook_idxs.shape[-1])])
        else:
            raise Exception('Unknown quantization layer type')

        return {'codebook': codebook_w, 'codebook_similarity': codebook_sim, 'codebook_usage': codebook_usage}