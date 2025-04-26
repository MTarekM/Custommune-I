# app.py
import os
import sys
import traceback
import pickle

import streamlit as st
import pandas as pd
import h5py
import tensorflow as tf
from tensorflow.keras import metrics
from tensorflow.keras.layers import Layer, MultiHeadAttention, Attention
from tensorflow.keras.utils import register_keras_serializable
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ============== Unified Custom Components ==============

@register_keras_serializable(package='CustomMetrics')
class F1Score(tf.keras.metrics.Metric):
    def __init__(self, name='f1', threshold=0.5, **kwargs):
        super().__init__(name=name, **kwargs)
        self.threshold = threshold
        self.precision = metrics.Precision(thresholds=self.threshold)
        self.recall = metrics.Recall(thresholds=self.threshold)

    def update_state(self, y_true, y_pred, sample_weight=None):
        self.precision.update_state(y_true, y_pred, sample_weight)
        self.recall.update_state(y_true, y_pred, sample_weight)

    def result(self):
        p = self.precision.result()
        r = self.recall.result()
        return 2 * ((p * r) / (p + r + tf.keras.backend.epsilon()))

    def get_config(self):
        return {'threshold': self.threshold}


@register_keras_serializable(package='CustomMetrics')
class NegativePredictiveValue(tf.keras.metrics.Metric):
    def __init__(self, name='npv', threshold=0.5, **kwargs):
        super().__init__(name=name, **kwargs)
        self.threshold = threshold
        self.true_negatives = self.add_weight(name='tn', initializer='zeros')
        self.false_negatives = self.add_weight(name='fn', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.cast(y_pred >= self.threshold, tf.float32)
        y_true = tf.cast(y_true, tf.float32)
        tn = tf.reduce_sum((1 - y_true) * (1 - y_pred))
        fn = tf.reduce_sum(y_true * (1 - y_pred))
        self.true_negatives.assign_add(tn)
        self.false_negatives.assign_add(fn)

    def result(self):
        return self.true_negatives / (self.true_negatives + self.false_negatives + 1e-7)

    def reset_state(self):
        self.true_negatives.assign(0)
        self.false_negatives.assign(0)

    def get_config(self):
        return {'threshold': self.threshold}


@register_keras_serializable(package='CustomOptimizers')
class AdamW(tf.keras.optimizers.legacy.Adam):
    def __init__(self, weight_decay=0.01, **kwargs):
        super().__init__(**kwargs)
        self.weight_decay = weight_decay

    def _resource_apply_dense(self, grad, var, apply_state):
        var_device, var_dtype = var.device, var.dtype.base_dtype
        coefficients = ((apply_state or {}).get((var_device, var_dtype)) 
                       or self._fallback_apply_state(var_device, var_dtype))
        lr = coefficients['lr_t']
        wd = tf.cast(self.weight_decay, var_dtype)
        var.assign_sub(var * wd * lr)
        return super()._resource_apply_dense(grad, var, apply_state)

    def get_config(self):
        config = super().get_config()
        config.update({'weight_decay': self.weight_decay})
        return config


@register_keras_serializable(package='CustomLayers')
class SafeAddLayer(Layer):
    def call(self, inputs):
        return tf.add(inputs[0], inputs[1])


@register_keras_serializable(package='CustomLayers')
class Swish(Layer):
    def call(self, inputs):
        return tf.nn.silu(inputs)


# ============== File & Version Verification ==============

def verify_files():
    required_files = {
        'model': 'best_combined_model.h5',
        'tokenizer': 'tokenizer.pkl',
        'hla_db': 'class1_pseudosequences.csv'
    }
    missing = []
    for name, path in required_files.items():
        if not os.path.exists(path):
            missing.append(path)
    if missing:
        raise FileNotFoundError(f"Missing required files: {', '.join(missing)}")


def verify_versions():
    required = {'tensorflow': '2.12.0', 'h5py': '3.7.0'}
    current = {
        'tensorflow': tf.__version__,
        'h5py': h5py.__version__
    }
    mismatches = [f"{lib} (required {req}, found {current[lib]})"
                  for lib, req in required.items() if current[lib] != req]
    if mismatches:
        raise EnvironmentError("Version mismatch: " + "; ".join(mismatches))


# ============== Load Resources (cached) ==============

@st.cache_resource
def load_model_and_data():
    # 1) Verify environment
    verify_versions()
    verify_files()

    # 2) Load model
    custom_objects = {
        'F1Score': F1Score,
        'NegativePredictiveValue': NegativePredictiveValue,
        'AdamW': AdamW,
        'SafeAddLayer': SafeAddLayer,
        'Swish': Swish,
        'MultiHeadAttention': MultiHeadAttention,
        'Attention': Attention,
        'tf.nn.silu': Swish(),
        'tf.__operators__.add': SafeAddLayer(),
    }
    model = tf.keras.models.load_model(
        'best_combined_model.h5',
        custom_objects=custom_objects
    )

    # 3) Tokenizer
    with open('tokenizer.pkl', 'rb') as f:
        tokenizer = pickle.load(f)

    # 4) HLA DB
    hla_db = pd.read_csv('class1_pseudosequences.csv', header=None)
    non_human_pattern = r'^BoLA|^Mamu|^Patr|^SLA|^Chi|^DLA|^Eqca|^H-2|^Gogo|^H2'
    hla_db = hla_db[~hla_db[0].str.contains(non_human_pattern, case=False, regex=True)]

    return model, tokenizer, hla_db


# ============== Preprocessing & Prediction ==============

def preprocess_sequence(sequence, tokenizer, max_length=50):
    seq_encoded = tokenizer.texts_to_sequences([sequence])
    return pad_sequences(seq_encoded, maxlen=max_length, padding='post')


def generate_kmers(sequence, k=9):
    return [sequence[i:i+k] for i in range(len(sequence)-k+1)] if len(sequence) >= k else [sequence]


def predict_binding(epitope, hla_allele, model, tokenizer, hla_db, threshold=0.5):
    try:
        hla_seq = hla_db.loc[hla_db[0] == hla_allele, 1].values[0]
        combined = f"{epitope}-{hla_seq}"
        proc = preprocess_sequence(combined, tokenizer)
        prob = float(model.predict(proc, verbose=0)[0][0])

        IC50_MIN = 0.1
        IC50_MAX = 50000.0
        IC50_CUTOFF = 5000.0
        if prob >= threshold:
            ic50 = IC50_MIN * (IC50_MAX/IC50_MIN) ** ((1 - prob)/(1 - threshold))
        else:
            ic50 = IC50_MAX + (IC50_CUTOFF - IC50_MAX) * ((threshold - prob)/threshold)

        if ic50 < 50:
            affinity = "High"
        elif ic50 < 500:
            affinity = "Intermediate"
        elif ic50 < 5000:
            affinity = "Low"
        else:
            affinity = "Non-Binder"

        return {
            'epitope': epitope,
            'hla_allele': hla_allele,
            'pseudosequence': hla_seq,
            'complex': combined,
            'probability': prob,
            'ic50': ic50,
            'affinity': affinity,
            'prediction': 'Binder' if prob >= threshold else 'Non-Binder'
        }
    except Exception as e:
        return {
            'epitope': epitope,
            'hla_allele': hla_allele,
            'pseudosequence': 'N/A',
            'complex': 'N/A',
            'probability': 0.0,
            'ic50': 0.0,
            'affinity': 'Error',
            'prediction': f'Error: {str(e)}'
        }


def predict_wrapper(ep_input, alleles, k_length, model, tokenizer, hla_db):
    eps = [e.strip() for e in ep_input.split(',') if e.strip()]
    rows = []
    for raw_ep in eps:
        kmers = generate_kmers(raw_ep, k=k_length)
        if not kmers:
            rows.append([
                raw_ep, 'N/A', 'N/A', 'N/A', 'N/A',
                0.0, 'Error', 'Invalid (length < k)'
            ])
            continue
        for ep in kmers:
            for allele in alleles:
                r = predict_binding(ep, allele, model, tokenizer, hla_db)
                rows.append([
                    raw_ep,
                    ep,
                    r['hla_allele'],
                    r['pseudosequence'],
                    r['complex'],
                    f"{r['probability']:.4f}",
                    f"{r['ic50']:.2f}",
                    r['affinity'],
                    r['prediction']
                ])
    return pd.DataFrame(
        rows,
        columns=[
            'Input Sequence', 'Processed k-mer', 'HLA Allele',
            'Pseudosequence', 'Complex', 'Probability',
            'IC50 (nM)', 'Affinity', 'Prediction'
        ]
    )


# ============== Streamlit UI ==============

def main():
    st.set_page_config(page_title="Custommune HLA-I Epitope Binding Prediction", layout="wide")
    st.title("🧬 Custommune HLA-I Epitope Binding Prediction")

    try:
        model, tokenizer, hla_db = load_model_and_data()
    except Exception as e:
        st.error(f"Initialization failed: {e}")
        st.stop()

    human_alleles = sorted(hla_db[0].tolist())
    default = 'HLA-A01:01'
    if default not in human_alleles:
        default = human_alleles[0]

    with st.sidebar:
        st.header("Input Parameters")
        ep_input = st.text_area(
            "Peptide Sequence(s)",
            help="Comma-separated epitopes, e.g. SIINFEKL, AGSIINFEKL"
        )
        k_length = st.number_input(
            "k-mer Length",
            min_value=1, max_value=15, value=9, step=1
        )
        alleles = st.multiselect(
            "HLA Allele(s)",
            options=human_alleles,
            default=[default]
        )
        if st.button("Predict Binding"):
            if not ep_input.strip():
                st.warning("Please enter at least one peptide sequence.")
            else:
                df = predict_wrapper(ep_input, alleles, k_length, model, tokenizer, hla_db)
                st.subheader("Prediction Results")
                st.dataframe(df, use_container_width=True, height=500)

    # Hide Streamlit footer/menu
    hide_streamlit_style = """
        <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        </style>
    """
    st.markdown(hide_streamlit_style, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
