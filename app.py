import os
import sys
import traceback
import pickle

import streamlit as st
import pandas as pd
import h5py
import tensorflow as tf
from packaging.version import parse as parse_version
from tensorflow.keras import metrics
from tensorflow.keras.layers import Layer, MultiHeadAttention, Attention
from tensorflow.keras.utils import register_keras_serializable
from tensorflow.keras.preprocessing.sequence import pad_sequences
import operator
import tensorflow as tf
from tensorflow.python.keras.layers.core import TFOpLambda

# ─── Register TFOpLambda itself ───────────────────────────────────────────────
tf.keras.utils.get_custom_objects()['TFOpLambda'] = TFOpLambda

tf.keras.utils.get_custom_objects()['tf.nn.silu'] = tf.nn.silu
tf.keras.utils.get_custom_objects()['tf.__operators__.add'] = operator.add

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

# ============== File Verification ==============
def verify_files():
    required_files = {
        'model': 'best_combined_model.h5',
        'tokenizer': 'tokenizer.pkl',
        'hla_db': 'class1_pseudosequences.csv'
    }
    missing = [p for p in required_files.values() if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(f"Missing required files: {', '.join(missing)}")

# ============== Version Verification (minimum only) ==============
def verify_versions():
    required_min = {'tensorflow': '2.12.0', 'h5py': '3.7.0'}
    current = {'tensorflow': tf.__version__, 'h5py': h5py.__version__}
    too_low = []
    for lib, min_ver in required_min.items():
        if parse_version(current[lib]) < parse_version(min_ver):
            too_low.append(f"{lib} {current[lib]} < required {min_ver}")
    if too_low:
        raise EnvironmentError("Version mismatch: " + "; ".join(too_low))
    else:
        print(f"✅ Versions OK: TensorFlow {current['tensorflow']}, h5py {current['h5py']}")

# ============== Load Resources (cached) ==============
@st.cache_resource
@st.cache_resource
def load_model_and_data():
    verify_versions()
    verify_files()

    # you no longer need to include TFOpLambda or the ops here
    custom_objs = {
        'F1Score': F1Score,
        'NegativePredictiveValue': NegativePredictiveValue,
        'AdamW': AdamW,
        'SafeAddLayer': SafeAddLayer,
        'Swish': Swish,
        'MultiHeadAttention': MultiHeadAttention,
        'Attention': Attention,
        # tf.nn.silu and tf.__operators__.add are already in get_custom_objects()
    }
    model = tf.keras.models.load_model(
        'best_combined_model.h5',
        custom_objects=custom_objs,
        compile=False     # safe when you only need predict()
    )
    +    # ── WORKAROUND FOR TFOpLambda DESERIALIZATION ERRORS ───────────
+    import h5py
+    from tensorflow.keras.models import model_from_json
+
+    # 1) Read the JSON graph out of the H5 attrs
+    with h5py.File('best_combined_model.h5', 'r') as f:
+        raw = f.attrs.get('model_config')
+    # If it comes back as bytes, decode it
+    model_json = raw.decode('utf-8') if isinstance(raw, (bytes, bytearray)) else raw
+
+    # 2) Rebuild architecture with your custom layers/metrics
+    model = model_from_json(model_json, custom_objects=custom_objs)
+
+    # 3) Load the weights from the same H5
+    model.load_weights('best_combined_model.h5')
+    # ────────────────────────────────────────────────────────────────


    # … load tokenizer, hla_db, etc …
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
