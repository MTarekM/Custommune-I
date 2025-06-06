import os
import sys
import traceback
import pickle
import operator

import streamlit as st
import pandas as pd
import h5py
import tensorflow as tf
from packaging.version import parse as parse_version
from tensorflow.keras import metrics
from tensorflow.keras.layers import Layer, MultiHeadAttention, Attention
from tensorflow.keras.utils import register_keras_serializable
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import model_from_json
from tensorflow.python.keras.layers.core import TFOpLambda
import operator
import tensorflow as tf
from tensorflow.python.keras.layers.core import TFOpLambda

# Register all needed ops
tf.keras.utils.get_custom_objects()['TFOpLambda'] = TFOpLambda
tf.keras.utils.get_custom_objects()['tf.nn.silu'] = tf.nn.silu
tf.keras.utils.get_custom_objects()['tf.__operators__.add'] = operator.add

def register_tf_ops():
    # Register common TensorFlow operations used in Lambda layers
    ops_to_register = {
        'add': tf.add,
        'multiply': tf.multiply,
        'subtract': tf.subtract,
        'divide': tf.divide,
        'sigmoid': tf.sigmoid,
        'tanh': tf.tanh,
        'relu': tf.nn.relu,
        'softmax': tf.nn.softmax,
        'silu': tf.nn.silu,
        'gelu': lambda x: x * tf.sigmoid(1.702 * x),  # Approximation of GELU
    }
    
    # Register operators
    ops_to_register.update({
        'tf.__operators__.add': operator.add,
        'tf.__operators__.mul': operator.mul,
        'tf.__operators__.sub': operator.sub,
        'tf.__operators__.truediv': operator.truediv,
    })
    
    # Register all with both naming conventions
    for name, func in ops_to_register.items():
        tf.keras.utils.get_custom_objects()[name] = func
        if '.' in name:
            # Also register with full path as it might be serialized this way
            tf.keras.utils.get_custom_objects()[name] = func

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
    required = ['best_combined_model.h5', 'tokenizer.pkl', 'class1_pseudosequences.csv']
    missing = [f for f in required if not os.path.exists(f)]
    if missing:
        raise FileNotFoundError(f"Missing required files: {', '.join(missing)}")


def verify_versions():
    req_min = {'tensorflow': '2.12.0', 'h5py': '3.7.0'}
    curr = {'tensorflow': tf.__version__, 'h5py': h5py.__version__}
    too_low = [f"{lib} {curr[lib]} < required {vr}" for lib, vr in req_min.items()
               if parse_version(curr[lib]) < parse_version(vr)]
    if too_low:
        raise EnvironmentError("Version mismatch: " + "; ".join(too_low))
    else:
        print(f"✅ Versions OK: TF {curr['tensorflow']}, h5py {curr['h5py']}")

# ============== Load Model, Tokenizer, HLA DB ==============
@st.cache_resource


# ============== Load Model, Tokenizer, HLA DB ==============
@st.cache_resource
def load_model_and_data():
    verify_versions()
    verify_files()

    # Step 1: Extract model architecture as JSON
    try:
        with h5py.File('best_combined_model.h5', 'r') as f:
            model_config = f.attrs.get('model_config')
            if model_config is None:
                # Try alternative location for config
                model_config = f.attrs.get('model_config')
            
            if model_config is not None:
                model_config = model_config.decode('utf-8')
            else:
                raise ValueError("Could not extract model config from H5 file")
            
            # Extract weights paths for later loading
            weight_names = [n.decode('utf-8') for n in f['model_weights'].attrs['weight_names']]
    except Exception as e:
        st.error(f"Error extracting model architecture: {str(e)}")
        st.stop()

    # Step 2: Register all custom components
    custom_objects = {
        # Core components
        'Functional': tf.keras.Model,
        'TFOpLambda': TFOpLambda,
        
        # Custom components
        'F1Score': F1Score,
        'NegativePredictiveValue': NegativePredictiveValue,
        'AdamW': AdamW,
        'SafeAddLayer': SafeAddLayer,
        'Swish': Swish,
        
        # TensorFlow operations
        'tf.nn.silu': tf.nn.silu,
        'tf.__operators__.add': operator.add,
        'tf.__operators__.mul': operator.mul,
        'tf.__operators__.sub': operator.sub,
        'tf.__operators__.truediv': operator.truediv,
        'add': tf.add,
        'multiply': tf.multiply,
        'subtract': tf.subtract,
        'divide': tf.divide,
        'sigmoid': tf.sigmoid,
        'tanh': tf.tanh,
        'relu': tf.nn.relu,
        'softmax': tf.nn.softmax,
    }

    # Register all custom objects
    for name, obj in custom_objects.items():
        tf.keras.utils.get_custom_objects()[name] = obj

    # Step 3: Try to reconstruct model from JSON
    try:
        with tf.keras.utils.custom_object_scope(custom_objects):
            model = model_from_json(model_config)
            
            # Load weights directly from H5 file
            model.load_weights('best_combined_model.h5')
            
            print("✅ Model successfully reconstructed from JSON and weights loaded")
    except Exception as e:
        st.error(f"Error reconstructing model: {str(e)}")
        traceback.print_exc()
        st.stop()

    # Load supporting data (unchanged)
    with open('tokenizer.pkl', 'rb') as f:
        tokenizer = pickle.load(f)

    hla_db = pd.read_csv('class1_pseudosequences.csv', header=None)
    pattern = r'^BoLA|^Mamu|^Patr|^SLA|^Chi|^DLA|^Eqca|^H-2|^Gogo|^H2'
    hla_db = hla_db[~hla_db[0].str.contains(pattern, case=False, regex=True)]

    return model, tokenizer, hla_db

# ============== Preprocessing & Prediction ==============
def preprocess_sequence(seq, tokenizer, max_length=50):
    enc = tokenizer.texts_to_sequences([seq])
    return pad_sequences(enc, maxlen=max_length, padding='post')

def generate_kmers(seq, k=9):
    return [seq[i:i+k] for i in range(len(seq)-k+1)] if len(seq) >= k else [seq]

def predict_binding(epitope, allele, model, tokenizer, hla_db, threshold=0.5):
    try:
        hla_seq = hla_db.loc[hla_db[0] == allele, 1].values[0]
        combo = f"{epitope}-{hla_seq}"
        proc = preprocess_sequence(combo, tokenizer)
        prob = float(model.predict(proc, verbose=0)[0][0])
        IC50_MIN, IC50_MAX, IC50_CUTOFF = 0.1, 50000.0, 5000.0
        if prob >= threshold:
            ic50 = IC50_MIN * (IC50_MAX / IC50_MIN) ** ((1 - prob) / (1 - threshold))
        else:
            ic50 = IC50_MAX + (IC50_CUTOFF - IC50_MAX) * ((threshold - prob) / threshold)
        if ic50 < 50:
            aff = "High"
        elif ic50 < 500:
            aff = "Intermediate"
        elif ic50 < 5000:
            aff = "Low"
        else:
            aff = "Non-Binder"
        return {'epitope': epitope, 'hla_allele': allele, 'pseudosequence': hla_seq, 'complex': combo, 'probability': prob, 'ic50': ic50, 'affinity': aff, 'prediction': 'Binder' if prob >= threshold else 'Non-Binder'}
    except Exception as e:
        return {'epitope': epitope, 'hla_allele': allele, 'pseudosequence': 'N/A', 'complex': 'N/A', 'probability': 0.0, 'ic50': 0.0, 'affinity': 'Error', 'prediction': f'Error: {e}'}

def predict_wrapper(ep_input, alleles, k_length, model, tokenizer, hla_db):
    eps = [e.strip() for e in ep_input.split(',') if e.strip()]
    rows = []
    for raw in eps:
        kmers = generate_kmers(raw, k=k_length)
        if not kmers:
            rows.append([raw, 'N/A'] + ['N/A']*5 + ['Error', 'Invalid (length < k)'])
            continue
        for km in kmers:
            for al in alleles:
                r = predict_binding(km, al, model, tokenizer, hla_db)
                rows.append([raw, km, r['hla_allele'], r['pseudosequence'], r['complex'], f"{r['probability']:.4f}", f"{r['ic50']:.2f}", r['affinity'], r['prediction']])
    return pd.DataFrame(rows, columns=['Input Sequence','Processed k-mer','HLA Allele','Pseudosequence','Complex','Probability','IC50 (nM)','Affinity','Prediction'])

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
    default = 'HLA-A01:01' if 'HLA-A01:01' in human_alleles else human_alleles[0]

    with st.sidebar:
        st.header("Input Parameters")
        ep_input = st.text_area("Peptide Sequence(s)", help="Comma-separated epitopes, e.g. SIINFEKL, AGSIINFEKL")
        k_length = st.number_input("k-mer Length", min_value=1, max_value=15, value=9, step=1)
        alleles = st.multiselect("HLA Allele(s)", options=human_alleles, default=[default])
        if st.button("Predict Binding"):
            if not ep_input.strip():
                st.warning("Please enter at least one peptide sequence.")
            else:
                df = predict_wrapper(ep_input, alleles, k_length, model, tokenizer, hla_db)
                st.subheader("Prediction Results")
                st.dataframe(df, use_container_width=True, height=500)

    st.markdown("""
        <style>#MainMenu{visibility:hidden;} footer{visibility:hidden;}</style>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
