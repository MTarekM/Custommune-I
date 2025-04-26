import json
import h5py
import streamlit as st
import tensorflow as tf
import pandas as pd
import pickle
import sys
import traceback
import numpy as np
from tensorflow.keras.layers import Layer, MultiHeadAttention, Attention
from tensorflow.keras.models import model_from_json
from tensorflow.keras import metrics
from tensorflow.keras.utils import register_keras_serializable
from tensorflow.keras.preprocessing.sequence import pad_sequences
import os
import inspect  

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow info messages
# ============== Custom Components ==============
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
    def __init__(self, **kwargs):
        # Filter out unexpected arguments before passing to super
        self._function = kwargs.pop('function', None)  # Capture but ignore
        super().__init__(**kwargs)
    
    def call(self, inputs):
        # Maintain original functionality
        if isinstance(inputs, list):
            return tf.add(inputs[0], inputs[1])
        return tf.add(inputs, inputs)
    
    def get_config(self):
        # Return only expected configuration
        config = super().get_config()
        return config

    @classmethod
    def from_config(cls, config):
        # Filter out unexpected config entries
        safe_keys = ['name', 'trainable', 'dtype']
        safe_config = {k: config[k] for k in safe_keys if k in config}
        return cls(**safe_config)

@register_keras_serializable(package='CustomLayers')
class Swish(Layer):
    def call(self, inputs):
        return tf.nn.silu(inputs)

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
@register_keras_serializable(package='CustomLayers')
class CompatibleMultiHeadAttention(MultiHeadAttention):
    def __init__(self, **kwargs):
        # Filter out unexpected arguments
        kwargs.pop('query_shape', None)
        kwargs.pop('key_shape', None)
        kwargs.pop('value_shape', None)
        super().__init__(**kwargs)

    @classmethod
    def from_config(cls, config):
        # Remove shape parameters before instantiation
        config.pop('query_shape', None)
        config.pop('key_shape', None)
        config.pop('value_shape', None)
        return cls(**config)
# ============== Utility Functions ==============
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
    required = {'tensorflow': '2.19.0', 'h5py': '3.13.0'}
    current = {
        'tensorflow': tf.__version__,
        'h5py': h5py.__version__
    }
    
    for lib, ver in required.items():
        if current[lib] != ver:
            raise EnvironmentError(f"Version mismatch for {lib}: Required {ver}, Found {current[lib]}")

# ============== Enhanced Custom Layers ==============
@register_keras_serializable(package='CustomLayers')
class FallbackLayer(Layer):
    """Handles unknown layer types and unexpected config parameters"""
    def __init__(self, **kwargs):
        self._unexpected = kwargs.pop('unexpected', {})
        super().__init__(**kwargs)
    
    def call(self, inputs):
        return inputs  # Identity operation for safety
    
    def get_config(self):
        config = super().get_config()
        config.update(self._unexpected)
        return config
    
    @classmethod
    def from_config(cls, config):
        expected_keys = ['name', 'trainable', 'dtype']
        safe_config = {k: config.pop(k) for k in expected_keys if k in config}
        return cls(unexpected=config, **safe_config)

@register_keras_serializable(package='CustomLayers')
class RobustMultiHeadAttention(MultiHeadAttention):
    """Handles MHA layers with extra shape information"""
    def __init__(self, **kwargs):
        # Get valid parent class parameters
        parent_args = inspect.getfullargspec(super().__init__).args
        parent_args.remove('self')
        
        # Separate valid and extra arguments
        valid_kwargs = {k: v for k, v in kwargs.items() if k in parent_args}
        self.extra_config = {k: v for k, v in kwargs.items() if k not in parent_args}
        
        super().__init__(**valid_kwargs)
    
    def get_config(self):
        config = super().get_config()
        config.update(self.extra_config)
        return config
    
    @classmethod
    def from_config(cls, config):
        # Get parent class parameters
        parent_args = inspect.getfullargspec(super().__init__).args
        parent_args.remove('self')
        
        # Split configuration
        parent_config = {k: v for k, v in config.items() if k in parent_args}
        extra_config = {k: v for k, v in config.items() if k not in parent_args}
        
        instance = cls(**parent_config)
        instance.extra_config = extra_config
        return instance

# ============== Updated Model Loading ==============
def load_model_with_custom_objects():
    # Comprehensive custom objects mapping
    custom_objects = {
        # Core components
        'F1Score': F1Score,
        'NegativePredictiveValue': NegativePredictiveValue,
        'AdamW': AdamW,
        'SafeAddLayer': SafeAddLayer,
        'Swish': Swish,
        
        # Enhanced layer handlers
        'MultiHeadAttention': RobustMultiHeadAttention,
        'Attention': Attention,
        'FallbackLayer': FallbackLayer,
        
        # TensorFlow operation mappings
        'tf.nn.silu': Swish(),
        'tf.__operators__.add': SafeAddLayer(),
        'TFOpLambda': FallbackLayer,
        'Lambda': FallbackLayer,
        'operators.add': SafeAddLayer(),
        'keras': tf.keras,
        
        # Legacy format support
        'function': FallbackLayer,
        'SymbolicException': FallbackLayer,
    }

    # Configure environment for stable loading
    tf.keras.config.enable_unsafe_deserialization = True
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Reduce TF logging
    
    try:
        # Attempt standard load
        model = tf.keras.models.load_model(
            'best_combined_model.h5',
            custom_objects=custom_objects
        )
    except Exception as e:
        # Fallback strategy for legacy models
        try:
            with h5py.File('best_combined_model.h5', 'r') as f:
                model = model_from_json(
                    f.attrs['model_config'],
                    custom_objects=custom_objects
                )
                model.load_weights(f['model_weights'])
        except Exception as inner_e:
            raise RuntimeError(f"Model loading failed: {str(inner_e)}") from inner_e

    # Verification with actual input shape
    try:
        sample_input = np.random.rand(1, 50).astype(np.float32)
        _ = model.predict(sample_input, verbose=0)
    except Exception as e:
        raise RuntimeError("Model verification failed") from e
    
    return model
@st.cache_data
def load_tokenizer():
    with open('tokenizer.pkl', 'rb') as f:
        return pickle.load(f)

@st.cache_data
def load_hla_database():
    hla_db = pd.read_csv('class1_pseudosequences.csv', header=None)
    non_human_pattern = r'^BoLA|^Mamu|^Patr|^SLA|^Chi|^DLA|^Eqca|^H-2|^Gogo|^H2'
    return hla_db[~hla_db[0].str.contains(non_human_pattern, case=False, regex=True)]

def generate_kmers(sequence, k=9):
    return [sequence[i:i+k] for i in range(len(sequence)-k+1)] if len(sequence) >= k else [sequence]

def preprocess_sequence(sequence, tokenizer, max_length=50):
    seq_encoded = tokenizer.texts_to_sequences([sequence])
    return pad_sequences(seq_encoded, maxlen=max_length, padding='post')

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

# ============== Streamlit UI ==============
def main():
    st.set_page_config(
        page_title="Custommune HLA Predictor",
        page_icon="🧬",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("Custommune HLA-I Epitope Binding Prediction")
    
    try:
        # Enable unsafe deserialization first
        tf.keras.config.enable_unsafe_deserialization = True
        
        verify_files()
        verify_versions()
        model = load_model_with_custom_objects()
        tokenizer = load_tokenizer()
        hla_db = load_hla_database()
        
        human_alleles = sorted(hla_db[0].tolist())
        default_allele = next((a for a in ['HLA-A01:01', 'HLA-A*01:01'] if a in human_alleles), human_alleles[0])
        
        with st.sidebar:
            st.header("Input Parameters")
            ep_input = st.text_area(
                "Peptide Sequence(s)",
                help="Enter comma-separated epitopes (e.g., SIINFEKL, AGSIINFEKL)",
                placeholder="Enter peptide sequences here..."
            )
            
            k_length = st.slider(
                "k-mer Length",
                min_value=1,
                max_value=15,
                value=9,
                help="Length of peptide fragments to analyze"
            )
            
            selected_alleles = st.multiselect(
                "HLA Allele(s)",
                options=human_alleles,
                default=[default_allele],
                help="Select one or more HLA alleles for analysis"
            )
            
            threshold = st.slider(
                "Prediction Threshold",
                min_value=0.0,
                max_value=1.0,
                value=0.5,
                step=0.01,
                help="Probability threshold for binding classification"
            )
            
            run_analysis = st.button("Run Analysis", type="primary")

        if run_analysis:
            if not ep_input.strip():
                st.error("Please enter at least one peptide sequence")
                return
                
            if not selected_alleles:
                st.error("Please select at least one HLA allele")
                return
                
            peptides = [e.strip() for e in ep_input.split(',') if e.strip()]
            results = []
            
            with st.spinner(f"Processing {len(peptides)} peptides..."):
                progress_bar = st.progress(0)
                total_steps = len(peptides) * len(selected_alleles)
                
                for i, raw_ep in enumerate(peptides):
                    kmers = generate_kmers(raw_ep, k=k_length)
                    
                    if not kmers:
                        results.append({
                            'Input Sequence': raw_ep,
                            'Processed k-mer': 'N/A',
                            'HLA Allele': 'N/A',
                            'Pseudosequence': 'N/A',
                            'Complex': 'N/A',
                            'Probability': 0.0,
                            'IC50 (nM)': 0.0,
                            'Affinity': 'Invalid (length < k)',
                            'Prediction': 'Error'
                        })
                        continue
                        
                    for allele in selected_alleles:
                        for kmer in kmers:
                            result = predict_binding(kmer, allele, model, tokenizer, hla_db, threshold)
                            results.append({
                                'Input Sequence': raw_ep,
                                'Processed k-mer': kmer,
                                'HLA Allele': allele,
                                'Pseudosequence': result['pseudosequence'],
                                'Complex': result['complex'],
                                'Probability': result['probability'],
                                'IC50 (nM)': result['ic50'],
                                'Affinity': result['affinity'],
                                'Prediction': result['prediction']
                            })
                        progress = ((i * len(selected_alleles)) + (selected_alleles.index(allele)+1)) / total_steps
                        progress_bar.progress(min(progress, 1.0))
                
                df = pd.DataFrame(results)
                st.success("Analysis completed!")
                
                st.subheader("Prediction Results")
                st.dataframe(
                    df,
                    use_container_width=True,
                    height=600,
                    column_config={
                        "IC50 (nM)": st.column_config.NumberColumn(
                            format="%.2f nM",
                        ),
                        "Probability": st.column_config.NumberColumn(
                            format="%.4f",
                        )
                    }
                )
                
                st.download_button(
                    label="Download Results as CSV",
                    data=df.to_csv(index=False).encode('utf-8'),
                    file_name='hla_predictions.csv',
                    mime='text/csv'
                )
                
    except Exception as e:
        st.error(f"Application Error: {str(e)}")
        st.error("Check terminal for detailed traceback")
        traceback.print_exc()
        st.stop()

if __name__ == "__main__":
    main()
