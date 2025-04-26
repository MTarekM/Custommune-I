import inspect
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

# Environment Configuration
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow logs
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Disable GPU

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

@register_keras_serializable(package='CustomLayers')
class SafeAddLayer(Layer):
    def call(self, inputs):
        if isinstance(inputs, list):
            return tf.add(inputs[0], inputs[1])
        return tf.add(inputs, inputs)
    
    def get_config(self):
        return super().get_config()

@register_keras_serializable(package='CustomLayers')
class Swish(Layer):
    def call(self, inputs):
        return tf.nn.silu(inputs)

@register_keras_serializable(package='CustomLayers')
class RobustMultiHeadAttention(MultiHeadAttention):
    def __init__(self, **kwargs):
        base_args = inspect.getfullargspec(super().__init__).args
        valid_kwargs = {k: v for k, v in kwargs.items() if k in base_args}
        self.extra_config = {k: v for k, v in kwargs.items() if k not in base_args}
        super().__init__(**valid_kwargs)

    def get_config(self):
        config = super().get_config()
        config.update(self.extra_config)
        return config

# ============== Core Functions ==============
def verify_files():
    required_files = {
        'model': 'converted_model',
        'tokenizer': 'tokenizer.pkl',
        'hla_db': 'class1_pseudosequences.csv'
    }
    missing = [path for path in required_files.values() if not os.path.exists(path)]
    if missing:
        raise FileNotFoundError(f"Missing required files: {', '.join(missing)}")

@st.cache_resource
def load_model():
    custom_objects = {
        'F1Score': F1Score,
        'SafeAddLayer': SafeAddLayer,
        'Swish': Swish,
        'RobustMultiHeadAttention': RobustMultiHeadAttention,
        'MultiHeadAttention': RobustMultiHeadAttention,
        'tf.nn.silu': Swish(),
        'tf.__operators__.add': SafeAddLayer(),
        'TFOpLambda': SafeAddLayer,
        'keras': tf.keras
    }

    try:
        return tf.keras.models.load_model('converted_model', custom_objects=custom_objects)
    except Exception as e:
        st.error(f"""Model loading failed: {str(e)}
                  Ensure you've converted the model using:
                  model.save('converted_model', save_format='tf')""")
        st.stop()

@st.cache_data
def load_tokenizer():
    with open('tokenizer.pkl', 'rb') as f:
        return pickle.load(f)

@st.cache_data
def load_hla_database():
    hla_db = pd.read_csv('class1_pseudosequences.csv', header=None)
    non_human_pattern = r'^BoLA|^Mamu|^Patr|^SLA|^Chi|^DLA|^Eqca|^H-2|^Gogo|^H2'
    return hla_db[~hla_db[0].str.contains(non_human_pattern, case=False, regex=True)]

# ============== Processing Functions ==============
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

        affinity = "High" if ic50 < 50 else "Intermediate" if ic50 < 500 else "Low" if ic50 < 5000 else "Non-Binder"

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
        verify_files()
        model = load_model()
        tokenizer = load_tokenizer()
        hla_db = load_hla_database()
        
        human_alleles = sorted(hla_db[0].tolist())
        default_allele = next((a for a in ['HLA-A01:01', 'HLA-A*01:01'] if a in human_alleles), human_alleles[0])
        
        with st.sidebar:
            st.header("Input Parameters")
            ep_input = st.text_area(
                "Peptide Sequence(s)",
                placeholder="Enter comma-separated epitopes (e.g., SIINFEKL, AGSIINFEKL)",
                height=150
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
            
            if st.button("Run Analysis", type="primary"):
                if not ep_input.strip():
                    st.error("Please enter at least one peptide sequence")
                    st.stop()
                if not selected_alleles:
                    st.error("Please select at least one HLA allele")
                    st.stop()

        if 'run_analysis' in locals():
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
                        "IC50 (nM)": st.column_config.NumberColumn(format="%.2f nM"),
                        "Probability": st.column_config.NumberColumn(format="%.4f")
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
