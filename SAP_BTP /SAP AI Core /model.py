# Add this to your app.py file
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc
import time

# Import your utils functions
# from utils import check_duplicate, collect_performance_metrics, prepare_test_data

def metrics_dashboard():
    st.title("Ticket Duplication Detection Metrics")
    
    # File uploader for test data with ground truth
    st.subheader("Upload Test Data")
    uploaded_file = st.file_uploader("Choose a CSV file with labeled test data", type="csv")
    
    if uploaded_file is not None:
        # Load test data
        test_data = prepare_test_data(uploaded_file)
        
        # Show test data summary
        st.write(f"Test dataset loaded: {len(test_data)} tickets")
        st.write(f"Duplicate tickets: {test_data['is_duplicate'].sum()} ({test_data['is_duplicate'].mean()*100:.1f}%)")
        
        # Button to run evaluation
        if st.button("Run Evaluation"):
            with st.spinner("Evaluating model performance..."):
                # Load master dataset (you'll need to adapt this to your data loading method)
                master_df = load_master_data()  # Replace with your loading function
                
                # Run evaluation
                metrics = collect_performance_metrics(test_data, master_df, check_duplicate)
                
                # Display overall metrics
                st.subheader("Overall Performance Metrics")
                
                # Create metrics columns
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Accuracy", f"{metrics['overall']['accuracy']:.3f}")
                with col2:
                    st.metric("Precision", f"{metrics['overall']['precision']:.3f}")
                with col3:
                    st.metric("Recall", f"{metrics['overall']['recall']:.3f}")
                with col4:
                    st.metric("F1 Score", f"{metrics['overall']['f1_score']:.3f}")
                
                # Display confusion matrix
                st.subheader("Confusion Matrix")
                cm_data = [
                    ["True Negatives", "False Positives"],
                    ["False Negatives", "True Positives"]
                ]
                cm_values = [
                    [metrics['overall']['true_negatives'], metrics['overall']['false_positives']],
                    [metrics['overall']['false_negatives'], metrics['overall']['true_positives']]
                ]
                
                fig, ax = plt.subplots(figsize=(6, 4))
                sns.heatmap(cm_values, annot=True, fmt='d', cmap='Blues', 
                            xticklabels=["Not Duplicate", "Duplicate"],
                            yticklabels=["Not Duplicate", "Duplicate"])
                plt.ylabel('Actual')
                plt.xlabel('Predicted')
                st.pyplot(fig)
                
                # Display ROC curve
                st.subheader("ROC Curve")
                y_true = metrics['overall']['y_true']
                scores = metrics['overall']['scores']
                
                fpr, tpr, _ = roc_curve(y_true, scores)
                roc_auc = auc(fpr, tpr)
                
                fig, ax = plt.subplots(figsize=(8, 6))
                plt.plot(fpr, tpr, color='darkorange', lw=2, 
                         label=f'ROC curve (AUC = {roc_auc:.2f})')
                plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
                plt.xlim([0.0, 1.0])
                plt.ylim([0.0, 1.05])
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                plt.title('Receiver Operating Characteristic')
                plt.legend(loc="lower right")
                st.pyplot(fig)
                
                # Performance by threshold
                st.subheader("Performance by Confidence Threshold")
                threshold_data = pd.DataFrame(metrics['by_threshold'])
                
                fig, ax = plt.subplots(figsize=(10, 6))
                plt.plot(threshold_data['threshold'], threshold_data['precision'], 'b-', label='Precision')
                plt.plot(threshold_data['threshold'], threshold_data['recall'], 'g-', label='Recall')
                plt.plot(threshold_data['threshold'], threshold_data['f1_score'], 'r-', label='F1 Score')
                plt.xlabel('Confidence Threshold')
                plt.ylabel('Score')
                plt.title('Performance vs Confidence Threshold')
                plt.legend()
                plt.grid(True)
                st.pyplot(fig)
                
                # Show metrics by component
                st.subheader("Performance by Component")
                
                # Convert component metrics to DataFrame
                component_data = []
                for component, data in metrics['by_component'].items():
                    component_data.append({
                        'Component': component,
                        'Sample Count': data['sample_count'],
                        'Precision': data['precision'],
                        'Recall': data['recall'],
                        'F1 Score': data['f1_score']
                    })
                
                if component_data:
                    component_df = pd.DataFrame(component_data)
                    st.dataframe(component_df)
                    
                    # Plot component F1 scores
                    fig, ax = plt.subplots(figsize=(10, 6))
                    component_df = component_df.sort_values('F1 Score', ascending=False)
                    sns.barplot(x='F1 Score', y='Component', data=component_df)
                    plt.title('F1 Score by Component')
                    plt.xlim(0, 1)
                    st.pyplot(fig)
                else:
                    st.write("Not enough data to analyze by component")
                
                # Timing metrics
                st.subheader("Processing Time Metrics")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Mean Processing Time", f"{metrics['timing']['mean_time']:.3f} sec")
                with col2:
                    st.metric("Maximum Time", f"{metrics['timing']['max_time']:.3f} sec")
                with col3:
                    st.metric("Total Processing Time", f"{metrics['timing']['total_time']:.1f} sec")

# Add this to your app.py main function:
# if __name__ == "__main__":
#     st.sidebar.title("Navigation")
#     page = st.sidebar.selectbox("Choose a page", ["Ticket Analysis", "Metrics Dashboard"])
#     
#     if page == "Ticket Analysis":
#         # Your existing ticket analysis page
#         pass
#     elif page == "Metrics Dashboard":
#         metrics_dashboard()
