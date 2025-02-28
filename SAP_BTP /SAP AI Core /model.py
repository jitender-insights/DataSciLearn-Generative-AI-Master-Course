# ------------------ Metrics Dashboard ------------------
st.header("📈 System Performance Metrics")

if st.button("🔄 Refresh Metrics"):
    st.experimental_rerun()

metrics = get_all_metrics()

if metrics['system']['total_tickets'] > 0:
    with st.expander("📊 System Overview", expanded=True):
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Tickets Processed", metrics['system']['total_tickets'])
        col2.metric("Duplicate Rate", f"{metrics['system']['duplicate_rate']:.1%}")
        col3.metric("Average Confidence", f"{metrics['system']['avg_confidence']:.2f}")

    with st.expander("🔍 Vector Search Performance"):
        # Similarity Distribution
        st.subheader("Similarity Score Distribution")
        if metrics['vector']['similarity_distribution']['duplicates'] or metrics['vector']['similarity_distribution']['non_duplicates']:
            fig, ax = plt.subplots()
            ax.hist(
                metrics['vector']['similarity_distribution']['duplicates'],
                bins=20, 
                alpha=0.5, 
                label='Duplicates',
                color='red'
            )
            ax.hist(
                metrics['vector']['similarity_distribution']['non_duplicates'],
                bins=20, 
                alpha=0.5, 
                label='Non-Duplicates',
                color='green'
            )
            ax.set_xlabel("Similarity Score")
            ax.set_ylabel("Count")
            ax.legend()
            st.pyplot(fig)
        else:
            st.info("No similarity data available yet")

        # Threshold Effectiveness
        st.subheader("Threshold Crossings")
        threshold_data = {
            'Definite Duplicates': metrics['vector']['successful_thresholds']['definite'],
            'Likely Duplicates': metrics['vector']['successful_thresholds']['likely'],
            'Below Threshold': metrics['vector']['successful_thresholds']['below_threshold']
        }
        threshold_df = pd.DataFrame.from_dict(
            threshold_data, 
            orient='index',
            columns=['Count']
        )
        st.bar_chart(threshold_df)

        # Search Performance
        st.subheader("Search Metrics")
        if metrics['vector']['avg_search_time'] > 0:
            search_col1, search_col2 = st.columns(2)
            search_col1.metric("Average Search Time", f"{metrics['vector']['avg_search_time']:.4f}s")
            total_searches = len(metrics['vector']['successful_thresholds']) 
            search_col2.metric("Total Vector Searches", total_searches)

    with st.expander("🧠 LLM Performance"):
        llm_col1, llm_col2, llm_col3 = st.columns(3)
        llm_col1.metric("Success Rate", 
                      f"{metrics['llm']['success_rate']:.1%}",
                      help="Percentage of successful LLM responses")
        llm_col2.metric("Parse Success Rate", 
                      f"{metrics['llm']['parse_success_rate']:.1%}",
                      help="Percentage of successfully parsed LLM responses")
        llm_col3.metric("Avg Response Time", 
                      f"{metrics['llm']['avg_response_time']:.2f}s",
                      help="Average time taken for LLM responses")

        # Retry Statistics
        st.subheader("Retry Statistics")
        if metrics['llm']['avg_retries'] > 0:
            retry_col1, retry_col2 = st.columns(2)
            retry_col1.metric("Average Retries per Call", f"{metrics['llm']['avg_retries']:.2f}")
            total_failed = len([c for c in metrics['raw_data']['llm_calls'] if not c['success']])
            retry_col2.metric("Total Failed Calls", total_failed)

    with st.expander("📝 Recent Decisions"):
        decisions_df = pd.DataFrame(metrics['raw_data']['decisions'])
        if not decisions_df.empty:
            # Add emojis for better visualization
            decisions_df['Status'] = decisions_df['is_duplicate'].apply(
                lambda x: '🔴 Duplicate' if x else '🟢 Unique'
            )
            decisions_df['Method'] = decisions_df['method'].apply(
                lambda x: f'⚡ Vector' if x == 'vector' else f'🧠 LLM' if x == 'llm' else '❌ Error'
            )
            
            st.dataframe(
                decisions_df[['timestamp', 'Status', 'Method', 'confidence']]
                .tail(10)
                .style.format({'confidence': '{:.2%}'}),
                column_config={
                    'timestamp': 'Time',
                    'confidence': 'Confidence',
                    'Method': 'Detection Method',
                    'Status': 'Decision'
                }
            )
        else:
            st.info("No decision data available yet")
else:
    st.warning("📭 No metrics available yet - process some tickets first!")
