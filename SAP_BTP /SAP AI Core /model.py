import os
import streamlit as st
import matplotlib.pyplot as plt
from utils import (
    load_master_data, load_new_ticket_data,
    check_duplicate, check_recently_closed_tickets,
    is_global_outage, create_subtask, get_all_metrics
)

st.title("🔍 Telefonica Ticket Duplication System")

# ------------------ Load Master Data ------------------
st.sidebar.header("Master Data Stats")
try:
    master_df = load_master_data()
    st.sidebar.write(f"✅ *Total Tickets:* {len(master_df)}")
    open_tickets = master_df[master_df["status"] == "open"]
    st.sidebar.write(f"🟢 *Open Tickets:* {len(open_tickets)}")
    st.sidebar.write(f"🏢 *Companies:* {master_df['company_code'].nunique()}")
    st.success("Master data loaded successfully!")
except Exception as e:
    st.error(f"❌ Error loading master data: {str(e)}")
    st.stop()

# ------------------ Manual Ticket Submission Form ------------------
st.header("📩 Submit New Ticket")
with st.form("ticket_form"):
    summary = st.text_input("📝 Summary")
    description = st.text_area("📄 Description")
    company_code = st.text_input("🏢 Company Code")
    component = st.text_input("🛠 Component")
    incident_type = st.selectbox("📌 Incident Type", ["Support", "Preagreed"])
    submitted = st.form_submit_button("🔍 Check for Duplicates")

if submitted:
    if not summary or not description or not company_code or not component:
        st.warning("⚠ Please fill in all required fields.")
    else:
        with st.spinner("Analyzing ticket..."):
            ticket_data = {
                "summary": summary,
                "description": description,
                "company_code": company_code,
                "component": component,
                "incident_type": incident_type
            }
            # *Step 1: Check for Global Outages*
            if is_global_outage(ticket_data, master_df):
                st.warning("🚨 Ticket matches an *ongoing global outage*. Marked as duplicate!")
            else:
                # *Step 2: Check Recently Closed Tickets*
                recent_closed_result = check_recently_closed_tickets(ticket_data, master_df)
                if recent_closed_result["is_duplicate"]:
                    st.warning(f"🚨 *Duplicate Found (Recently Closed Ticket)!* Original Ticket: {recent_closed_result['original_ticket_id']}")
                    st.info(f"Confidence: {recent_closed_result['confidence']:.2%}")
                    st.info(f"Reasoning: {recent_closed_result['reasoning']}")
                    subtask = create_subtask(recent_closed_result["original_ticket_id"], ticket_data)
                    st.info("✅ Created subtask:")
                    st.json(subtask)
                else:
                    # *Step 3: Check for Duplicates via LLM Analysis*
                    duplicate_result = check_duplicate(master_df, ticket_data)
                    if duplicate_result["is_duplicate"]:
                        st.warning(f"🚨 *Duplicate Ticket Found!* Original Ticket: {duplicate_result['original_ticket_id']}")
                        st.info(f"Confidence: {duplicate_result['confidence']:.2%}")
                        st.info(f"Reasoning: {duplicate_result['reasoning']}")
                        subtask = create_subtask(duplicate_result["original_ticket_id"], ticket_data)
                        st.info("✅ Created subtask:")
                        st.json(subtask)
                    else:
                        st.success("✅ No duplicates found - *ticket can be created*!")
                        if "reasoning" in duplicate_result:
                            st.info(f"Analysis: {duplicate_result['reasoning']}")

# ------------------ Process Tickets from Dataset ------------------
st.header("📊 Process New Ticket Data from Dataset")
if st.button("📂 Process Dataset Tickets"):
    try:
        new_tickets = load_new_ticket_data()
        if not new_tickets:
            st.warning("⚠ No new ticket data found.")
        else:
            for ticket_data in new_tickets:
                st.subheader(f"🆕 Ticket: {ticket_data['summary']}")
                with st.spinner("Analyzing ticket..."):
                    # *Step 1: Check for Global Outages*
                    if is_global_outage(ticket_data, master_df):
                        st.warning("🚨 Ticket matches an *ongoing global outage*. Marked as duplicate!")
                        continue
                    # *Step 2: Check Recently Closed Tickets*
                    recent_closed_result = check_recently_closed_tickets(ticket_data, master_df)
                    if recent_closed_result["is_duplicate"]:
                        st.warning(f"🚨 *Duplicate Found (Recently Closed Ticket)!* Original Ticket: {recent_closed_result['original_ticket_id']}")
                        st.info(f"Confidence: {recent_closed_result['confidence']:.2%}")
                        st.info(f"Reasoning: {recent_closed_result['reasoning']}")
                        subtask = create_subtask(recent_closed_result["original_ticket_id"], ticket_data)
                        st.info("✅ Created subtask:")
                        st.json(subtask)
                        continue
                    # *Step 3: Check for Duplicates via LLM Analysis*
                    duplicate_result = check_duplicate(master_df, ticket_data)
                    if duplicate_result["is_duplicate"]:
                        st.warning(f"🚨 *Duplicate Ticket Found!* Original Ticket: {duplicate_result['original_ticket_id']}")
                        st.info(f"Confidence: {duplicate_result['confidence']:.2%}")
                        st.info(f"Reasoning: {duplicate_result['reasoning']}")
                        subtask = create_subtask(duplicate_result["original_ticket_id"], ticket_data)
                        st.info("✅ Created subtask:")
                        st.json(subtask)
                    else:
                        st.success("✅ No duplicates found - *ticket can be created*!")
                        if "reasoning" in duplicate_result:
                            st.info(f"Analysis: {duplicate_result['reasoning']}")
    except Exception as e:
        st.error(f"❌ Error processing dataset: {str(e)}")

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
        st.subheader("Similarity Score Distribution")
        fig, ax = plt.subplots()
        ax.hist(
            metrics['vector']['similarity_distribution']['duplicates'],
            bins=20, alpha=0.5, label='Duplicates'
        )
        ax.hist(
            metrics['vector']['similarity_distribution']['non_duplicates'],
            bins=20, alpha=0.5, label='Non-Duplicates'
        )
        ax.set_xlabel("Similarity Score")
        ax.set_ylabel("Count")
        ax.legend()
        st.pyplot(fig)

        st.subheader("Threshold Effectiveness")
        threshold_df = pd.DataFrame({
            'Threshold Type': ['Definite', 'Likely', 'Below'],
            'Count': [
                metrics['vector']['successful_thresholds']['definite'],
                metrics['vector']['successful_thresholds']['likely'],
                metrics['vector']['successful_thresholds']['none']
            ]
        })
        st.bar_chart(threshold_df.set_index('Threshold Type'))

    with st.expander("🧠 LLM Performance"):
        llm_col1, llm_col2, llm_col3 = st.columns(3)
        llm_col1.metric("Success Rate", f"{metrics['llm']['success_rate']:.1%}")
        llm_col2.metric("Parse Success Rate", f"{metrics['llm']['parse_success_rate']:.1%}")
        llm_col3.metric("Avg Response Time", f"{metrics['llm']['avg_response_time']:.2f}s")

    with st.expander("📝 Recent Decisions"):
        st.dataframe(pd.DataFrame(metrics['raw_data']['decisions']).tail(10))
else:
    st.warning("No metrics available yet - process some tickets first!")
