import streamlit as st

def display_incident_card(incident):
    """Displays an incident card with key details."""
    st.subheader(f"Incident: {incident['incident_id'][:8]}")
    st.write(f"Status: {incident['status']}")
    st.write(f"Priority: {incident['priority']}")
    st.write(f"Reported At: {incident['reported_at']}")
    # Add more details as needed
    st.markdown("---")

def display_agent_card(agent):
    """Displays an agent card with key details."""
    st.subheader(f"Agent: {agent['name']}")
    st.write(f"Status: {agent['status']}")
    st.write(f"Skills: {', '.join(agent['skills'])}")
    # Add more details as needed
    st.markdown("---")

def display_knowledge_base_article(article):
    """Displays a knowledge base article."""
    st.subheader(article['title'])
    st.write(article['content'])
    st.markdown("---")

def display_error_message(message):
    """Displays an error message to the user."""
    st.error(message)

def display_success_message(message):
    """Displays a success message to the user."""
    st.success(message)