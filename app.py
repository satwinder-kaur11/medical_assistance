import streamlit as st
from concurrent.futures import ThreadPoolExecutor, as_completed

from Agents import Cardiologist, Psychologist, Pulmonologist, MultidisciplinaryTeam


st.title(" Multi-Agent Medical Analyzer")

uploaded_file = st.file_uploader("Upload Medical Report", type=["txt"])


if uploaded_file:

    medical_report = uploaded_file.read().decode()

    st.subheader("Medical Report")
    st.write(medical_report)

    agents = {
        "Cardiologist": Cardiologist(medical_report),
        "Psychologist": Psychologist(medical_report),
        "Pulmonologist": Pulmonologist(medical_report)
    }

    if st.button("Analyze Report"):

        responses = {}

        with ThreadPoolExecutor() as executor:

            futures = {
                executor.submit(agent.run): name
                for name, agent in agents.items()
            }

            for future in as_completed(futures):

                agent_name = futures[future]
                responses[agent_name] = future.result()

        st.subheader("Specialist Analysis")

        st.write("### Cardiologist")
        st.write(responses["Cardiologist"])

        st.write("### Psychologist")
        st.write(responses["Psychologist"])

        st.write("### Pulmonologist")
        st.write(responses["Pulmonologist"])

        team_agent = MultidisciplinaryTeam(
            responses["Cardiologist"],
            responses["Psychologist"],
            responses["Pulmonologist"]
        )

        final = team_agent.run()

        st.subheader("Final Diagnosis")
        st.write(final)