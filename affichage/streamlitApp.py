
import streamlit as st
import requests


st.set_page_config(page_title="Aide au diagnostic à destination des médecins généralistes", layout="wide")

st.title("Aide au diagnostic à destination des médecins généralistes")

texte_input = st.text_input("Inscrivez la liste des symptômes du patient:", key="champ_texte")

col1, col2 = st.columns(2)

with col1:
    if st.button("Déterminer maladie", use_container_width=True):
        if texte_input:
            try:
                response = requests.post(
                    f"http://localhost:8000/detection",
                    json={"texte": texte_input}
                )
                if response.status_code == 200:
                    st.session_state['resultat'] = response.json()
                else:
                    st.session_state['resultat'] = {"erreur": f"Erreur {response.status_code}"}
            except Exception as e:
                st.session_state['resultat'] = {"erreur": str(e)}
        else:
            st.warning("Veuillez entrer du texte avant de cliquer sur Déterminer")

with col2:
    if st.button("Exemple", use_container_width=True):
        try:
            response = requests.get(f"http://localhost:8000/exemple")
            if response.status_code == 200:
                st.session_state['resultat'] = response.json()
            else:
                st.session_state['resultat'] = {"erreur": f"Erreur {response.status_code}"}
        except Exception as e:
            st.session_state['resultat'] = {"erreur": str(e)}

st.subheader("Résultat")
if 'resultat' in st.session_state:
    st.html(st.session_state['resultat']['resultat'])
else:
    st.info("Veuillez cliquer sur un des boutons pour faire apparaitre un résultat")
