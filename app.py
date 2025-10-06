import collections.abc
# This is the patch for the compatibility issue
collections.Mapping = collections.abc.Mapping

import streamlit as st
import pandas as pd
from experta import *
import spacy
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Load spaCy model
@st.cache_resource
def load_models():
    nlp = spacy.load("en_core_web_sm")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    return nlp, model

nlp, model = load_models()

# Load and process the corpus
with open('corpus.txt', 'r', encoding='utf-8') as f:
    corpus = f.read()

# Split the corpus into sentences
doc = nlp(corpus)
sentences = [sent.text.strip() for sent in doc.sents]

# Create embeddings for the sentences
sentence_embeddings = model.encode(sentences)

# Create a FAISS index
index = faiss.IndexFlatL2(sentence_embeddings.shape[1])
index.add(sentence_embeddings)

class BloodPressure(Fact):
    """Information about the user's blood pressure."""
    pass

class Lifestyle(Fact):
    """Information about the user's lifestyle."""
    pass

class HypertensionScreening(KnowledgeEngine):
    @DefFacts()
    def _initial_action(self):
        yield Fact(action="get_bp")

    @Rule(Fact(action='get_bp'),
          NOT(BloodPressure(systolic=W(), diastolic=W())))
    def ask_bp(self):
        self.declare(Fact(systolic=st.session_state.systolic, diastolic=st.session_state.diastolic))

    @Rule(BloodPressure(systolic=P(lambda x: x < 120), diastolic=P(lambda x: x < 80)))
    def normal_bp(self):
        self.declare(Fact(bp_category="Normal"))

    @Rule(BloodPressure(systolic=P(lambda x: 120 <= x <= 129), diastolic=P(lambda x: x < 80)))
    def elevated_bp(self):
        self.declare(Fact(bp_category="Elevated"))

    @Rule(OR(BloodPressure(systolic=P(lambda x: 130 <= x <= 139)),
             BloodPressure(diastolic=P(lambda x: 80 <= x <= 89))))
    def hypertension_stage1(self):
        self.declare(Fact(bp_category="Hypertension Stage 1"))

    @Rule(OR(BloodPressure(systolic=P(lambda x: 140 <= x <= 179)),
             BloodPressure(diastolic=P(lambda x: 90 <= x <= 119))))
    def hypertension_stage2(self):
        self.declare(Fact(bp_category="Hypertension Stage 2"))

    @Rule(OR(BloodPressure(systolic=P(lambda x: x >= 180)),
             BloodPressure(diastolic=P(lambda x: x >= 120))))
    def hypertensive_crisis(self):
        self.declare(Fact(bp_category="Hypertensive Crisis"))

    # Add 25 more rules based on various risk factors
    @Rule(Lifestyle(smoker="Yes"))
    def smoker_risk(self):
        self.declare(Fact(risk="High risk due to smoking"))

    @Rule(Lifestyle(alcohol="Yes"))
    def alcohol_risk(self):
        self.declare(Fact(risk="Increased risk due to alcohol consumption"))

    @Rule(Lifestyle(physical_activity="No"))
    def inactivity_risk(self):
        self.declare(Fact(risk="Increased risk due to lack of physical activity"))

    @Rule(Lifestyle(diet="Unhealthy"))
    def diet_risk(self):
        self.declare(Fact(risk="Increased risk due to unhealthy diet"))

    @Rule(Fact(bp_category="Elevated"), Lifestyle(smoker="Yes"))
    def elevated_smoker(self):
        self.declare(Fact(recommendation="Strongly advise to quit smoking and monitor BP"))

    @Rule(Fact(bp_category="Hypertension Stage 1"), Lifestyle(physical_activity="No"))
    def stage1_inactive(self):
        self.declare(Fact(recommendation="Recommend starting a regular exercise program"))

    @Rule(Fact(bp_category="Hypertension Stage 2"), Lifestyle(diet="Unhealthy"))
    def stage2_unhealthy_diet(self):
        self.declare(Fact(recommendation="Recommend consulting a nutritionist for a DASH diet plan"))

    @Rule(Fact(bp_category="Hypertensive Crisis"))
    def crisis_recommendation(self):
        self.declare(Fact(recommendation="Seek immediate medical attention!"))

    @Rule(BloodPressure(systolic=P(lambda x: x > 140)), Lifestyle(age=P(lambda x: x > 60)))
    def elderly_high_systolic(self):
        self.declare(Fact(risk="High risk for cardiovascular events in elderly"))

    @Rule(Lifestyle(family_history="Yes"))
    def family_history_risk(self):
        self.declare(Fact(risk="Increased risk due to family history of hypertension"))

    @Rule(Lifestyle(diabetes="Yes"))
    def diabetes_risk(self):
        self.declare(Fact(risk="Significantly increased risk due to diabetes"))

    @Rule(Lifestyle(kidney_disease="Yes"))
    def kidney_disease_risk(self):
        self.declare(Fact(risk="High risk due to pre-existing kidney disease"))

    @Rule(Fact(bp_category="Hypertension Stage 1"), Lifestyle(diabetes="Yes"))
    def stage1_diabetes(self):
        self.declare(Fact(recommendation="Crucial to manage both blood pressure and blood sugar levels. Consult a doctor."))

    @Rule(Fact(bp_category="Normal"), Lifestyle(physical_activity="Yes"), Lifestyle(diet="Healthy"))
    def healthy_lifestyle(self):
        self.declare(Fact(recommendation="Maintain your healthy lifestyle to keep your blood pressure in the normal range."))

    @Rule(BloodPressure(systolic=P(lambda x: x < 90), diastolic=P(lambda x: x < 60)))
    def low_bp(self):
        self.declare(Fact(bp_category="Hypotension (Low Blood Pressure)"))

    @Rule(Fact(bp_category="Hypotension (Low Blood Pressure)"))
    def low_bp_recommendation(self):
        self.declare(Fact(recommendation="Monitor for symptoms like dizziness. Consult a doctor if symptoms persist."))

    @Rule(Lifestyle(stress="High"))
    def high_stress_risk(self):
        self.declare(Fact(risk="Increased risk due to high stress levels"))

    @Rule(Fact(bp_category="Elevated"), Lifestyle(stress="High"))
    def elevated_stress(self):
        self.declare(Fact(recommendation="Practice stress management techniques like meditation or yoga."))

    @Rule(Lifestyle(obese="Yes"))
    def obesity_risk(self):
        self.declare(Fact(risk="High risk due to obesity"))

    @Rule(Fact(bp_category="Hypertension Stage 1"), Lifestyle(obese="Yes"))
    def stage1_obese(self):
        self.declare(Fact(recommendation="Weight management is crucial. Consider a weight loss plan."))

    @Rule(Lifestyle(sleep_apnea="Yes"))
    def sleep_apnea_risk(self):
        self.declare(Fact(risk="Increased risk due to sleep apnea"))

    @Rule(Fact(bp_category="Hypertension Stage 2"), Lifestyle(sleep_apnea="Yes"))
    def stage2_sleep_apnea(self):
        self.declare(Fact(recommendation="Treating sleep apnea can help lower blood pressure. Consult a sleep specialist."))

    @Rule(Lifestyle(high_cholesterol="Yes"))
    def high_cholesterol_risk(self):
        self.declare(Fact(risk="Increased risk due to high cholesterol"))

    @Rule(Fact(bp_category="Hypertension Stage 1"), Lifestyle(high_cholesterol="Yes"))
    def stage1_high_cholesterol(self):
        self.declare(Fact(recommendation="Manage both blood pressure and cholesterol through diet, exercise, and possibly medication."))

    @Rule(Lifestyle(age=P(lambda x: x > 50)))
    def age_risk(self):
        self.declare(Fact(risk="Increased risk with age"))

    @Rule(Fact(bp_category="Elevated"), Lifestyle(age=P(lambda x: x > 50)))
    def elevated_age(self):
        self.declare(Fact(recommendation="Regular blood pressure checks are important as you age."))

    @Rule(Lifestyle(gender="Male"))
    def gender_risk_male(self):
        self.declare(Fact(risk="Men are at a higher risk of hypertension at a younger age."))

    @Rule(Lifestyle(gender="Female"), Lifestyle(age=P(lambda x: x > 65)))
    def gender_risk_female(self):
        self.declare(Fact(risk="Women are more likely to develop high blood pressure after menopause."))

    @Rule(Fact(bp_category="Normal"), Lifestyle(family_history="Yes"))
    def normal_bp_family_history(self):
        self.declare(Fact(recommendation="Continue to monitor your blood pressure regularly due to family history."))

    # This rule is now more robust and will fire even if only a category is found
    @Rule(Fact(action="get_recommendation"),
          Fact(bp_category=W("category")),
          # Use salience to make sure it runs after other rules
          salience=-1)
    def generate_recommendation(self, category):
        # Gather all risks and recommendations from the engine
        all_risks = [f['risk'] for f in self.facts.values() if 'risk' in f]
        all_recs = [f['recommendation'] for f in self.facts.values() if 'recommendation' in f]

        query = f"What are the recommendations for a person with {category} blood pressure?"
        if all_risks:
            query += f" They have the following risks: {', '.join(all_risks)}."
        if all_recs:
            query += f" The current advice is: {', '.join(all_recs)}."

        query_embedding = model.encode([query])
        D, I = index.search(query_embedding, k=3)
        recommendations = [sentences[i] for i in I[0]]
        st.session_state.recommendations = "\n".join(recommendations)

def main():
    st.title("Hybrid Intelligent System for Hypertension Screening")

    st.sidebar.header("User Input")
    st.session_state.systolic = st.sidebar.number_input("Systolic Pressure (mm Hg)", min_value=70, max_value=250, value=120)
    st.session_state.diastolic = st.sidebar.number_input("Diastolic Pressure (mm Hg)", min_value=40, max_value=150, value=80)
    st.session_state.age = st.sidebar.number_input("Age", min_value=1, max_value=120, value=30)
    st.session_state.gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
    st.session_state.smoker = st.sidebar.radio("Are you a smoker?", ("Yes", "No"))
    st.session_state.alcohol = st.sidebar.radio("Do you consume alcohol regularly?", ("Yes", "No"))
    st.session_state.physical_activity = st.sidebar.radio("Do you engage in regular physical activity?", ("Yes", "No"))
    st.session_state.diet = st.sidebar.selectbox("How would you describe your diet?", ["Healthy", "Average", "Unhealthy"])
    st.session_state.family_history = st.sidebar.radio("Do you have a family history of hypertension?", ("Yes", "No"))
    st.session_state.diabetes = st.sidebar.radio("Do you have diabetes?", ("Yes", "No"))
    st.session_state.kidney_disease = st.sidebar.radio("Do you have kidney disease?", ("Yes", "No"))
    st.session_state.stress = st.sidebar.selectbox("How are your stress levels?", ["Low", "Moderate", "High"])
    st.session_state.obese = st.sidebar.radio("Are you obese?", ("Yes", "No"))
    st.session_state.sleep_apnea = st.sidebar.radio("Do you have sleep apnea?", ("Yes", "No"))
    st.session_state.high_cholesterol = st.sidebar.radio("Do you have high cholesterol?", ("Yes", "No"))

    if st.sidebar.button("Screen"):
        engine = HypertensionScreening()
        engine.reset()
        engine.declare(BloodPressure(systolic=st.session_state.systolic, diastolic=st.session_state.diastolic))
        engine.declare(Lifestyle(smoker=st.session_state.smoker,
                                 alcohol=st.session_state.alcohol,
                                 physical_activity=st.session_state.physical_activity,
                                 diet=st.session_state.diet,
                                 age=st.session_state.age,
                                 gender=st.session_state.gender,
                                 family_history=st.session_state.family_history,
                                 diabetes=st.session_state.diabetes,
                                 kidney_disease=st.session_state.kidney_disease,
                                 stress=st.session_state.stress,
                                 obese=st.session_state.obese,
                                 sleep_apnea=st.session_state.sleep_apnea,
                                 high_cholesterol=st.session_state.high_cholesterol))
        engine.run()

        # RAG part - This now runs after the initial facts are processed
        engine.declare(Fact(action="get_recommendation"))
        engine.run()

        bp_category = ""
        risks = set()
        recommendations_rule = set()

        for fact in engine.facts.values():
            if 'bp_category' in fact:
                bp_category = fact['bp_category']
            if 'risk' in fact:
                risks.add(fact['risk'])
            if 'recommendation' in fact:
                recommendations_rule.add(fact['recommendation'])

        st.subheader("Screening Results")
        st.write(f"**Blood Pressure Category:** {bp_category}")

        if risks:
            st.write("**Identified Risks:**")
            for risk in risks:
                st.write(f"- {risk}")

        if recommendations_rule:
            st.write("**Recommendations from Rule-Based System:**")
            for rec in recommendations_rule:
                st.write(f"- {rec}")

        if "recommendations" in st.session_state and st.session_state.recommendations:
            st.subheader("Personalized Recommendations from Generative AI")
            st.write(st.session_state.recommendations)
            st.session_state.recommendations = "" # Clear for next run


if __name__ == "__main__":
    main()