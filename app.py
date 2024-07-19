import streamlit as st
import pandas as pd

import PIL
from PIL import Image
import base64

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import KNNImputer
from sklearn.decomposition import PCA
import joblib

# ***************************************************************************

# Load the trained model
model_clinical = joblib.load('models/voting_classifier_model_clinical_78.pkl')
model_relevant = joblib.load('models/gb_model_relevant_78.pkl')



# ***************************************************************************

@st.cache_data
def get_img_as_base64(file):
    with open(file, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()


bg_img = get_img_as_base64("img/bg.png")

# *************************************************************************************************************************************

def main_bg():
    # Add custom CSS for setting the background image
    background_css = f"""
        <style>
        [data-testid="stAppViewContainer"] > .main {{
        background-image: url("data:image/png;base64,{bg_img}");
        background-size: 100%;
        background-position: top left;
        background-repeat: no-repeat;
        background-attachment: local;
        }}



        </style>
    """
    st.markdown(background_css, unsafe_allow_html=True)


# *************************************************************************************************************************************


# Function to preprocess user input
def preprocess_input(df_input):
    
    print(df_input)
    # Label encoding for categorical values

    categorical_columns = ['type_of_breast_surgery', 'cancer_type_detailed', 'cellularity', 'pam50_+_claudin-low_subtype',
                       'cohort', 'er_status_measured_by_ihc', 'er_status', 'neoplasm_histologic_grade',
                       'her2_status_measured_by_snp6', 'her2_status', 'tumor_other_histologic_subtype',
                       'inferred_menopausal_state', 'integrative_cluster', 'primary_tumor_laterality',
                       'oncotree_code', 'pr_status', '3-gene_classifier_subtype', 'tumor_stage']

    label_encoders = {}
    for column in df_input.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        df_input[column] = le.fit_transform(df_input[column])
        label_encoders[column] = le

    print(df_input)

    scaler = StandardScaler()
    X = pd.DataFrame(scaler.fit_transform(df_input), columns=df_input.columns)
    print(df_input)

    return df_input

# *************************************************************************************************************************************

# Home page
def home():
    main_bg()
    st.write("""
    #                          
    #                                                
                                                                                                        
    """)
    
    st.write("##                                                             ")
    st.write("##                                                             ")

    col1, col2 = st.columns(2)
    
    if col1.button('Predict with Clinical Attributes'):
        st.session_state.page = 'clinical'
    
    if col2.button('Predict with Relevant Features'):
        st.session_state.page = 'relevant'

# *************************************************************************************************************************************

# Clinical Attributes page
def clinical():
    st.title('Breast Cancer Survival Prediction Using Clinical Features')
    def user_input_features():
        age_at_diagnosis = st.sidebar.slider('Age', 0, 150, 50, key='age_slider')

        type_of_breast_surgery_options = ['MASTECTOMY', 'BREAST CONSERVING']
        type_of_breast_surgery = st.sidebar.selectbox('Type of breast surgery', type_of_breast_surgery_options, key='type_of_breast_surgery_selectbox')

        cancer_type_detailed_options = ['Breast','Breast Invasive Ductal Carcinoma', 'Breast Mixed Ductal and Lobular Carcinoma', 'Breast Invasive Lobular Carcinoma', 'Breast Invasive Mixed Mucinous Carcinoma', 'Metaplastic Breast Cancer']
        cancer_type_detailed = st.sidebar.selectbox('Cancer type', cancer_type_detailed_options, key='cancer_type_detailed_selectbox')

        cellularity_options = ['High','Moderate', 'Low']
        cellularity = st.sidebar.selectbox('Cellularity', cellularity_options, key='cellularity_selectbox')

        chemotherapy_options = [0, 1]
        chemotherapy = st.sidebar.selectbox('Chemotherapy', chemotherapy_options, key='chemotherapy_selectbox')

        pam50_options = ['LumA','LumB', 'Her2', 'claudin-low', 'Basal', 'Normal', 'NC']
        pam50 = st.sidebar.selectbox('PAM50 + Claudin-low Subtype', pam50_options, key='pam50_selectbox')

        cohort_options = [1.0, 2.0, 3.0, 4.0, 5.0]
        cohort = st.sidebar.selectbox('Cohort', cohort_options, key='cohort_selectbox')

        er_status_measured_by_ihc_options = ['Positive', 'Negative']
        er_status_measured_by_ihc = st.sidebar.selectbox('ER Status (measured by IHC)', er_status_measured_by_ihc_options, key='er_status_measured_by_ihc_selectbox')

        er_status_options = ['Positive', 'Negative']
        er_status = st.sidebar.selectbox('Er Status (Overall)', er_status_options, key='er_status_selectbox')

        neoplasm_histologic_grade_options = [1.0, 2.0, 3.0]
        neoplasm_histologic_grade = st.sidebar.selectbox('Neoplasm Histologic Grade', neoplasm_histologic_grade_options, key='neoplasm_histologic_grade_selectbox')

        her2_status_measured_by_snp6_options = ['Neutral', 'Gain', 'Loss', 'Undef']
        her2_status_measured_by_snp6 = st.sidebar.selectbox('Her2 Status (snp6)', her2_status_measured_by_snp6_options, key='her2_status_measured_by_snp6_selectbox')

        her2_status_options = ['Positive', 'Negative']
        her2_status = st.sidebar.selectbox('Her2 Status (Overall)', her2_status_options, key='her2_status_selectbox')

        tumor_other_histologic_subtype_options = ['Ductal/NST', 'Mixed', 'Lobular', 'Medullary','Mucinous', 'Tabular/Cribriform', 'Mataplastic', 'Other']
        tumor_other_histologic_subtype = st.sidebar.selectbox('Tumor Other Histologic Subtype', tumor_other_histologic_subtype_options, key='tumor_other_histologic_subtype_selectbox')

        hormone_therapy_options = [0, 1]
        hormone_therapy = st.sidebar.selectbox('Hormone Therapy', hormone_therapy_options, key='hormone_therapy_selectbox')

        inferred_menopausal_state_options = ['Post', 'Pre']
        inferred_menopausal_state = st.sidebar.selectbox('Menopausal State', inferred_menopausal_state_options, key='inferred_menopausal_state_selectbox')

        integrative_cluster_options = ['1', '1', '2', '4ER+', '4ER-', '5', '6', '7', '8', '9', '10']
        integrative_cluster = st.sidebar.selectbox('Integrative Cluster', integrative_cluster_options, key='integrative_cluster_selectbox')

        primary_tumor_laterality_options = ['Left', 'Right']
        primary_tumor_laterality = st.sidebar.selectbox('Primary tumor laterality', primary_tumor_laterality_options, key='primary_tumor_laterality_selectbox')

        lymph_nodes_examined_positive = st.sidebar.slider('Lymph Nodes Examined Positive', 0.0, 45.0, 0.0, key='lymph_nodes_examined_positive_slider')

        mutation_count = st.sidebar.slider('Mutation count', 0.0, 80.0, 0.0, key='mutation_count_slider')

        nottingham_prognostic_index = st.sidebar.slider('Nottingham prognostic index', 1.000, 7.000, 4.000, key='nottingham_prognostic_index_slider')

        oncotree_code_options = ['IDC', 'MDLC', 'ILC', 'IMMC', 'BREAST', 'MBC']
        oncotree_code = st.sidebar.selectbox('Oncotree code', oncotree_code_options, key='oncotree_code_selectbox')

        pr_status_options = ['Positive', 'Negative']
        pr_status = st.sidebar.selectbox('PR Status', pr_status_options, key='pr_status_selectbox')

        radio_therapy_options = [0, 1]
        radio_therapy = st.sidebar.selectbox('Radio Therapy', radio_therapy_options, key='radio_therapy_selectbox')

        _3gene_classifier_subtype_options = ['ER+/HER2- High Prolif', 'ER-/HER2-', 'HER2+']
        _3gene_classifier_subtype = st.sidebar.selectbox('3 gene classifier subtype', _3gene_classifier_subtype_options, key='_3gene_classifier_subtype_selectbox')

        tumor_size = st.sidebar.slider('Tumor size', 1.0, 200.0, 26.0, key='tumor_size_slider')

        tumor_stage_options = [1.0, 2.0, 3.0, 4.0]
        tumor_stage = st.sidebar.selectbox('Tumor stage', tumor_stage_options, key='tumor_stage_selectbox')

        overall_survival_months = st.sidebar.text_input("Count of overall survival months", value='50', key='overall_survival_months')

        # Convert inputs to appropriate data types
        
        overall_survival_months = int(overall_survival_months)
        # except ValueError:
        #     st.sidebar.error("Please enter a valid integer for age.")

        features = pd.DataFrame({
            'age_at_diagnosis': [age_at_diagnosis],
            'type_of_breast_surgery': [type_of_breast_surgery],
            'cancer_type_detailed': [cancer_type_detailed],
            'cellularity': [cellularity],
            'chemotherapy': [chemotherapy],
            'pam50_+_claudin-low_subtype': [pam50],
            'cohort': [cohort],
            'er_status_measured_by_ihc': [er_status_measured_by_ihc],
            'er_status': [er_status],
            'neoplasm_histologic_grade': [neoplasm_histologic_grade],
            'her2_status_measured_by_snp6': [her2_status_measured_by_snp6],
            'her2_status': [her2_status],
            'tumor_other_histologic_subtype': [tumor_other_histologic_subtype],
            'hormone_therapy': [hormone_therapy],
            'inferred_menopausal_state': [inferred_menopausal_state],
            'integrative_cluster': [integrative_cluster],
            'primary_tumor_laterality': [primary_tumor_laterality],
            'lymph_nodes_examined_positive': [lymph_nodes_examined_positive],
            'mutation_count': [mutation_count],
            'nottingham_prognostic_index': [nottingham_prognostic_index],
            'oncotree_code': [oncotree_code],
            'pr_status': [pr_status],
            'radio_therapy': [radio_therapy],
            '3-gene_classifier_subtype': [_3gene_classifier_subtype],
            'tumor_size': [tumor_size],
            'tumor_stage': [tumor_stage],
            'overall_survival_months': [overall_survival_months]
        })

        return features
    df = user_input_features()

    # Preprocess user input including PCA
    df_processed = preprocess_input(df)

    # Display processed input
    st.subheader('Processed User Input (after PCA)')
    st.write(df)

    # Predict
    prediction = model_clinical.predict(df_processed)

    # Display prediction
    st.subheader('Prediction')
    if prediction == 0:
        st.write("The model predicts: Patient will not survive.")
    else:
        st.write("The model predicts: Patient will survive.")

    if st.button('Back to Home'):
        st.session_state.page = 'home'

# *************************************************************************************************************************************

# Relevant Features page
def relevant():
    st.title('Breast Cancer Survival Prediction Using Relevant Features')
    def user_input_features():

        age_at_diagnosis = st.sidebar.slider('Age', 0, 150, 50, key='age')

        overall_survival_months = st.sidebar.text_input("Count of overall survival months", value=50, key='overall_survival_months')

        type_of_breast_surgery_options = ['Mastectomy', 'Breast Conserving']
        type_of_breast_surgery = st.sidebar.selectbox('Type of breast surgery', type_of_breast_surgery_options, key='type_of_breast_surgery')

        radio_therapy_options = [0, 1]
        radio_therapy = st.sidebar.selectbox('Radio Therapy', radio_therapy_options, key='radio_therapy')

        inferred_menopausal_state_options = ['Post', 'Pre']
        inferred_menopausal_state = st.sidebar.selectbox('Menopausal State', inferred_menopausal_state_options, key='inferred_menopausal_state')

        tumor_stage_options = [1.0, 2.0, 3.0, 4.0]
        tumor_stage = st.sidebar.selectbox('Tumor stage', tumor_stage_options, key='tumor_stage')

        cohort_options = [1.0, 2.0, 3.0, 4.0, 5.0]
        cohort = st.sidebar.selectbox('Cohort', cohort_options, key='cohort')

        tumor_size = st.sidebar.slider('Tumor size', 1.0, 200.0, 26.0, key='tumor_size')

        lymph_nodes_examined_positive = st.sidebar.slider('Lymph nodes examined positive', 0.0, 45.0, 26.0, key='lymph_nodes_examined_positive')

        hsd17b11 = st.sidebar.text_input("Value for hsd17b11", value=0.2456, key='hsd17b11')
        hsd17b11 = float(hsd17b11)

        cdkn2c = st.sidebar.text_input("Value for cdkn2c", value=-0.4452, key='cdkn2c')
        cdkn2c = float(cdkn2c)

        jak1 = st.sidebar.text_input("Value for jak1", value=-0.3341, key='jak1')
        jak1 = float(jak1)

        spry2 = st.sidebar.text_input("Value for spry2", value=0.1826, key='spry2')
        spry2 = float(spry2)

        lama2 = st.sidebar.text_input("Value for lama2", value=-1.3408, key='lama2')
        lama2 = float(lama2)

        casp8 = st.sidebar.text_input("Value for casp8", value=0.5126, key='casp8')
        casp8 = float(casp8)

        tgfbr2 = st.sidebar.text_input("Value for tgfbr2", value=0.5883, key='tgfbr2')
        tgfbr2 = float(tgfbr2)

        abcb1 = st.sidebar.text_input("Value for abcb1", value=-0.8729, key='abcb1')
        abcb1 = float(abcb1)

        kit = st.sidebar.text_input("Value for kit", value=-0.4208, key='kit')
        kit = float(kit)

        pdgfra = st.sidebar.text_input("Value for pdgfra", value=-0.4208, key='pdgfra')
        pdgfra = float(pdgfra)

        igf1 = st.sidebar.text_input("Value for igf1", value=-0.6965, key='igf1')
        igf1 = float(igf1)

        myc = st.sidebar.text_input("Value for myc", value=-0.8727, key='myc')
        myc = float(myc)

        stat5a = st.sidebar.text_input("Value for stat5a", value=-0.6999, key='stat5a')
        stat5a = float(stat5a)

        smad4 = st.sidebar.text_input("Value for smad4", value=-0.2832, key='smad4')
        smad4 = float(smad4)

        ccnd2 = st.sidebar.text_input("Value for ccnd2", value=-0.2832, key='ccnd2')
        ccnd2 = float(ccnd2)

        rps6 = st.sidebar.text_input("Value for rps6", value=-0.7996, key='rps6')
        rps6 = float(rps6)

        jak2 = st.sidebar.text_input("Value for jak2", value=-0.0428, key='jak2')
        jak2 = float(jak2)

        rheb = st.sidebar.text_input("Value for rheb", value=-0.0428, key='rheb')
        rheb = float(rheb)

        folr2 = st.sidebar.text_input("Value for folr2", value=-0.0428, key='folr2')
        folr2 = float(folr2)

        fas = st.sidebar.text_input("Value for fas", value=-0.0428, key='fas')
        fas = float(fas)

        sf3b1 = st.sidebar.text_input("Value for sf3b1", value=-0.0428, key='sf3b1')
        sf3b1 = float(sf3b1)

        arid5b = st.sidebar.text_input("Value for arid5b", value=-0.0428, key='arid5b')
        arid5b = float(arid5b)

        psen1 = st.sidebar.text_input("Value for psen1", value=-0.0428, key='psen1')
        psen1 = float(psen1)

        syne1 = st.sidebar.text_input("Value for syne1", value=-0.0428, key='syne1')
        syne1 = float(syne1)

        mmp7 = st.sidebar.text_input("Value for mmp7", value=-0.0428, key='mmp7')
        mmp7 = float(mmp7)

        flt3 = st.sidebar.text_input("Value for flt3", value=-0.0428, key='flt3')
        flt3 = float(flt3)

        mapk14 = st.sidebar.text_input("Value for mapk14", value=-0.0428, key='mapk14')
        mapk14 = float(mapk14)

        fgf1 = st.sidebar.text_input("Value for fgf1", value=-0.0428, key='fgf1')
        fgf1 = float(fgf1)

        pms2 = st.sidebar.text_input("Value for pms2", value=-0.0428, key='pms2')
        pms2 = float(pms2)

        nr3c1 = st.sidebar.text_input("Value for nr3c1", value=-0.0428, key='nr3c1')
        nr3c1 = float(nr3c1)

        casp6 = st.sidebar.text_input("Value for casp6", value=-0.0428, key='casp6')
        casp6 = float(casp6)

        eif4e = st.sidebar.text_input("Value for eif4e", value=-0.0428, key='eif4e')
        eif4e = float(eif4e)

        rb1 = st.sidebar.text_input("Value for rb1", value=-0.0428, key='rb1')
        rb1 = float(rb1)

        lamb3 = st.sidebar.text_input("Value for lamb3", value=-0.0422, key='lamb3')
        lamb3 = float(lamb3)

        ran = st.sidebar.text_input("Value for ran", value=-0.0428, key='ran')
        ran = float(ran)

        gata3_mut = st.sidebar.text_input("Value for gata3_mut", value='-0.0428', key='gata3_mut') #string

        adam10 = st.sidebar.text_input("Value for adam10", value=-0.0428, key='adam10')
        adam10 = float(adam10)

        sik1 = st.sidebar.text_input("Value for sik1", value=-0.0428, key='sik1')
        sik1 = float(sik1)

        arid1a = st.sidebar.text_input("Value for arid1a", value=-0.0428, key='arid1a')
        arid1a = float(arid1a)

        bmp6 = st.sidebar.text_input("Value for bmp6", value=-0.0428, key='bmp6')
        bmp6 = float(bmp6)

        zfp36l1 = st.sidebar.text_input("Value for zfp36l1", value=-0.0428, key='zfp36l1')
        zfp36l1 = float(zfp36l1)

        akr1c3 = st.sidebar.text_input("Value for akr1c3", value=-0.0428, key='akr1c3')
        akr1c3 = float(akr1c3)

        ubr5 = st.sidebar.text_input("Value for ubr5", value=-0.0428, key='ubr5')
        ubr5 = float(ubr5)

        bard1 = st.sidebar.text_input("Value for bard1", value=-0.0428, key='bard1')
        bard1 = float(bard1)

        slc19a1 = st.sidebar.text_input("Value for slc19a1", value=-0.0428, key='slc19a1')
        slc19a1 = float(slc19a1)

        dtx3 = st.sidebar.text_input("Value for dtx3", value=-0.0428, key='dtx3')
        dtx3 = float(dtx3)

        map3k13 = st.sidebar.text_input("Value for map3k13", value=-0.0428, key='map3k13')
        map3k13 = float(map3k13)

        fancd2 = st.sidebar.text_input("Value for fancd2", value=-0.0428, key='fancd2')
        fancd2 = float(fancd2)

        notch3 = st.sidebar.text_input("Value for notch3", value=-0.0428, key='notch3')
        notch3 = float(notch3)

        erbb4 = st.sidebar.text_input("Value for erbb4", value=-0.0428, key='erbb4')
        erbb4 = float(erbb4)

        rptor = st.sidebar.text_input("Value for rptor", value=-0.0428, key='rptor')
        rptor =float(rptor)

        mmp11 = st.sidebar.text_input("Value for mmp11", value=-0.0428, key='mmp11')
        mmp11 = float(mmp11)

        ccnb1 = st.sidebar.text_input("Value for ccnb1", value=-0.0428, key='ccnb1')
        ccnb1 = float(ccnb1)

        bcl2l1 = st.sidebar.text_input("Value for bcl2l1", value=-0.0428, key='bcl2l1')
        bcl2l1 = float(bcl2l1)

        akt1s1 = st.sidebar.text_input("Value for akt1s1", value=-0.0428, key='akt1s1')
        akt1s1 = float(akt1s1)

        setd1a = st.sidebar.text_input("Value for setd1a", value=-0.0428, key='setd1a')
        setd1a = float(setd1a)

        smad3 = st.sidebar.text_input("Value for smad3", value=-0.0428, key='smad3')
        smad3 = float(smad3)

        mmp15 = st.sidebar.text_input("Value for mmp15", value=-0.0428, key='mmp15')
        mmp15= float(mmp15)

        stat2 = st.sidebar.text_input("Value for stat2", value=-0.0428, key='stat2')
        stat2 = float(stat2)

        maml1 = st.sidebar.text_input("Value for maml1", value=-0.0428, key='maml1')
        maml1 = float(maml1)

        wwox = st.sidebar.text_input("Value for wwoox", value=-0.0428, key='wwox')
        wwox = float(wwox)

        bap1 = st.sidebar.text_input("Value for bap1", value=-0.0428, key='bap1')
        bap1 = float(bap1)

        rps6kb2 = st.sidebar.text_input("Value for rps6kb2", value=-0.0428, key='rps6kb2')
        rps6kb2 = float(rps6kb2)

        ar = st.sidebar.text_input("Value for ar", value=-0.0428, key='ar')
        ar = float(ar)

        tubb4b = st.sidebar.text_input("Value for tubb4b", value=-0.0428, key='tubb4b')
        tubb4b = float(tubb4b)

        map2k2 = st.sidebar.text_input("Value for map2k2", value=-0.0428, key='map2k2')
        map2k2 = float(map2k2)

        sdc4 = st.sidebar.text_input("Value for sdc4", value=-0.0428, key='sdc4')
        sdc4 = float(sdc4)

        afdn = st.sidebar.text_input("Value for afdn", value=-0.0428, key='afdn')
        afdn = float(afdn)

        akt1 = st.sidebar.text_input("Value for akt1", value=-0.0428, key='akt1')
        akt1 = float(akt1)

        nottingham_prognostic_index = st.sidebar.text_input("Value for Nottingham Prognostic Index", value=-0.0428, key='nottingham_prognostic_index')
        nottingham_prognostic_index = float(nottingham_prognostic_index)

        ncoa3 = st.sidebar.text_input("Value for ncoa3", value=-0.0428, key='ncoa3')
        ncoa3 = float(ncoa3)

        pdgfb = st.sidebar.text_input("Value for pdgfb", value=-0.0428, key='pdgfb')
        pdgfb = float(pdgfb)
        
        tnk2 = st.sidebar.text_input("Value for tnk2", value=-0.0428, key='tnk2')
        tnk2 = float(tnk2)

        tsc2 = st.sidebar.text_input("Value for tsc2", value=-0.0428, key='tsc2')
        tsc2 = float(tsc2)

        map4 = st.sidebar.text_input("Value for map4", value=-0.0428, key='map4')
        map4 = float(map4)

        kmt2c = st.sidebar.text_input("Value for sf3b1", value=-0.0428, key='kmt2c')
        kmt2c = float(kmt2c)

        gsk3b = st.sidebar.text_input("Value for sf3b1", value=-0.0428, key='gsk3b')
        kmt2c = float(kmt2c)

        

        # Convert inputs to appropriate data types
        # overall_survival_months = int(overall_survival_months)
        
        features = pd.DataFrame({
            'overall_survival_months': [overall_survival_months],
            'hsd17b11': [hsd17b11],
            'cdkn2c': [cdkn2c],
            'jak1': [jak1],
            'spry2': [spry2],
            'spry2': [spry2],
            'lama2': [lama2],
            'inferred_menopausal_state': [inferred_menopausal_state],
            'casp8': [casp8],
            'tgfbr2': [tgfbr2],
            'abcb1': [abcb1],
            'kit': [kit],
            'pdgfra': [pdgfra],
            'igf1': [igf1],
            'myc': [myc],
            'stat5a': [stat5a],
            'smad4': [smad4],
            'ccnd2': [ccnd2],
            'rps6': [rps6],
            'jak2': [jak2],
            'rheb': [rheb],
            'folr2': [folr2],
            'fas': [fas],
            'sf3b1': [sf3b1],
            'arid5b': [arid5b],
            'psen1': [psen1],
            'syne1': [syne1],
            'mmp7': [mmp7],
            'flt3': [flt3],
            'mapk14': [mapk14],
            'fgf1': [fgf1],
            'pms2': [pms2],
            'nr3c1': [nr3c1],
            'casp6': [casp6],
            'eif4e': [eif4e],
            'rb1': [rb1],
            'lamb3': [lamb3],
            'radio_therapy': [radio_therapy],
            'ran': [ran],
            'gata3_mut': [gata3_mut],
            'adam10': [adam10],
            'sik1': [sik1],
            'arid1a': [arid1a],
            'bmp6': [bmp6],
            'zfp36l1': [zfp36l1],
            'akr1c3': [akr1c3],
            'ubr5': [ubr5],
            'bard1': [bard1],
            'slc19a1': [slc19a1],
            'dtx3': [dtx3],
            'map3k13': [map3k13],
            'fancd2': [fancd2],
            'notch3': [notch3],
            'erbb4': [erbb4],
            'rptor': [rptor],
            'mmp11': [mmp11],
            'ccnb1': [ccnb1],
            'bcl2l1': [bcl2l1],
            'akt1s1': [akt1s1],
            'setd1a': [setd1a],
            'smad3': [smad3],
            'mmp15': [mmp15],
            'stat2': [stat2],
            'maml1': [maml1],
            'wwox': [wwox],
            'bap1': [bap1],
            'rps6kb2': [rps6kb2],
            'ar': [ar],
            'tubb4b': [tubb4b],
            'map2k2': [map2k2],
            'sdc4': [sdc4],
            'afdn': [afdn],
            'akt1': [akt1],
            'nottingham_prognostic_index': [nottingham_prognostic_index],
            'ncoa3': [ncoa3],
            'pdgfb': [pdgfb],
            'cohort': [cohort],
            'tnk2': [tnk2],
            'tsc2': [tsc2],
            'tumor_size': [tumor_size],
            'lymph_nodes_examined_positive': [lymph_nodes_examined_positive],
            'map4': [map4],
            'tumor_stage': [tumor_stage],
            'kmt2c': [kmt2c],
            'type_of_breast_surgery': [type_of_breast_surgery],
            'gsk3b': [gsk3b],
            'age_at_diagnosis': [age_at_diagnosis],
        })

        return features
    df = user_input_features()

    # Preprocess user input including PCA
    df_processed = preprocess_input(df)

    # Display processed input
    st.subheader('Processed User Input (after PCA)')
    st.write(df)

    # Predict
    prediction = model_relevant.predict(df_processed)

    # Display prediction
    st.subheader('Prediction')
    if prediction == 0:
        st.write("The model predicts: Patient will not survive.")
    else:
        st.write("The model predicts: Patient will survive.")

    if st.button('Back to Home'):
        st.session_state.page = 'home'


# *************************************************************************************************************************************

# Main navigation logic
if 'page' not in st.session_state:
    st.session_state.page = 'home'

if st.session_state.page == 'home':
    home()
elif st.session_state.page == 'clinical':
    clinical()
elif st.session_state.page == 'relevant':
    relevant()
