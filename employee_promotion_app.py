import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.preprocessing import MinMaxScaler

# Page Configuration
st.set_page_config(
    page_title="Employee Promotion Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)


# Custom CSS to AGGRESSIVELY reduce spacing
st.markdown("""
    <style>
        /* Remove ALL top padding from main content */
        .block-container {
            padding-top: 0rem !important;
            padding-bottom: 0rem !important;
        }
        
        /* Remove ALL top padding from sidebar */
        section[data-testid="stSidebar"] {
            padding-top: 0rem !important;
        }
        
        section[data-testid="stSidebar"] > div {
            padding-top: 0rem !important;
        }
        
        /* Remove space above sidebar image */
        section[data-testid="stSidebar"] img {
            margin-top: 0rem !important;
            padding-top: 0rem !important;
        }
        
        /* Remove header bar space */
        header[data-testid="stHeader"] {
            display: none;
        }
        
        /* Compact title */
        h1 {
            margin-top: 0rem !important;
            padding-top: 0rem !important;
        }
        
        /* Remove space before subheader */
        h2, h3 {
            margin-top: 0.5rem !important;
            padding-top: 0rem !important;
        }
    </style>
""", unsafe_allow_html=True)


# Load dataset
df = pd.read_csv('Employee_Promotion_Cleaned.csv')

# Title
st.title("Employee Promotion Analysis & Prediction")

# Tabs
tab1, tab2, tab3 = st.tabs(['Executive Overview 📊', 'Demographics & Organization 👥', 'ML Predictions 🤖'])



##_________________________________________________________________________

# Sidebar
st.sidebar.title("**🏢 Employee Insights**")
st.sidebar.image("Employee Promotion.jpg")
st.sidebar.markdown("**This interactive dashboard provides comprehensive analysis of employee promotion patterns using machine learning and data analytics.**")
st.sidebar.divider()

# Initialize reset counter in session state
if 'filter_reset_counter' not in st.session_state:
    st.session_state.filter_reset_counter = 0

# Use the counter to create unique keys that force widget recreation
reset_suffix = st.session_state.filter_reset_counter

# Filters with dynamic keys that change on reset
selected_departments = st.sidebar.multiselect(
    "Select Department(s)",
    options=df['department'].unique().tolist(),
    default=[],
    key=f'departments_select_{reset_suffix}'
)

selected_gender = st.sidebar.multiselect(
    "Select Gender",
    options=df['gender'].unique().tolist(),
    default=[],
    key=f'gender_select_{reset_suffix}'
)

selected_education = st.sidebar.multiselect(
    "Select Education Level",
    options=df['education'].unique().tolist(),
    default=[],
    key=f'education_select_{reset_suffix}'
)

age_range = st.sidebar.slider(
    "Age Range",
    min_value=int(df['age'].min()),
    max_value=int(df['age'].max()),
    value=(int(df['age'].min()), int(df['age'].max())),
    key=f'age_slider_{reset_suffix}'
)

# Reset button - Increment counter to force new widget keys
if st.sidebar.button("🔄 Reset Filters", key='reset_btn'):
    st.session_state.filter_reset_counter += 1
    st.rerun()

st.sidebar.divider()
st.sidebar.markdown("Made with [Eng.Abdallah Ibrahim❤️](https://www.linkedin.com/in/abdallah-ibrahim-4556792a5/)")


# Apply Filters - Only filter if user made selections
filtered_df = df.copy()

if selected_departments:
    filtered_df = filtered_df[filtered_df['department'].isin(selected_departments)]

if selected_gender:
    filtered_df = filtered_df[filtered_df['gender'].isin(selected_gender)]

if selected_education:
    filtered_df = filtered_df[filtered_df['education'].isin(selected_education)]

# Age filter always applies
filtered_df = filtered_df[
    (filtered_df['age'] >= age_range[0]) & 
    (filtered_df['age'] <= age_range[1])
]





# ### _______________________________________________________________________


# Tab 1 Content
with tab1:
    st.subheader("📊 Key Performance Indicators")

    # Metrics
    c1, c2, c3, c4  = st.columns(4)
    with c1:
        total_employee = len(filtered_df)
        st.metric(label="**Total Employees**", value=total_employee)

    with c2:
        promotion_rate = (filtered_df['is_promoted_encoded'].sum() / total_employee ) * 100
        st.metric(label= "**Promotion Rate %**" , value=f"{promotion_rate:.1f}%") 

    with c3:
        avg_age = filtered_df['age'].mean()
        st.metric(label="**AVG Age**" , value=f"{avg_age:.1f}") 

    with c4:
        # Top performing department by promotion rate
        dept_promo = filtered_df.groupby('department')['is_promoted_encoded'].mean()
        top_dept = dept_promo.idxmax()
        top_dept_rate = dept_promo.max() * 100
        st.metric(
            label="**Top Department by Promotion Rate**", 
            value=top_dept, 
        )

    ## Charts
    st.markdown("---")
    st.subheader("📈 Promotion Analytics")


    ##_______________________________________________________________________________
    # Row 1: Graphs 1 & 2
    chart1, chart2 = st.columns(2)

    # Graph 1: Simple Donut Chart
    with chart1:
        fig, ax = plt.subplots(figsize=(6, 4))
        
        promo_counts = filtered_df['is_promoted_encoded'].value_counts()
        ax.pie(promo_counts.values, labels=['Not Promoted', 'Promoted'], autopct='%1.1f%%', pctdistance=0.79)

        # Add center circle for donut effect
        centre_circle = plt.Circle((0, 0), 0.70, fc='white')
        ax.add_artist(centre_circle)

        ax.set_title('Promotion Distribution')
        
        st.pyplot(fig)
        plt.close()


    # Graph 2: Bar Chart
    with chart2:
        fig, ax = plt.subplots(figsize=(6, 4))
        
        dept_promo = filtered_df.groupby('department')['is_promoted_encoded'].mean() * 100
        dept_promo = dept_promo.sort_values()
        
        dept_promo.plot(kind='barh', ax=ax)
        ax.set_xlabel('Promotion Rate (%)')
        ax.set_title('Promotion Rate by Department')
        
        st.pyplot(fig)
        plt.close()
    

    ##_______________________________________________________________________________
    # Row 2: Graphs 3, 4 & 5
    chart3, chart4, chart5 = st.columns(3)

    # Graph 3:  Box Plot - Training Score
    with chart3:
        fig, ax = plt.subplots(figsize=(5, 4))
        
        data_to_plot = [
            filtered_df[filtered_df['is_promoted_encoded']==0]['avg_training_score'],
            filtered_df[filtered_df['is_promoted_encoded']==1]['avg_training_score']
        ]
        
        ax.boxplot(data_to_plot)
        ax.set_xticklabels(['Not Promoted', 'Promoted'])
        ax.set_ylabel('Avg Training Score')
        ax.set_title('Training Score by Promotion')
        
        st.pyplot(fig)
        plt.close()

    # Graph 4:  Bar Chart - Rating vs Promotion
    with chart4:
        fig, ax = plt.subplots(figsize=(5, 4))
        
        rating_promo = filtered_df.groupby('previous_year_rating')['is_promoted_encoded'].mean() * 100
        
        rating_promo.plot(kind='bar', ax=ax, color='steelblue')
        ax.set_xlabel('Previous Year Rating')
        ax.set_ylabel('Promotion Rate (%)')
        ax.set_title('Promotion Rate by Rating')
        ax.set_xticklabels(rating_promo.index, rotation=0)
        
        st.pyplot(fig)
        plt.close()

    # Graph 5: Simple Histogram - Age Distribution
    with chart5:
        fig, ax = plt.subplots(figsize=(5, 4))
        
        filtered_df[filtered_df['is_promoted_encoded']==0]['age'].hist(ax=ax, bins=15, alpha=0.6, label='Not Promoted')
        filtered_df[filtered_df['is_promoted_encoded']==1]['age'].hist(ax=ax, bins=15, alpha=0.6, label='Promoted')
        
        ax.set_xlabel('Age')
        ax.set_ylabel('Count')
        ax.set_title('Age Distribution by Promotion')
        ax.legend()
        
        st.pyplot(fig)
        plt.close()




### _______________________________________________________________________

# Tab 2 Content

with tab2:
    st.subheader("👥 Who Gets Promoted? Demographic Patterns")

    # Metrics
    c1,c2,c3,c4 = st.columns(4)

    with c1:
        no_of_departments = filtered_df['department'].nunique()
        st.metric(label = "**Total Departments**" , value=no_of_departments)

    with c2:
        # Gender with highest promotion rate
        gender_promo = filtered_df.groupby('gender')['is_promoted_encoded'].mean() * 100
        top_gender = gender_promo.idxmax()
        top_gender_rate = gender_promo.max()
        st.metric(
            label="**Top Gender (Promo Rate)**", 
            value=f"{top_gender.upper()}: {top_gender_rate:.1f}%"
        )

    with c3:
        # Best Education Level
        edu_promo = filtered_df.groupby('education')['is_promoted_encoded'].mean() * 100
        top_edu = edu_promo.idxmax()
        top_edu_rate = edu_promo.max()
        st.metric(
            label="**Best Education Level**", 
            value=f"{top_edu_rate:.1f}%",
            delta=top_edu
        )

    with c4:
        # Best Recruitment Channel
        channel_promo = filtered_df.groupby('recruitment_channel')['is_promoted_encoded'].mean() * 100
        top_channel = channel_promo.idxmax()
        top_channel_rate = channel_promo.max()
        st.metric(
            label="**Best Recruitment Channel**", 
            value=f"{top_channel_rate:.1f}%",
            delta=top_channel
        )




    ## Charts
    st.markdown("---")

    ##_______________________________________________________________________________
    # Row 1: Graphs 1 & 2
    chart1 , chart2 = st.columns(2)

    # Chart 1: Gender & Education Promotion Comparison (Grouped Bar)
    with chart1:
        fig, ax = plt.subplots(figsize=(7, 5))
        
        # Prepare data
        gender_edu_promo = filtered_df.groupby(['gender', 'education'])['is_promoted_encoded'].mean() * 100
        gender_edu_promo = gender_edu_promo.unstack()

        gender_edu_promo.plot(kind='bar', ax=ax)

        # Increase font sizes
        ax.set_xlabel('Gender', fontsize=14)
        ax.set_ylabel('Promotion Rate (%)', fontsize=14)
        ax.set_title('Promotion Rate by Gender & Education', fontsize=16)
        
        # FIXED: Get actual index labels dynamically
        current_labels = [str(x).upper() for x in gender_edu_promo.index]
        ax.set_xticklabels(current_labels, rotation=0, fontsize=12)
        ax.tick_params(axis='y', labelsize=12)
        ax.legend(title='Education', bbox_to_anchor=(1.05, 1), fontsize=11, title_fontsize=12)
        
        st.pyplot(fig)
        plt.close()
        
    # Chart 2: Regional Promotion Analysis (Horizontal Bar)
    with chart2:
        fig, ax = plt.subplots(figsize=(7, 5))
        
        region_promo = filtered_df.groupby('region')['is_promoted_encoded'].mean().head() * 100
        region_promo = region_promo.sort_values()
        
        region_promo.plot(kind='barh', ax=ax, color='teal')
        ax.set_xlabel('Promotion Rate (%)')
        ax.set_ylabel('Region')
        ax.set_title('Top 5 Promotion Rate by Region')
        
        st.pyplot(fig)
        plt.close()    

    ##_______________________________________________________________________________
    # Row 2: 3 Compact Charts
    chart3, chart4, chart5 = st.columns(3)


    # Chart 3: Recruitment Channel Promotion Rate (Donut)
    with chart3:
        fig, ax = plt.subplots(figsize=(5, 4))
        
        # Calculate promotion rate per channel
        channel_stats = filtered_df.groupby('recruitment_channel')['is_promoted_encoded'].agg(['sum', 'count'])
        channel_stats['rate'] = (channel_stats['sum'] / channel_stats['count'] * 100).round(1)
        
        # Create donut with total employees per channel
        ax.pie(channel_stats['count'].values, 
            labels=[f"{idx}\n({channel_stats.loc[idx, 'rate']:.1f}%)" for idx in channel_stats.index],
            autopct='%1.1f%%', 
            pctdistance=0.85)
        
        # Add center circle for donut effect
        centre_circle = plt.Circle((0, 0), 0.70, fc='white')
        ax.add_artist(centre_circle)
        
        ax.set_title('Employee Distribution by Recruitment Channel')
        
        st.pyplot(fig)
        plt.close()

    # Chart 4: Length of Service Distribution (Histogram)
    with chart4:
        fig, ax = plt.subplots(figsize=(5, 4))
        
        filtered_df[filtered_df['is_promoted_encoded']==0]['length_of_service'].hist(ax=ax, bins=15, alpha=0.6, label='Not Promoted')
        filtered_df[filtered_df['is_promoted_encoded']==1]['length_of_service'].hist(ax=ax, bins=15, alpha=0.6, label='Promoted')
        
        ax.set_xlabel('Years of Service')
        ax.set_ylabel('Count')
        ax.set_title('Service Length by Promotion')
        ax.legend()
        
        st.pyplot(fig)
        plt.close()

    # Chart 5: Department Size vs Promotion Rate (Scatter)
    with chart5:
        fig, ax = plt.subplots(figsize=(5, 4))
        
        dept_stats = filtered_df.groupby('department').agg({
            'is_promoted_encoded': 'mean',
            'employee_id': 'count'
        })
        dept_stats.columns = ['promo_rate', 'size']
        dept_stats['promo_rate'] = dept_stats['promo_rate'] * 100
        
        ax.scatter(dept_stats['size'], dept_stats['promo_rate'], s=100, alpha=0.6, color='purple')
        ax.set_xlabel('Department Size')
        ax.set_ylabel('Promotion Rate (%)')
        ax.set_title('Dept Size vs Promotion Rate')
        
        st.pyplot(fig)
        plt.close()




###_______________________________________________________###
         ## Predictin Model 

with tab3:
    # Header
    st.markdown("### Employee Details")
    
    # Row 1 - Department (left), Region (right)
    col1, col2 = st.columns(2)
    
    with col1:
        department = st.selectbox(
            "Department",
            options=['Sales & Marketing', 'Operations', 'Technology', 'Analytics', 
                    'R&D', 'Procurement', 'Finance', 'HR', 'Legal'],
            key="dept"
        )
    
    with col2:
        region = st.selectbox(
            "Region",
            options=[f'region_{i}' for i in range(1, 35)],
            key="region"
        )
    
    # Row 2 - Education (left), Gender (right)
    col1, col2 = st.columns(2)
    
    with col1:
        education = st.selectbox(
            "Education Level",
            # CORRECTED: Exact options from your dataset
            options=["Bachelor's", "Master's & above", "Below Secondary"],
            key="edu"
        )
    
    with col2:
        gender = st.selectbox(
            "Gender",
            options=['Male', 'Female'],
            key="gender"
        )
    
    # Row 3 - Recruitment Channel (left), No. of Trainings (right)
    col1, col2 = st.columns(2)
    
    with col1:
        recruitment_channel = st.selectbox(
            "Recruitment Channel",
            options=['sourcing', 'other', 'referred'],
            key="recruit"
        )
    
    with col2:
        no_of_trainings = st.number_input(
            "No. of Trainings",
            min_value=1,
            max_value=10,
            value=1,
            key="trainings"
        )
    
    # Row 4 - Age (left), Previous Year Rating (right)
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.number_input(
            "Age",
            min_value=20,
            max_value=60,
            value=30,
            key="age"
        )
    
    with col2:
        previous_year_rating = st.selectbox(
            "Previous Year Rating",
            options=[1.0, 2.0, 3.0, 4.0, 5.0],
            key="rating"
        )
    
    # Row 5 - Length of Service
    length_of_service = st.number_input(
        "Length of Service (years)",
        min_value=1,
        max_value=35,
        value=5,
        key="service"
    )

    # Submit button
    btn = st.button("Predict Promotion", type="primary")


    # Launching the App
    if btn:
        try:
            # Load model and scaler
            model = joblib.load('best_model.pkl')
            scaler = joblib.load('scaler.pkl')
            
            # Create encoding mappings
            department_mapping = {
                'Analytics': 0,
                'Finance': 1,
                'HR': 2,
                'Legal': 3,
                'Operations': 4,
                'Procurement': 5,
                'R&D': 6,
                'Sales & Marketing': 7,
                'Technology': 8
            }
            
            region_mapping = {
                'region_1': 0, 'region_10': 1, 'region_11': 2, 'region_12': 3,
                'region_13': 4, 'region_14': 5, 'region_15': 6, 'region_16': 7,
                'region_17': 8, 'region_18': 9, 'region_19': 10, 'region_2': 11,
                'region_20': 12, 'region_21': 13, 'region_22': 14, 'region_23': 15,
                'region_24': 16, 'region_25': 17, 'region_26': 18, 'region_27': 19,
                'region_28': 20, 'region_29': 21, 'region_3': 22, 'region_30': 23,
                'region_31': 24, 'region_32': 25, 'region_33': 26, 'region_34': 27,
                'region_4': 28, 'region_5': 29, 'region_6': 30, 'region_7': 31,
                'region_8': 32, 'region_9': 33
            }
            
            education_mapping = {
                "Bachelor's": 0,
                "Below Secondary": 1,      
                "Master's & above": 2    
            }
            
            gender_mapping = {
                'Female': 0,
                'Male': 1
            }
            
            recruitment_mapping = {
                'other': 0,
                'referred': 1,
                'sourcing': 2
            }
            
            # Encode categorical variables
            department_encoded = department_mapping.get(department, 0)
            region_encoded = region_mapping.get(region, 0)
            education_encoded = education_mapping.get(education, 0)
            gender_encoded = gender_mapping.get(gender, 0)
            recruitment_channel_encoded = recruitment_mapping.get(recruitment_channel, 0)
            
            # Create array with ONLY the 9 features that need scaling
            features_to_scale = np.array([[
                department_encoded,           # 1
                region_encoded,               # 2
                education_encoded,            # 3
                gender_encoded,               # 4
                recruitment_channel_encoded,  # 5
                no_of_trainings,              # 6
                age,                          # 7
                previous_year_rating,         # 8
                length_of_service             # 9
            ]])
            
            # Scale ONLY these 9 features
            features_scaled = scaler.transform(features_to_scale)
            
            # Create final input with ALL 10 features:
            # 9 scaled features + 1 unscaled feature (avg_training_score)
            input_data_final = np.array([[
                features_scaled[0, 0],   # department_encoded (scaled)
                features_scaled[0, 1],   # region_encoded (scaled)
                features_scaled[0, 2],   # education_encoded (scaled)
                features_scaled[0, 3],   # gender_encoded (scaled)
                features_scaled[0, 4],   # recruitment_channel_encoded (scaled)
                features_scaled[0, 5],   # no_of_trainings (scaled)
                features_scaled[0, 6],   # age (scaled)
                features_scaled[0, 7],   # previous_year_rating (scaled)
                features_scaled[0, 8]   # length_of_service (scaled)
            ]])
            
            # Make prediction
            prediction = model.predict(input_data_final)[0]
            prediction_proba = model.predict_proba(input_data_final)[0]
            

            
            # Show prediction
            if prediction == 1:
                st.success(f"**PROMOTED**")
                st.write("")
            else:
                st.error(f"**NOT PROMOTED**")
                st.write("")
        except Exception as e:
            st.error(f"❌ Error making prediction: {str(e)}")
            
