import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import datetime
#colors style
st.markdown(
    """
    <style>
    .stButton>button {
        background: linear-gradient(90deg, #4a90e2, #357ABD); /* أزرق هادئ */
        color: white;
        border-radius: 8px;
        transition: 0.3s;
    }
    .stButton>button:hover {
        background: linear-gradient(90deg, #5dade2, #2e86c1); /* أزرق أفتح عند المرور */
        transform: scale(1.05);
    }
    </style>
    """,
    unsafe_allow_html=True
)




# -------- إعداد الصفحة --------
st.set_page_config(page_title="Titanic Dashboard", layout="wide")

# -------- تهيئة session_state --------
if "df" not in st.session_state:
    st.session_state.df = None
if "issues" not in st.session_state:
    st.session_state.issues = {}



st.markdown(
    """
    <style>
    .stMetric {
        background-color: #f0f8ff;
        border-radius: 10px;
        padding: 10px;
    }
    .stButton>button {
        background-color: #e74c3c; /* أحمر */
        color: white;
        border-radius: 8px;
    }
    .stDataFrame {
        border: 2px solid #e74c3c; /* أحمر */
        border-radius: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <style>
    body {
        background-color: #fafafa;
    }
    </style>
    """,
    unsafe_allow_html=True
)

#funcations
# -------- Functions --------
def load_data():
    return pd.read_csv("Titanic-Dataset.csv")

def get_issues(df):
    return {
        "duplicates": df.duplicated().sum(),
        "missing_age": df["Age"].isna().sum(),
        "missing_embarked": df["Embarked"].isna().sum()
    }

def remove_duplicates(df):
    return df.drop_duplicates()

def fill_missing_age(df):
    if "Age" in df.columns:
        df["Age"].fillna(df["Age"].median(), inplace=True)
    return df

def fill_missing_embarked(df):
    if "Embarked" in df.columns:
        df["Embarked"].fillna(df["Embarked"].mode()[0], inplace=True)
    return df

def convert_sex_to_numeric(df):
    if "Sex" in df.columns:
        df["Sex"] = df["Sex"].map({"male": 0, "female": 1})
    return df

def encode_pclass(df):
    if "Pclass" in df.columns:
        df = pd.get_dummies(df, columns=["Pclass"], prefix="Class")
    return df

def ensure_fare_numeric(df):
    if "Fare" in df.columns:
        df["Fare"] = pd.to_numeric(df["Fare"], errors="coerce")
    return df

def convert_to_categorical(df):
    
    if "Pclass" in df.columns:
        df["Pclass"] = df["Pclass"].astype("category")
    return df

def drop_columns(df,col_name):
    if col_name in df.columns:
        df = df.drop(columns=[col_name])
    return df

#set two pages

page = st.sidebar.radio("Navigation", ["Clean Data", "Dashboard"])
# =========================================================
# User Instructions (Sidebar)
# =========================================================
st.sidebar.markdown("## 📖 Instructions")

st.sidebar.info("""
1. Go to **Clean Data** first:
   - Click 📂 **Load Data** to import the Titanic dataset.
   - Use the buttons to remove duplicates or fill missing values.
   - Drop unnecessary columns if needed.

2. After cleaning, click 💾 **Save Changes** to store a cleaned version.

3. Switch to **Dashboard**:
   - Apply filters (Embarked, Sex, Pclass, Age).
   - Monitor the KPIs (Passengers, Survival Rate, Average Age, Average Fare).
   - Explore the charts to understand passengers distribution and survival patterns.
""")

# =========================================================
# Welcome Section
# =========================================================
if page not in ["Clean Data", "Dashboard"]:
    st.title("🚢 Welcome to the Titanic Dashboard")
    st.markdown("""
    This application helps you explore and clean the Titanic dataset, and 
    visualize survival statistics with interactive charts.

    ### 🔎 Features:
    - **Data Cleaning:** Handle duplicates, missing values, and column transformations.
    - **Dashboard:** Apply filters, monitor KPIs, and view interactive charts.
    
    👉 Use the sidebar to navigate between **Clean Data** and **Dashboard**.
    """)


# =========================================================
# صفحة تنظيف البيانات
# =========================================================
if page == 'Clean Data':
    # --- العنوان ---
    st.title("🧹 Data Cleaning")

    # --- زر تحميل البيانات ---
    if st.button('📂 Load Data'):
        st.session_state.df = load_data()
        st.session_state.issues = get_issues(st.session_state.df)
        st.success("✅ Data loaded successfully!")

    if st.session_state.df is not None:
        # --- الأزرار ---
        col1, col2, col3, col4, col5, col6 = st.columns(6)

        with col1:
            if st.button('Remove Duplicates'):
                st.session_state.df = remove_duplicates(st.session_state.df)
                st.session_state.issues = get_issues(st.session_state.df)
                st.success("✅ Duplicates removed!")

        with col3:
            if st.button('Convert Sex to Numeric'):
                st.session_state.df = convert_sex_to_numeric(st.session_state.df)
                st.session_state.issues = get_issues(st.session_state.df)
                st.success("✅ 'Sex' column converted to numeric!")

        with col4:
            if st.button('Ensure Fare is Numeric'):
                st.session_state.df = ensure_fare_numeric(st.session_state.df)
                st.session_state.issues = get_issues(st.session_state.df)
                st.success("✅ Fare column converted to numeric!")

        with col5:
            if st.button('Convert to Categorical'):
                st.session_state.df = convert_to_categorical(st.session_state.df)
                st.session_state.issues = get_issues(st.session_state.df)
                st.success("✅ Columns converted to categorical!")

        # --- اختيار العمود + fill/drop ---
        col1, col2, col3 = st.columns([3,1,1])
        with col1:
            selected_col = st.selectbox("Select Column:", st.session_state.df.columns)
        with col2:
            st.write('Fill missing values')
            if st.button('Fill Missing'):
                if st.session_state.df[selected_col].dtype in ['int64', 'float64']:
                    st.session_state.df[selected_col].fillna(
                        st.session_state.df[selected_col].median(), inplace=True)
                else:
                    st.session_state.df[selected_col].fillna(
                        st.session_state.df[selected_col].mode()[0], inplace=True)

                # تحديث القيم
                st.session_state.issues = get_issues(st.session_state.df)
                st.success(f"✅ Missing values in '{selected_col}' filled!")

        with col3:
            st.write('Remove the selected column')
            if st.button("Drop Column"):
                st.session_state.df = drop_columns(st.session_state.df, selected_col)

                # تحديث القيم
                st.session_state.issues = get_issues(st.session_state.df)
                st.success(f"✅ Column '{selected_col}' removed!")

        # --- الكاردز الملونة ---
        st.markdown("### 📊 Data Issues Overview")
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.markdown(f"""
                <div style="background: linear-gradient(90deg, #e74c3c, #c0392b);
                            color: white; padding: 15px; border-radius: 8px; 
                            text-align: center; font-weight: bold; font-size:18px;">
                    Duplicates<br><span style="font-size:24px;">{st.session_state.issues["duplicates"]}</span>
                </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown(f"""
                <div style="background: linear-gradient(90deg, #f39c12, #d35400);
                            color: white; padding: 15px; border-radius: 8px; 
                            text-align: center; font-weight: bold; font-size:18px;">
                    Missing Age<br><span style="font-size:24px;">{st.session_state.issues["missing_age"]}</span>
                </div>
            """, unsafe_allow_html=True)

        with col3:
            st.markdown(f"""
                <div style="background: linear-gradient(90deg, #3498db, #2980b9);
                            color: white; padding: 15px; border-radius: 8px; 
                            text-align: center; font-weight: bold; font-size:18px;">
                    Missing Embarked<br><span style="font-size:24px;">{st.session_state.issues["missing_embarked"]}</span>
                </div>
            """, unsafe_allow_html=True)

        with col4:
            st.markdown(f"""
                <div style="background: linear-gradient(90deg, #2ecc71, #27ae60);
                            color: white; padding: 15px; border-radius: 8px; 
                            text-align: center; font-weight: bold; font-size:18px;">
                    Sex state<br><span style="font-size:24px;">Updated</span>
                </div>
            """, unsafe_allow_html=True)

        # --- عرض الجدول ---
        st.subheader("📋 Current Data")
        st.dataframe(st.session_state.df, height=400)

        # --- زر حفظ ---
        sal1, sal2, sal3 = st.columns([1, 2, 1])
        with sal1:
            if st.button("💾 Save Changes"):
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                saved_filename = f"Titanic-Cleaned-{timestamp}.csv"

                st.session_state.df.to_csv(saved_filename, index=False)
                st.session_state["saved_df"] = st.session_state.df.copy()

                st.success(f"✅ Changes saved successfully as '{saved_filename}'!")

 


# =========================================================
# صفحة الداشبورد
# =========================================================
elif page == "Dashboard":
    st.title("📊 Dashboard")

    @st.cache_data
    def load_data():
        return pd.read_csv("Titanic_cleaning_Data.csv")

    # إذا فيه بيانات محفوظة نستخدمها
    if "saved_df" in st.session_state and st.session_state["saved_df"] is not None:
      df_raw = st.session_state["saved_df"]
      df = df_raw.copy()
      st.info("📂 Using the cleaned dataset (saved).")
    else:
      df_raw = load_data()
      df = df_raw.copy()
      st.warning("⚠️ No saved cleaned data found, using raw dataset.")
 


    st.title("🚢 Titanic Dashboard")
                                      
    # -------- الفلاتر --------
    st.markdown("### Filters")
    f1, f2, f3, f4 = st.columns(4)

    with f1:
        if "Embarked" in df.columns:
            embarked_vals = ["All"] + sorted([x for x in df["Embarked"].dropna().unique()])
            embarked_sel = st.selectbox("Embarked (Port)", embarked_vals, index=0)
            if embarked_sel != "All":
                df = df[df["Embarked"] == embarked_sel]

    with f2:
        if "Sex" in df.columns:
            if df["Sex"].dtype in [np.int64, np.float64]:
                df["Sex"] = df["Sex"].map({0: "Female", 1: "Male"})
            sexes = ["Male", "Female"]
            sex_sel = st.multiselect("Sex", options=sexes, default=sexes)
            if sex_sel:
                df = df[df["Sex"].isin(sex_sel)]

    with f3:
        if "Pclass" in df.columns:
            classes = sorted(df["Pclass"].dropna().unique())
            pclass_sel = st.multiselect("Passenger Class (Pclass)", options=classes, default=classes)
            if pclass_sel:
                df = df[df["Pclass"].isin(pclass_sel)]

    with f4:
        if "Age" in df.columns:
            min_age = int(np.floor(df_raw["Age"].dropna().min())) if df_raw["Age"].notna().any() else 0
            max_age = int(np.ceil(df_raw["Age"].dropna().max())) if df_raw["Age"].notna().any() else 80
            age_range = st.slider("Age Range", min_age, max_age, (min_age, max_age))
            df = df[(df["Age"].between(age_range[0], age_range[1])) | (df["Age"].isna())]

    st.markdown("---")

    # -------- KPIs --------
    k1, k2, k3, k4, k5 = st.columns([1,1,1,1,1])
    total_passengers = len(df)
    k1.metric("Number of Passengers", total_passengers)

    if "Survived" in df.columns and total_passengers > 0:
        surv_rate = round(df["Survived"].mean() * 100, 2)
        k2.metric("Survival Rate", f"{surv_rate}%")
    else:
        k2.metric("Survival Rate", "-")

    if "Age" in df.columns and df["Age"].notna().any():
        k3.metric("Average Age", round(df["Age"].mean(), 1))
    else:
        k3.metric("Average Age", "-")

    if "Fare" in df.columns and df["Fare"].notna().any():
        k4.metric("Average Fare", f"${round(df['Fare'].mean(), 2)}")
    else:
        k4.metric("Average Fare", "-")

    k5.metric("Embarked Filter", embarked_sel if "embarked_sel" in locals() else "All")

    st.markdown("---")

    # -------- الرسومات --------
    c1, c2 = st.columns(2)

    if {"Pclass"}.issubset(df.columns) and len(df) > 0:
        c1.subheader("Number of Passengers by Class")
        by_class = df.groupby("Pclass").size().reset_index(name="Count")
        chart_class = (
            alt.Chart(by_class, title="Passengers by Pclass")
            .mark_bar()
            .encode(
                x=alt.X("Pclass:O", title="Pclass"),
                y=alt.Y("Count:Q", title="Count"),
                color=alt.Color("Pclass:O", scale=alt.Scale(scheme="set2")),
                tooltip=["Pclass","Count"]
            )
            .properties(height=350)
        )
        c1.altair_chart(chart_class, use_container_width=True)

    if {"Sex","Survived"}.issubset(df.columns) and len(df) > 0:
        c2.subheader("Average Survival Rate by Sex")
        by_sex = df.groupby("Sex")["Survived"].mean().reset_index()
        by_sex["Survival Rate (%)"] = (by_sex["Survived"] * 100).round(2)
        chart_sex = (
            alt.Chart(by_sex, title="Survival Rate by Sex")
            .mark_bar()
            .encode(
                y=alt.Y("Sex:N", title="Sex"),
                x=alt.X("Survival Rate (%):Q", title="Survival Rate (%)"),
                color=alt.Color("Sex:N", scale=alt.Scale(domain=["Male","Female"], range=["#1f77b4","#ff69b4"])),
                tooltip=["Sex","Survival Rate (%)"]
            )
            .properties(height=350)
        )
        c2.altair_chart(chart_sex, use_container_width=True)

    c3, c4 = st.columns(2)

    if {"Age","Survived"}.issubset(df.columns) and df["Age"].notna().any():
        c3.subheader("Survival Rate by Age Group")
        bins = list(range(0,81,5))
        age_binned = pd.cut(df["Age"], bins=bins, include_lowest=True)
        surv_by_agebin = df.groupby(age_binned)["Survived"].mean().reset_index()
        surv_by_agebin["Survival Rate (%)"] = (surv_by_agebin["Survived"] * 100).round(2)
        surv_by_agebin["Age Band"] = surv_by_agebin["Age"].astype(str)

        line_age = (
            alt.Chart(surv_by_agebin, title="Survival Rate across Age Bands")
            .mark_line(point=True, color="#ff7f0e")
            .encode(
                x=alt.X("Age Band:N", title="Age Band (years)"),
                y=alt.Y("Survival Rate (%):Q", title="Survival Rate (%)"),
                tooltip=["Age Band","Survival Rate (%)"]
            )
            .properties(height=350)
        )
        c3.altair_chart(line_age, use_container_width=True)

    if {"Fare","Pclass"}.issubset(df.columns) and df["Fare"].notna().any():
        c4.subheader("Average Fare by Class")
        fare_by_class = df.groupby("Pclass")["Fare"].mean().reset_index()
        fare_by_class["Average Fare"] = fare_by_class["Fare"].round(2)
        line_fare = (
            alt.Chart(fare_by_class, title="Average Fare by Pclass")
            .mark_line(point=True, color="#2ca02c")
            .encode(
                x=alt.X("Pclass:O", title="Pclass"),
                y=alt.Y("Average Fare:Q", title="Average Fare ($)"),
                tooltip=["Pclass","Average Fare"]
            )
            .properties(height=350)
        )
        c4.altair_chart(line_fare, use_container_width=True)
