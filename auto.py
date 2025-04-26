import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import trimesh
import plotly.graph_objects as go
import seaborn as sb
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns  # For correlation heatmap
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, mean_squared_error
from scipy.stats import zscore  # For anomaly detection
import io
import base64
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure
import certifi

# ------------------ Helper Function for Report Sections ------------------
# Modified to store head and body separately
def set_report_section(key, title, head, body, sec_type):
    """
    Save/update a visualization section in session_state using a unique key.
    Both the head (styles and scripts) and the body (the figure's HTML) are stored.
    """
    if "report_sections" not in st.session_state:
        st.session_state["report_sections"] = {}
    st.session_state["report_sections"][key] = {
        "title": title,
        "head": head,
        "body": body,
        "type": sec_type
    }

# ------------------ Helper to extract full HTML (head and body) from a Plotly figure ------------------
def extract_full_fig_html(fig):
    """
    Returns a tuple (head, body) by generating the full HTML for a Plotly figure.
    Before generating the HTML, we force the figure to use the light template,
    so that the report will show visuals in their original colors.
    """
    fig.update_layout(template="plotly")
    full_html = fig.to_html(full_html=True, include_plotlyjs='cdn')
    # Remove DOCTYPE
    full_html = full_html.replace("<!DOCTYPE html>", "").strip()
    head = ""
    body = full_html
    if "<head>" in full_html and "</head>" in full_html:
        head = full_html.split("<head>")[1].split("</head>")[0]
    if "<body>" in full_html and "</body>" in full_html:
        body = full_html.split("<body>")[1].split("</body>")[0]
    return head, body

# ------------------ Login/Registration Functions ------------------
from passlib.hash import bcrypt
from pymongo import MongoClient
def load_users():
    try:
       return pd.read_csv("users.csv")
    except FileNotFoundError:
         return pd.DataFrame(columns=["username", "password_hash"])

def register_user(username, password):
    users_df = load_users()
    if not username or not password:
        return False
    if username in users_df["username"].values:
        return False
    hashed_password = bcrypt.hash(password)
    new_user = pd.DataFrame([[username, hashed_password]], columns=["username", "password_hash"])
    users_df = pd.concat([users_df, new_user], ignore_index=True)
    users_df.to_csv("users.csv", index=False)
    return True

def authenticate_user(username, password):
     users_df = load_users()
     user_row = users_df[users_df["username"] == username]
     if user_row.empty:
         return False
     hashed_password = user_row.iloc[0]["password_hash"]
#     return bcrypt.verify(password, hashed_password)



client = MongoClient("mongodb+srv://gitesh:12345@cluster0.svqxx.mongodb.net/?retryWrites=true&w=majority&appName=Cluster00")
db = client["Autodash"]
user_collection = db["Users"]


def register_user(username, password):
    if not username or not password:
        return False
    # Check if the user already exists
    if user_collection.find_one({"username": username}):
        return False
    # Hash the password and insert a new user document
    hashed_password = bcrypt.hash(password)
    user_doc = {"username": username, "password_hash": hashed_password}
    user_collection.insert_one(user_doc)
    return True



def authenticate_user(username, password):
    user_doc = user_collection.find_one({"username": username})
    if not user_doc:
        return False
    hashed_password = user_doc.get("password_hash")
    return bcrypt.verify(password, hashed_password)




def sidebar_login():
    with st.sidebar.expander("🔐 Login / Register", expanded=True):
        if "authenticated" not in st.session_state:
            st.session_state.authenticated = False

        if st.session_state.authenticated:
            st.write(f"Welcome, **{st.session_state.username}**!")
            if st.button("🚪 Logout", key="logout"):
                st.session_state.authenticated = False
                st.session_state.pop("username", None)
                try:
                    st.rerun()
                except AttributeError:
                    pass
        else:
            auth_choice = st.radio("Choose an option", ["Login", "Register"], horizontal=True, key="auth_choice")
            login_username = st.text_input("Username", key="login_username")
            login_password = st.text_input("Password", type="password", key="login_password")
            if auth_choice == "Register":
                login_password_confirm = st.text_input("Confirm Password", type="password", key="login_password_confirm")
                if st.button("Register", key="register"):
                    if login_password != login_password_confirm:
                        st.error("Passwords do not match")
                    elif register_user(login_username, login_password):
                        st.success("Registration successful! Please login.")
                    else:
                        st.error("Registration failed – username may already exist")
            else:
                if st.button("Login", key="login"):
                    if authenticate_user(login_username, login_password):
                        st.session_state.username = login_username
                        st.session_state.authenticated = True
                        try:
                            st.rerun()
                        except AttributeError:
                            pass
                    else:
                        st.error("Invalid username or password")
# ------------------ End Login Functions ------------------

# Load dataset
def load_data(uploaded_file):
    try:
        if uploaded_file.name.endswith('.csv'):
            return pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith(('.xls', '.xlsx')):
            return pd.read_excel(uploaded_file)
    except Exception as e:
        st.error(f"Error loading file: {str(e)}")
    return None

# Clean dataset
def clean_data(df):
    try:
        df.columns = [col.lower().replace(' ', '_') for col in df.columns]
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                df[col] = df[col].fillna(df[col].median())
            else:
                df[col] = df[col].fillna('Unknown')
        df_cleaned = df.drop_duplicates()
        if df_cleaned.empty:
            st.warning("Cleaned data is empty after removing duplicates and filling missing values.")
        return df_cleaned
    except Exception as e:
        st.error(f"Error during cleaning: {str(e)}")
        return df

# Sample dataset for demo
def get_sample_data():
    return pd.DataFrame({
        "category": ["A", "B", "A", "C", "B", "C"],
        "state_code": ["TX", "CA", "NY", "TX", "CA", "NY"],
        "values": np.random.randint(1, 100, 6),
        "date": pd.date_range(start='1/1/2023', periods=6)
    })

# Toggle Dark/Light Mode
def toggle_theme():
    return "light" if st.session_state.get("theme", "light") == "dark" else "dark"

# 3D Tactile Data Sculpture Feature
def generate_3d_sculpture(data, output_file="data_sculpture.stl"):
    x = np.linspace(0, 10, len(data))
    y = np.linspace(0, 10, len(data))
    X, Y = np.meshgrid(x, y)
    
    Z = np.sin(X) + np.cos(Y) * data

    fig = go.Figure(data=[go.Surface(
        x=X, 
        y=Y, 
        z=Z,
        colorscale='Viridis',
        opacity=0.9,
        colorbar=dict(title="Height"),
        contours={"z": {"show": True, "color": "black"}}
    )])
    fig.update_layout(title='3D Tactile Data Sculpture')
    st.plotly_chart(fig, use_container_width=True)

    head_content, body_content = extract_full_fig_html(fig)
    set_report_section("3d_sculpture", "3D Tactile Data Sculpture", head_content, body_content, "plotly")
        
    vertices = np.array([X.flatten(), Y.flatten(), Z.flatten()]).T
    faces = []
    for i in range(len(x) - 1):
        for j in range(len(y) - 1):
            faces.append([i * len(y) + j, i * len(y) + j + 1, (i + 1) * len(y) + j])
            faces.append([(i + 1) * len(y) + j, i * len(y) + j + 1, (i + 1) * len(y) + j + 1])
    
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    mesh.export(output_file)
    st.write(f"3D sculpture saved as {output_file}")

def main():
    st.set_page_config(page_title="AutoDash", page_icon="📊", layout="wide")
    
    sidebar_login()
    
    st.sidebar.title("📌 About AutoDash")
    st.sidebar.info("AutoDash automatically cleans your dataset and provides insightful visualizations.")
    st.sidebar.subheader("🔧 Features")
    st.sidebar.checkbox("Enable Data Cleaning", value=True)
    st.sidebar.checkbox("Enable AI Insights", value=True)

    if "theme" not in st.session_state:
        st.session_state["theme"] = "light"
    if st.sidebar.button("🌗 Toggle Dark/Light Mode"):
        st.session_state["theme"] = toggle_theme()
    
    if st.session_state["theme"] == "dark":
        st.markdown(
            """
            <style>
            html, body, .stApp, [data-testid="stAppViewContainer"], [data-testid="stSidebar"] {
                background-color: #0e1117 !important;
                color: white !important;
            }
            * { color: white !important; }
            input, textarea, select, button {
                background-color: #0e1117 !important;
                color: white !important;
                border: 1px solid white !important;
            }
            [data-testid="stFileUploader"] {
                background-color: #0e1117 !important;
                border: 2px dashed white !important;
            }
            [data-testid="stFileUploader"] * { color: white !important; }
            [data-testid="stSelectbox"], [data-testid="stMultiselect"] {
                background-color: #0e1117 !important;
                border: 1px solid white !important;
            }
            [data-testid="stSelectbox"] * , [data-testid="stMultiselect"] * { color: white !important; }
            .css-1uccc91-singleValue, .css-1wa3eu0-placeholder, .css-1n76uvr {
                color: white !important;
                background-color: #0e1117 !important;
            }
            ::placeholder { color: white !important; }
            </style>
            """, unsafe_allow_html=True)
    else:
        st.markdown(
            """
            <style>
            html, body, .stApp, [data-testid="stAppViewContainer"] {
                background-color: white;
                color: black !important;
            }
            [data-testid="stSidebar"] {
                background-color: #f0f0f0 !important;
                color: black !important;
            }
            </style>
            """, unsafe_allow_html=True)
    
    # ------------------ Added CSS for Animations ------------------
    st.markdown(
    """
    <style>
    /* Animate file uploader on hover */
    div[data-testid="stFileUploader"] {
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    div[data-testid="stFileUploader"]:hover {
        transform: scale(1.05);
        box-shadow: 0 4px 15px rgba(0,0,0,0.3);
    }
    /* Animate pie chart slices on hover without altering original colors */
    div[data-testid="stPlotlyChart"] svg g.slice:hover {
        transform: scale(1.1);
        transform-origin: center;
        transition: transform 0.3s ease;
    }
    /* Animate bar graph bars on hover without altering original colors */
    div[data-testid="stPlotlyChart"] svg rect:hover {
        transform: scale(1.1);
        transform-origin: center;
        transition: transform 0.3s ease;
    }
    </style>
    """, unsafe_allow_html=True)
    # ------------------ End Added CSS ------------------

    if "report_sections" not in st.session_state:
        st.session_state["report_sections"] = {}

    st.markdown('<h1>🚀 AutoDash</h1>', unsafe_allow_html=True)
    st.markdown('<p>Upload your dataset for automatic cleaning and visualization.</p>', unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("📂 Upload Your Data File", type=['csv', 'xlsx'])

    # ------------- NEW: Clear old report sections if a new file or sample data is chosen -------------
    if uploaded_file is not None:
        # If new file is uploaded, reset old sections so only current visualizations appear
        if "previous_file_name" not in st.session_state or st.session_state["previous_file_name"] != uploaded_file.name:
            st.session_state["report_sections"] = {}
            st.session_state["previous_file_name"] = uploaded_file.name
    else:
        # Using sample data if no file is uploaded
        if "previous_file_name" not in st.session_state or st.session_state["previous_file_name"] != "sample_data":
            st.session_state["report_sections"] = {}
            st.session_state["previous_file_name"] = "sample_data"
    # ------------- END NEW CODE -------------
    
    if uploaded_file is not None:
        try:
            st.success("File Uploaded Successfully! ✅")
            raw_df = load_data(uploaded_file)
            if raw_df is None:
                st.error("Error: Could not load file. Please check the format and try again.")
                return
        except Exception as e:
            st.error(f"Unexpected error: {str(e)}")
            return
    else:
        st.info("No file uploaded. Using sample dataset for demonstration.")
        raw_df = get_sample_data()
    
    cleaned_df = clean_data(raw_df)
    
    st.subheader("🔍 Data Summary")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Original Rows", len(raw_df))
    col2.metric("Columns", len(raw_df.columns))
    col3.metric("Duplicate Rows", raw_df.duplicated().sum())
    col4.metric("Cleaned Rows", len(cleaned_df))
    
    st.subheader("🧹 Clean Your Data")
    drop_columns = st.multiselect("Select columns to drop", raw_df.columns)
    if drop_columns:
        cleaned_df = cleaned_df.drop(columns=drop_columns)
    
    with st.expander("📁 View Cleaned Data Sample"):
         if cleaned_df.empty:
            st.write("No data to display, as the cleaned dataset is empty.")
         else:
            st.dataframe(cleaned_df)  
    
    st.subheader("📊 Data Visualizations")
    if not cleaned_df.empty:
        selected_col = st.selectbox("Select a column to visualize", cleaned_df.columns)
        chart_type = st.radio("Select Chart Type", ["Histogram", "Box Plot", "Pie Chart", "Line Chart", "Map Chart"])
        if chart_type == "Histogram":
            fig = px.histogram(cleaned_df, x=selected_col, title=f"Distribution of {selected_col}")
        elif chart_type == "Box Plot":
            fig = px.box(cleaned_df, y=selected_col, title=f"Box Plot of {selected_col}")
        elif chart_type == "Pie Chart":
            count_data = cleaned_df[selected_col].value_counts()
            fig = px.pie(count_data, values=count_data.values, names=count_data.index, title=f"{selected_col} Distribution")
        elif chart_type == "Line Chart":
            ts_data = cleaned_df[selected_col].value_counts().sort_index()
            fig = px.line(x=ts_data.index, y=ts_data.values, title=f"Time Series of {selected_col}")
        elif chart_type == "Map Chart":
            if "latitude" in cleaned_df.columns and "longitude" in cleaned_df.columns:
                fig = px.scatter_mapbox(cleaned_df,
                                        lat="latitude",
                                        lon="longitude",
                                        hover_name=selected_col,
                                        title=f"Map Chart of {selected_col}",
                                        mapbox_style="open-street-map")
            else:
                st.error("Dataset must contain 'latitude' and 'longitude' columns for Map Chart.")
                return
        st.plotly_chart(fig, use_container_width=True)
        head_content, body_content = extract_full_fig_html(fig)
        set_report_section("data_vis", f"Data Visualization - {chart_type}: {selected_col}", head_content, body_content, "plotly")

        st.subheader("📈 Correlation Heatmap")
        numeric_df = cleaned_df.select_dtypes(include=[np.number])
        if not numeric_df.empty and len(numeric_df.columns) > 1:
            data = numeric_df.corr()
            fig, ax = plt.subplots()
            sns.heatmap(data.corr(numeric_only=True), ax=ax, cmap="YlGnBu", annot=True)
            st.pyplot(fig, use_container_width=True)
            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=200)
            buf.seek(0)
            img_base64 = base64.b64encode(buf.read()).decode("utf-8")
            html_img = f'<img src="data:image/png;base64,{img_base64}" style="width:100%;">'
            set_report_section("corr_heatmap", "Correlation Heatmap", "", html_img, "image")
        else:
            st.write("Not enough numeric columns to generate correlation heatmap.")
    
    st.subheader("3D Tactile Data Sculpture")
    numeric_columns = cleaned_df.select_dtypes(include=[np.number]).columns.tolist()
    if numeric_columns:
        selected_numeric_col = st.selectbox("Select a numeric column for 3D sculpture", numeric_columns)
        if st.button("Generate 3D Sculpture"):
            data_for_sculpture = cleaned_df[selected_numeric_col].dropna().values
            if len(data_for_sculpture) > 1:
                generate_3d_sculpture(data_for_sculpture)
            else:
                st.error("Not enough data points to generate a sculpture.")
    else:
        st.write("No numeric columns available for 3D sculpture generation.")
    
    st.subheader("🤖 Machine Learning Features")
    if not cleaned_df.empty:
        numeric_cols = cleaned_df.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_cols) >= 2:
            target_col = st.selectbox("Select target column", numeric_cols)
            feature_cols = st.multiselect("Select feature columns", numeric_cols, default=[col for col in numeric_cols if col != target_col])
            if target_col and feature_cols:
                X = cleaned_df[feature_cols]
                y = cleaned_df[target_col]
                if st.button("Train Model"):
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                    if pd.api.types.is_numeric_dtype(y):
                        model = LinearRegression()
                        model_type = "Regression"
                    else:
                        model = RandomForestClassifier()
                        model_type = "Classification"
                    
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    
                    st.write(f"### {model_type} Results")
                    if model_type == "Regression":
                        mse = mean_squared_error(y_test, y_pred)
                        st.metric("Mean Squared Error", f"{mse:.2f}")
                        min_val = y_test.min()
                        max_val = y_test.max()
                        fig_ml = px.scatter(x=y_test, y=y_pred,
                                            labels={'x': "Actual", 'y': "Predicted"},
                                            title="Actual vs Predicted")
                        fig_ml.add_shape(
                            type="line",
                            x0=min_val, y0=min_val,
                            x1=max_val, y1=max_val,
                            line=dict(color="red", dash="dash")
                        )
                        st.plotly_chart(fig_ml, use_container_width=True)
                        head_ml, body_ml = extract_full_fig_html(fig_ml)
                        set_report_section("ml_regression", "ML Regression: Actual vs Predicted", head_ml, body_ml, "plotly")
                    else:
                        from sklearn.metrics import confusion_matrix
                        cm = confusion_matrix(y_test, y_pred)
                        fig_ml = px.imshow(cm,
                                             text_auto=True,
                                             labels=dict(x="Predicted", y="Actual"),
                                             title="Confusion Matrix")
                        st.plotly_chart(fig_ml, use_container_width=True)
                        head_ml, body_ml = extract_full_fig_html(fig_ml)
                        set_report_section("ml_classification", "ML Classification: Confusion Matrix", head_ml, body_ml, "plotly")
        else:
            st.write("Need at least 2 numeric columns for ML modeling.")

    st.subheader("🤖 AI Insights")
    if "date" in cleaned_df.columns:
        try:
            cleaned_df["date"] = pd.to_datetime(cleaned_df["date"])
            st.write("### Trend Analysis")
            numeric_cols = cleaned_df.select_dtypes(include=[np.number]).columns.tolist()
            if numeric_cols:
                for col in numeric_cols:
                    cleaned_df["timestamp"] = cleaned_df["date"].apply(lambda x: x.timestamp())
                    corr = cleaned_df[col].corr(cleaned_df["timestamp"])
                    st.write(f"**Trend in {col}:** Correlation with time = {corr:.2f}")
                    fig_trend = px.scatter(cleaned_df, x="date", y=col, title=f"Trend of {col} over Time")
                    fig_trend.update_traces(mode="lines+markers")
                    st.plotly_chart(fig_trend, use_container_width=True)
                    head_trend, body_trend = extract_full_fig_html(fig_trend)
                    set_report_section(f"trend_{col}", f"Trend Analysis: {col}", head_trend, body_trend, "plotly")
            else:
                st.write("No numeric columns available for trend analysis.")
        except Exception as e:
            st.error("Error in trend analysis: " + str(e))
    else:
        st.write("No datetime column ('date') found for trend analysis.")
    
    st.write("### Anomaly Detection")
    numeric_cols = cleaned_df.select_dtypes(include=[np.number]).columns.tolist()
    if numeric_cols:
        anomaly_summary = {}
        for col in numeric_cols:
            try:
                z_scores = np.abs(zscore(cleaned_df[col].astype(float)))
                anomalies = cleaned_df[z_scores > 3]
                anomaly_summary[col] = anomalies.shape[0]
            except Exception as e:
                anomaly_summary[col] = "Error"
        anomaly_df = pd.DataFrame(list(anomaly_summary.items()), columns=["Numeric Column", "Number of Anomalies"])
        st.write(anomaly_df)
        set_report_section("anomaly", "Anomaly Detection", "", anomaly_df.to_html(index=False), "table")
    else:
        st.write("No numeric columns available for anomaly detection.")
    
    if not cleaned_df.empty and 'funding_type' in cleaned_df.columns:
        st.subheader("🎓 Funding Type Breakdown")
        funding_types = cleaned_df['funding_type'].unique()
        for ft in funding_types:
            with st.expander(f"{ft}", expanded=False):
                relevant_cols = [col for col in ['student_id', 'faculty_id', 'id', 
                                                  'name', 'department', 'year', 
                                                  'amount', 'salary'] if col in cleaned_df.columns]
                ft_df = cleaned_df[cleaned_df['funding_type'] == ft]
                st.dataframe(ft_df[relevant_cols] if relevant_cols else ft_df)
    
    st.subheader("📄 Automatic Report Generator")
    if st.button("Generate Report"):
        # Build a custom head that forces a light background and adds histogram hover animation
        report_head = '''
        <meta charset="utf-8">
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        <style>
        body { background-color: #fff; color: #000; }
        .plotly-graph-div { background: #fff !important; }
        .plotly-graph-div svg rect:hover {
            transform: scale(1.1);
            transform-origin: center;
            transition: transform 0.3s ease;
        }
        </style>
        '''
        report_html = "<html><head>" + report_head + "</head><body>"
        report_html += "<h1>AutoDash Automatic Report</h1>"
        report_html += "<h2>Data Summary</h2>"
        report_html += f"<p>Original Rows: {len(raw_df)}</p>"
        report_html += f"<p>Columns: {len(raw_df.columns)}</p>"
        report_html += f"<p>Duplicate Rows: {raw_df.duplicated().sum()}</p>"
        report_html += f"<p>Cleaned Rows: {len(cleaned_df)}</p>"
        
        # Include all visualized sections (only from the current file or sample data)
        for sec in st.session_state["report_sections"].values():
            report_html += f"<h2>{sec['title']}</h2>"
            report_html += sec["body"]
        report_html += "</body></html>"
        
        st.download_button("Download Report", report_html, file_name="AutoDash_Report.html", mime="text/html")
    
    st.subheader("💬 Feedback")
    feedback = st.text_area("Tell us how we can improve AutoDash!")
    if st.button("Submit Feedback"):
        st.success("Thanks for your feedback! 😊")
    
if __name__ == "__main__":
    main()
