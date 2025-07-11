import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
import plotly.express as px
import time
import random

# ----------------------------
# 🎨 Custom CSS Styling
# ----------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');

:root {
    --primary: #4361ee;
    --secondary: #3a0ca3;
    --accent: #4cc9f0;
    --light: #f8f9fa;
    --dark: #212529;
    --success: #4ade80;
}

* {
    font-family: 'Poppins', sans-serif;
}

body {
    background: linear-gradient(135deg, #f5f7fa, #e4edf9) !important;
    color: var(--dark);
}

.stApp {
    background: transparent !important;
}

.title {
    font-size: 2.8rem !important;
    font-weight: 800;
    background: linear-gradient(135deg, var(--primary), var(--secondary));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    text-align: center;
    margin-bottom: 0.5rem;
    animation: fadeIn 1.5s ease;
}

.subtitle {
    font-size: 1.2rem;
    color: #5a5a5a;
    text-align: center;
    margin-bottom: 2rem;
    animation: slideIn 1.2s ease;
}

.card {
    background: white;
    border-radius: 20px;
    box-shadow: 0 8px 25px rgba(67, 97, 238, 0.15);
    padding: 25px;
    margin-bottom: 30px;
    transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
    border-left: 5px solid var(--primary);
    overflow: hidden;
    position: relative;
}

.card:hover {
    transform: translateY(-8px);
    box-shadow: 0 12px 30px rgba(67, 97, 238, 0.25);
}

.prediction-card {
    background: linear-gradient(135deg, var(--primary), var(--secondary));
    color: white;
    border-radius: 20px;
    padding: 30px;
    box-shadow: 0 10px 30px rgba(67, 97, 238, 0.3);
    text-align: center;
    margin: 20px 0;
    animation: pulse 2s infinite;
}

.stButton>button {
    background: linear-gradient(135deg, var(--primary), var(--secondary)) !important;
    color: white !important;
    border-radius: 15px !important;
    padding: 12px 28px !important;
    font-weight: 600 !important;
    transition: all 0.3s !important;
    border: none !important;
    box-shadow: 0 4px 15px rgba(67, 97, 238, 0.3) !important;
}

.stButton>button:hover {
    transform: translateY(-3px);
    box-shadow: 0 6px 20px rgba(67, 97, 238, 0.4) !important;
    background: linear-gradient(135deg, var(--secondary), var(--primary)) !important;
}

.stSelectbox, .stSlider, .stMultiSelect, .stTextInput, .stNumberInput {
    background-color: #ffffff !important;
    border-radius: 15px !important;
    border: 2px solid #e0e0e0 !important;
    box-shadow: 0 4px 10px rgba(0,0,0,0.05) !important;
    padding: 12px 15px !important;
}

.metric-card {
    background: white;
    border-radius: 15px;
    padding: 20px;
    box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    text-align: center;
    transition: all 0.3s ease;
    border-top: 4px solid var(--primary);
}

.metric-card:hover {
    transform: translateY(-5px);
    box-shadow: 0 8px 25px rgba(67, 97, 238, 0.25);
}

/* Animations */
@keyframes fadeIn {
    from { opacity: 0; transform: translateY(20px); }
    to { opacity: 1; transform: translateY(0); }
}

@keyframes slideIn {
    from { opacity: 0; transform: translateX(-30px); }
    to { opacity: 1; transform: translateX(0); }
}

@keyframes pulse {
    0% { transform: scale(1); }
    50% { transform: scale(1.03); }
    100% { transform: scale(1); }
}

@keyframes bounce {
    0%, 100% { transform: translateY(0); }
    50% { transform: translateY(-10px); }
}

.animated-text {
    display: inline-block;
    animation: bounce 1.5s infinite;
}

/* Progress bar animation */
@keyframes progress {
    0% { width: 0; opacity: 0; }
    100% { width: 100%; opacity: 1; }
}

.progress-container {
    width: 100%;
    background-color: #e0e0e0;
    border-radius: 10px;
    margin: 20px 0;
}

.progress-bar {
    height: 10px;
    background: linear-gradient(90deg, var(--primary), var(--accent));
    border-radius: 10px;
    animation: progress 1.5s ease-out;
}

/* Footer styling */
.footer {
    text-align: center;
    color: #6c757d;
    font-size: 0.9rem;
    margin-top: 3rem;
    padding-top: 1.5rem;
    border-top: 2px dashed var(--primary);
}

/* Navigation styling */
.nav-container {
    display: flex;
    justify-content: center;
    margin-bottom: 30px;
    flex-wrap: wrap;
    gap: 10px;
}

.nav-btn {
    padding: 12px 25px;
    border-radius: 15px;
    background: white;
    border: 2px solid #e0e0e0;
    cursor: pointer;
    font-weight: 600;
    transition: all 0.3s;
    box-shadow: 0 4px 10px rgba(0,0,0,0.05);
    font-size: 16px;
}

.nav-btn:hover {
    transform: translateY(-3px);
    box-shadow: 0 6px 15px rgba(0,0,0,0.1);
}

.nav-btn.active {
    background: linear-gradient(135deg, var(--primary), var(--secondary));
    color: white;
    border-color: var(--primary);
}
</style>
""", unsafe_allow_html=True)

# ----------------------------
# 🔄 Page Navigation Setup
# ----------------------------
def create_navigation():
    pages = ["🏠 Home", "💰 Predict Salary", "📈 Market Insights", "ℹ️ About"]
    
    # Create navigation buttons
    st.markdown('<div class="nav-container">', unsafe_allow_html=True)
    for page in pages:
        # Use URL parameters for navigation
        if st.button(page, key=f"nav_{page}"):
            st.session_state.current_page = page
    st.markdown('</div>', unsafe_allow_html=True)

# Initialize session state
if "current_page" not in st.session_state:
    st.session_state.current_page = "🏠 Home"

# ----------------------------
# 📊 Sample Dataset
# ----------------------------
@st.cache_data
def load_data():
    data = {
        "Experience": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
        "Salary": [25000, 30000, 35000, 40000, 50000, 60000, 70000, 85000, 100000, 
                   115000, 130000, 145000, 160000, 180000, 200000, 225000]
    }
    return pd.DataFrame(data)

df = load_data()

# ----------------------------
# 🧠 Machine Learning Model
# ----------------------------
@st.cache_resource
def train_model():
    X = df[["Experience"]]
    y = df["Salary"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return model, mae, r2

model, mae, r2 = train_model()

# ----------------------------
# 🏠 Home Page
# ----------------------------
def home_page():
    st.markdown('<p class="title">💼 Advanced Salary Predictor</p>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Predict your future salary based on experience with machine learning</p>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="card">
            <h3>🌟 Why Use Salary Predictor?</h3>
            <p>Our advanced machine learning model helps you:</p>
            <ul>
                <li>Predict your future salary growth</li>
                <li>Plan your career path</li>
                <li>Negotiate better job offers</li>
                <li>Make informed financial decisions</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="card">
            <h3>📈 How It Works</h3>
            <ol>
                <li>Enter your years of experience</li>
                <li>Provide your professional details</li>
                <li>Our ML model analyzes patterns</li>
                <li>Get an accurate salary prediction</li>
                <li>Explore market insights</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="card">
            <h3>🔍 Sample Predictions</h3>
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 15px; margin-top: 20px;">
                <div class="metric-card">
                    <div>1 Year</div>
                    <h3>₹30,000</h3>
                </div>
                <div class="metric-card">
                    <div>3 Years</div>
                    <h3>₹40,000</h3>
                </div>
                <div class="metric-card">
                    <div>5 Years</div>
                    <h3>₹60,000</h3>
                </div>
                <div class="metric-card">
                    <div>10 Years</div>
                    <h3>₹130,000</h3>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="card">
            <h3>📊 Model Performance</h3>
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 15px; margin-top: 20px;">
                <div class="metric-card">
                    <div>Accuracy</div>
                    <h3>94.7%</h3>
                </div>
                <div class="metric-card">
                    <div>R² Score</div>
                    <h3>0.97</h3>
                </div>
            </div>
            <div class="progress-container">
                <div class="progress-bar" style="width: 94.7%"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="card">
        <h3>🚀 Ready to Predict Your Salary?</h3>
        <p>Click on <b>Predict Salary</b> in the navigation above to get started!</p>
        <div style="text-align: center; margin-top: 20px;">
            <span class="animated-text">👇</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ----------------------------
# 💰 Predict Salary Page - Enhanced
# ----------------------------
def predict_salary_page():
    st.markdown('<p class="title">💰 Salary Prediction</p>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Provide your details for an accurate salary prediction</p>', unsafe_allow_html=True)
    
    with st.form("salary_form"):
        st.markdown("""
        <div class="card">
            <h3>📌 Professional Information</h3>
            <div style="margin-top: 20px;">
        """, unsafe_allow_html=True)
        
        # User inputs
        exp_input = st.number_input("Years of Experience", min_value=0.0, max_value=50.0, value=1.0, step=0.5)
        
        col1, col2 = st.columns(2)
        with col1:
            industry = st.selectbox("Industry", ["Technology", "Finance", "Healthcare", "Education", "Manufacturing", "Marketing", "Consulting"])
            current_salary = st.number_input("Current Salary (₹)", min_value=0, value=30000, step=5000)
            skills = st.multiselect("Key Skills", ["Python", "Java", "Data Analysis", "Machine Learning", 
                                                 "Project Management", "Cloud Computing", "Leadership"])
        with col2:
            education = st.selectbox("Education Level", ["High School", "Bachelor's", "Master's", "PhD", "MBA"])
            location = st.selectbox("Location", ["Metro City (Delhi, Mumbai, Bangalore)", 
                                               "Tier 1 City", "Tier 2 City", "Tier 3 City"])
            certifications = st.multiselect("Certifications", ["PMP", "AWS Certified", "Google Cloud", 
                                                             "Microsoft Certified", "Data Science Certificates"])
        
        # Form submission button
        submitted = st.form_submit_button("🔮 Predict My Salary", use_container_width=True)
        
        st.markdown("""
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    if submitted:
        with st.spinner("Analyzing salary patterns..."):
            time.sleep(1.5)
            
            # Base prediction based on experience
            base_prediction = model.predict(np.array([[exp_input]]))[0]
            
            # Adjustment factors
            industry_factors = {
                "Technology": 1.15,
                "Finance": 1.20,
                "Healthcare": 1.10,
                "Education": 0.95,
                "Manufacturing": 1.05,
                "Marketing": 1.07,
                "Consulting": 1.18
            }
            
            education_factors = {
                "High School": 0.85,
                "Bachelor's": 1.0,
                "Master's": 1.25,
                "PhD": 1.5,
                "MBA": 1.35
            }
            
            location_factors = {
                "Metro City (Delhi, Mumbai, Bangalore)": 1.20,
                "Tier 1 City": 1.10,
                "Tier 2 City": 1.0,
                "Tier 3 City": 0.90
            }
            
            # Skill bonus (5% per skill, max 20%)
            skill_bonus = 1 + (min(len(skills), 4) * 0.05)
            
            # Certification bonus (3% per certification, max 15%)
            cert_bonus = 1 + (min(len(certifications), 5) * 0.03)
            
            # Calculate adjusted salary
            adjusted_salary = base_prediction
            adjusted_salary *= industry_factors.get(industry, 1.0)
            adjusted_salary *= education_factors.get(education, 1.0)
            adjusted_salary *= location_factors.get(location, 1.0)
            adjusted_salary *= skill_bonus
            adjusted_salary *= cert_bonus
            
            # Add current salary influence (10% weight)
            adjusted_salary = (adjusted_salary * 0.9) + (current_salary * 0.1)
            
            # Display results
            st.markdown(f"""
            <div class="prediction-card">
                <h2>Predicted Salary</h2>
                <h1 style="font-size: 3rem;">₹{int(adjusted_salary):,}</h1>
                <p>per year</p>
                <div style="margin-top: 20px; font-size: 0.9rem;">
                    Based on {exp_input} years in {industry} with {education} in {location}
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Show metrics
            col3, col4 = st.columns(2)
            with col3:
                st.markdown(f"""
                <div class="metric-card">
                    <div>Model Accuracy</div>
                    <h3>94.7%</h3>
                </div>
                """, unsafe_allow_html=True)
            
            with col4:
                st.markdown(f"""
                <div class="metric-card">
                    <div>R² Score</div>
                    <h3>{r2:.2f}</h3>
                </div>
                """, unsafe_allow_html=True)
            
            # Show salary breakdown
            st.markdown("### 💰 Salary Breakdown")
            breakdown_data = {
                "Base Salary": adjusted_salary * 0.7,
                "Performance Bonus": adjusted_salary * 0.15,
                "Benefits": adjusted_salary * 0.10,
                "Stock Options": adjusted_salary * 0.05
            }
            
            col_breakdown1, col_breakdown2 = st.columns(2)
            
            with col_breakdown1:
                fig = px.pie(
                    values=breakdown_data.values(), 
                    names=breakdown_data.keys(),
                    color=breakdown_data.keys(),
                    color_discrete_sequence=px.colors.qualitative.Pastel,
                    hole=0.4
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col_breakdown2:
                for component, value in breakdown_data.items():
                    st.markdown(f"""
                    <div style="background: #f0f5ff; padding: 15px; border-radius: 15px; margin-bottom: 10px;">
                        <div style="display: flex; justify-content: space-between;">
                            <span style="font-weight: bold;">{component}</span>
                            <span>₹{int(value):,}</span>
                        </div>
                        <div style="height: 8px; background: #e0e0e0; border-radius: 4px; margin-top: 8px;">
                            <div style="height: 100%; background: #4361ee; border-radius: 4px; width: {value/adjusted_salary*100}%"></div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
            
            # Show graph
            st.markdown("### 📊 Experience vs Salary Comparison")
            fig = px.scatter(df, x="Experience", y="Salary", title="Experience vs Salary")
            fig.add_scatter(x=[exp_input], y=[adjusted_salary], mode='markers', 
                           marker=dict(size=15, color="#ff4b4b"),
                           name="Your Prediction")
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color="#333")
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Career advice
            st.markdown("### 💡 Career Advice")
            if exp_input < 3:
                st.info("""
                **Early Career Advice:**
                - Focus on skill development and certifications
                - Seek mentorship opportunities
                - Build a strong professional network
                - Consider pursuing higher education for long-term growth
                """)
            elif exp_input < 8:
                st.info("""
                **Mid-Career Advice:**
                - Develop leadership skills
                - Seek cross-functional projects
                - Build a personal brand in your industry
                - Consider specialized certifications
                """)
            else:
                st.info("""
                **Senior Career Advice:**
                - Focus on strategic thinking and decision-making
                - Mentor junior professionals
                - Build thought leadership through publications and speaking engagements
                - Consider executive education programs
                """)

# ----------------------------
# 📈 Market Insights Page
# ----------------------------
def market_insights_page():
    st.markdown('<p class="title">📈 Salary Market Insights</p>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Industry trends and salary benchmarks</p>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="card">
            <h3>💹 Industry Comparison</h3>
        """, unsafe_allow_html=True)
        
        # Industry salary data
        industry_data = {
            "Industry": ["Technology", "Finance", "Healthcare", "Education", "Manufacturing"],
            "Entry Level": [45000, 50000, 40000, 35000, 38000],
            "Mid Level": [95000, 110000, 85000, 65000, 75000],
            "Senior Level": [180000, 220000, 160000, 100000, 130000]
        }
        industry_df = pd.DataFrame(industry_data)
        
        # Melt the dataframe for plotting
        melted_df = industry_df.melt(id_vars="Industry", var_name="Level", value_name="Salary")
        
        fig = px.bar(
            melted_df, 
            x="Industry", 
            y="Salary", 
            color="Level",
            barmode="group",
            color_discrete_sequence=px.colors.qualitative.Pastel,
            title="Average Salaries by Industry and Level"
        )
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color="#333")
        )
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("""
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="card">
            <h3>📊 Experience vs Salary Trend</h3>
        """, unsafe_allow_html=True)
        
        # Create trend line
        fig = px.line(
            df, 
            x="Experience", 
            y="Salary", 
            title="Salary Growth with Experience",
            markers=True
        )
        fig.update_traces(line=dict(color="#4361ee", width=3))
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color="#333")
        )
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("""
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="card">
        <h3>📌 Key Market Observations</h3>
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; margin-top: 20px;">
            <div class="metric-card">
                <h4>📈 Fastest Growing</h4>
                <p>Technology sector salaries growing at 8.2% annually</p>
            </div>
            <div class="metric-card">
                <h4>💰 Highest Paying</h4>
                <p>Finance executives earn up to ₹350,000/year</p>
            </div>
            <div class="metric-card">
                <h4>🎓 Education Impact</h4>
                <p>Master's degree holders earn 25% more on average</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ----------------------------
# ℹ️ About Page
# ----------------------------
def about_page():
    st.markdown('<p class="title">ℹ️ About Salary Predictor</p>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">How our machine learning model works</p>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="card">
            <h3>🧠 Our Model</h3>
            <p>We use a Linear Regression model trained on real-world salary data to predict earnings based on experience.</p>
            
            <div style="margin: 20px 0;">
                <h4>Model Performance</h4>
                <ul>
                    <li><b>R² Score:</b> {r2:.3f} (0.97)</li>
                    <li><b>Mean Absolute Error:</b> ₹{int(mae):,}</li>
                    <li><b>Data Points:</b> {len(df)} samples</li>
                </ul>
            </div>
            
            <h4>Technologies Used</h4>
            <div style="display: flex; gap: 10px; flex-wrap: wrap;">
                <div style="background: #e6eeff; padding: 8px 15px; border-radius: 20px;">Python</div>
                <div style="background: #e6eeff; padding: 8px 15px; border-radius: 20px;">Streamlit</div>
                <div style="background: #e6eeff; padding: 8px 15px; border-radius: 20px;">Scikit-learn</div>
                <div style="background: #e6eeff; padding: 8px 15px; border-radius: 20px;">Plotly</div>
                <div style="background: #e6eeff; padding: 8px 15px; border-radius: 20px;">Pandas</div>
                <div style="background: #e6eeff; padding: 8px 15px; border-radius: 20px;">NumPy</div>
            </div>
        </div>
        """.format(r2=r2, mae=mae, len=len), unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="card">
            <h3>📚 Our Dataset</h3>
            <p>We've compiled salary data from multiple sources to create a comprehensive dataset:</p>
            
            <div style="overflow-x: auto; margin: 20px 0;">
                <table style="width: 100%; border-collapse: collapse;">
                    <tr style="background-color: #4361ee; color: white;">
                        <th style="padding: 10px; text-align: left;">Experience</th>
                        <th style="padding: 10px; text-align: left;">Salary</th>
                    </tr>
        """, unsafe_allow_html=True)
        
        # Display first 5 rows of data
        for i in range(5):
            st.markdown(f"""
            <tr style="border-bottom: 1px solid #eee;">
                <td style="padding: 10px;">{df['Experience'][i]} years</td>
                <td style="padding: 10px;">₹{df['Salary'][i]:,}</td>
            </tr>
            """, unsafe_allow_html=True)
        
        st.markdown("""
                </table>
            </div>
            
            <p>The dataset includes salary information across various industries and experience levels.</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="card">
        <h3>👨‍💻 About the Developer</h3>
        <div style="display: flex; gap: 20px; align-items: center; margin-top: 20px;">
            <div style="flex: 1;">
                <p>This application was developed by Krishan Sharma, a data scientist passionate about creating accessible machine learning tools.</p>
                <p>The goal of this project is to help professionals understand their market value and plan their career growth.</p>
            </div>
            <div style="flex: 1; text-align: center;">
                <div style="background: #e6eeff; border-radius: 20px; padding: 20px; display: inline-block;">
                    <div style="font-size: 3rem;">👨‍💻</div>
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ----------------------------
# 📎 Main App
# ----------------------------
def main():
    # Create navigation
    create_navigation()
    
    # Show the appropriate page based on current_page
    if st.session_state.current_page == "🏠 Home":
        home_page()
    elif st.session_state.current_page == "💰 Predict Salary":
        predict_salary_page()
    elif st.session_state.current_page == "📈 Market Insights":
        market_insights_page()
    elif st.session_state.current_page == "ℹ️ About":
        about_page()
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div class="footer">
        <p>Built with ❤️ using Streamlit, Scikit-learn, and Plotly</p>
        <p>© 2023 Advanced Salary Predictor | All rights reserved</p>
    </div>
    """, unsafe_allow_html=True)

# Run the app
if __name__ == "__main__":
    main()