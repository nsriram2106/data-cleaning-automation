Step 1: Open Command Prompt

Step 2: Navigate to Your Project Directory
cd C:\Streamlit_Project
Step 3: Create a Virtual Environment
python -m venv venv

Step 4: Activate the Virtual Environment
Run the appropriate command based on your OS:
venv\Scripts\activate

Step 5: Install Required Libraries
Once activated, install Streamlit and other dependencies:
pip install streamlit pandas matplotlib seaborn scikit-learn

Structure Your Project
Your project should be structured like this:
📂 PythonProject3
│── 📜 app.py                                # Main Streamlit app
│── 📜 data_processing.py     # Data Preprocessing Code
│── 📜 eda.py                                # EDA Code
│── 📂 venv                                    # Virtual Environment
│── 📂 data                                   # Folder for dataset (if needed)


Running the Application
Once everything is set up, activate the virtual environment and run:

venv\Scripts\activate
streamlit run app.py












