# Predicting Game Success on Steam Using NLP 🎮

## 📖 Project Overview
This project aims to predict the success of games on Steam using **Machine Learning** and **Natural Language Processing (NLP)** techniques. Sentiment Analysis was conducted on game data to visualize and understand the criteria behind a game's market performance.

---

## ⚠️ Important Note: Data Source Pivot
**Due to a technical limitation with the source dataset (Kaggle), the "Reviews" (User Comments) file was unavailable.**

To demonstrate the functionality of the NLP algorithm, the sentiment analysis was performed on **"Game Descriptions"** instead. 

* **Current State:** Analysis uses game description text to determine sentiment metrics.
* **Scalability:** The code architecture is designed to function seamlessly with user review data once it becomes available.

---

## 📂 Dataset Setup
The dataset used in this project is too large to be hosted directly on GitHub.

1.  **Download:** You can download the full dataset from Kaggle here:  
    👉 **[Download Link (Kaggle: Steam Store Games)](https://www.kaggle.com/datasets/nikdavis/steam-store-games)**

2.  **Setup:** Please create a folder named `data/` inside the project directory and place the downloaded CSV files into it.

    **Directory Structure:**
    ```text
    SteamProject/
    ├── data/
    │   ├── steam.csv
    │   ├── steam_description_data.csv
    │   └── ... (other files)
    ├── Final_Project.ipynb
    └── README.md
    ```

---

## 🚀 Execution Instructions (IMPORTANT)
To ensure the data files load correctly and relative paths function as intended, please follow these steps strictly:

1.  **Extract:** Unzip/Extract the downloaded project folder (`"SteamProject"`) to your computer (e.g., Desktop).
2.  **Launch:** Open Jupyter Notebook.
3.  **Navigate:** Inside the Jupyter interface, navigate specifically **into** the unzipped `"SteamProject"` folder.
4.  **Open:** Click on the `Final_Project.ipynb` file located inside that folder.
5.  **Run:** Click **"Run All"** from the top menu to execute the analysis.

> **Note:** If the notebook is not opened from *within* the project folder (i.e., if you open it via a shortcut or from a different root directory), it may fail to locate the `data/` directory.

---

## 🛠️ Technologies Used
* **Python**
* **Pandas** (Data Manipulation)
* **TextBlob** (Sentiment Analysis/NLP)
* **Scikit-Learn** (Machine Learning)
* **Matplotlib / Seaborn** (Data Visualization)
