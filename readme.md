# 📊 AutoAnalyzer – Universal Data Analyzer Tool 

**AutoAnalyzer** is a Django web application that allows users to upload datasets in CSV, Excel, or JSON format, and automatically generates rich data reports and visualizations. It handles data cleaning, statistical summaries, and renders interactive charts — all without writing a single line of code.

## 🚀 Features

- Upload datasets in `.csv`, `.xlsx`, or `.json` formats
- Automatic detection and reporting of:
  - Missing values
  - Duplicate records
  - Data types and distributions
- Generates:
  - Summary statistics
  - Histograms & boxplots (numeric data)
  - Bar & pie charts (categorical data)
  - Correlation heatmap
- Export reports as downloadable PDF files

---

## 🛠️ How It Works

### 1. Upload Dataset

Users upload a file via the home page. Supported formats:
- `.csv`
- `.xlsx` (Excel)
- `.json`

### 2. Dataset Analysis (`report` view)

The app processes the file and generates:

- ✅ Cleaned data (nulls replaced with `"NULL"`)
- ✅ Duplicate detection and removal
- 📊 Summary statistics (via `pandas.describe()`)
- 🔍 Most frequent values per column
- 📈 Charts:
  - Histograms and boxplots (for numeric data)
  - Bar charts and pie charts (for categorical data)
  - Correlation heatmap (if >1 numeric column)

### 3. PDF Export (`download_pdf` view)

A downloadable PDF report is created using `reportlab`, which includes:
- Metadata (upload date, file name, shape)
- Top missing values
- Most frequent values
- Summary table
- All generated charts

---

## 🧪 Tech Stack

- **Backend:** Django
- **Data Analysis:** Pandas, NumPy
- **Visualization:** Plotly
- **PDF Reports:** ReportLab
- **Frontend:** HTML (with Bootstrap-compatible tables)

---

## 📦 Installation

```bash
# Clone the repository
git clone https://github.com/priyamjha/AutoAnalyzer.git
cd autoanalyzer

# Create virtual environment
python -m venv env
source env/bin/activate  # On Windows use `env\Scripts\activate`

# Install dependencies
pip install -r requirements.txt

# Make migrations
python manage.py makemigrations

# Run migrations
python manage.py migrate

# Start the server
python manage.py runserver
```

---

## 📄 Requirements

Include these in `requirements.txt`:

```txt
Django>=4.0
pandas
numpy
plotly
openpyxl
reportlab
```

---

## 📝 License

MIT License. Feel free to use and modify!

---
