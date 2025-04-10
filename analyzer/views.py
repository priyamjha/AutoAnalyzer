import os  # For file path manipulation
import pandas as pd  # For working with dataframes (tabular data)
import numpy as np  # For numerical operations

# Django imports for handling HTTP requests and rendering templates
from django.shortcuts import render, redirect
from django.http import HttpResponse
from django.conf import settings

# Importing form and model classes from the current Django app
from .forms import UploadForm
from .models import UploadedDataset

# Plotly for creating visualizations
import plotly.express as px
import plotly.io as pio

# Home page where users can upload datasets
def index(request):
    if request.method == 'POST':  # If the form is submitted
        form = UploadForm(request.POST, request.FILES)  # Create a form instance with the submitted data
        if form.is_valid():  # Check if the form is valid
            dataset = form.save()  # Save the dataset to the database
            return redirect('report', dataset_id=dataset.id)  # Redirect to the report view with the dataset ID
    else:
        form = UploadForm()  # Create an empty form for GET request
    return render(request, 'analyzer/index.html', {'form': form})  # Render the form in the index template


# Report view that generates statistics and visualizations for the uploaded dataset
def report(request, dataset_id):
    dataset = UploadedDataset.objects.get(id=dataset_id)

    # Load the dataset into a pandas dataframe based on the file type
    if dataset.file.name.endswith('.xlsx'):
        df = pd.read_excel(dataset.file.path, engine='openpyxl')  # Use openpyxl engine for better compatibility
    elif dataset.file.name.endswith('.json'):
        df = pd.read_json(dataset.file.path)
    elif dataset.file.name.endswith('.csv'):
        df = pd.read_csv(dataset.file.path)
    else:
        return HttpResponse("Unsupported file format.", status=400)

    # Calculate missing values before replacing them
    missing_columns = df.isnull().sum().sort_values(ascending=False)
    missing_columns = missing_columns[missing_columns > 0].to_dict()

    # Calculate the total number of columns with missing values
    total_missing_columns = len(missing_columns)

    # Handle missing values in all columns (replace with 'NULL')
    # For object columns (strings, categories)
    obj_cols = df.select_dtypes(include=['object', 'category']).columns
    df[obj_cols] = df[obj_cols].fillna("NULL")  # Replace NaN in object columns with 'NULL'
    
    # For numeric columns (e.g., Age, Score), replace NaN with 'NULL' as well
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna("NULL")  # Replace NaN in numeric columns with 'NULL'

    # Replace empty strings with "NULL" in all columns
    df[obj_cols] = df[obj_cols].replace("", "NULL")

    # Create a boolean mask where 'NULL' values have been inserted
    null_mask = df == "NULL"  # Check all columns for the value "NULL"
    rows_with_null = df[null_mask.any(axis=1)]  # Get all rows where any column has 'NULL'

    # Convert these rows to an HTML table for rendering
    null_table_html = rows_with_null.to_html(classes='table table-striped table-bordered', border=0, index=True)

    # Detect and store duplicate rows in the dataset
    duplicate_rows = df[df.duplicated()]
    duplicate_rows_html = duplicate_rows.to_html(classes='table table-striped table-bordered', border=0, index=False)

    # Remove duplicate rows from the dataframe
    df = df.drop_duplicates()

    # Generate a summary of the dataset (statistics) and convert to HTML
    summary = df.describe(include='all').transpose()
    summary_html = summary.to_html(classes='table table-bordered', border=0)

    # Initialize an empty list for holding plot HTML components
    plot_html = []
    
    # Generate histograms and boxplots for numeric columns (up to 5 columns)
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    for col in numeric_cols[:5]:
        if df[col].dropna().nunique() <= 1:
            continue
        # Generate a histogram for the column
        fig_hist = px.histogram(df, x=col, nbins=30, title=f"{col} - Histogram")
        fig_hist.update_layout(margin=dict(l=10, r=10, t=40, b=10))
        plot_html.append(pio.to_html(fig_hist, full_html=False)) 
        
        # Generate a boxplot for the column
        fig_box = px.box(df, y=col, title=f"{col} - Boxplot")
        fig_box.update_layout(margin=dict(l=10, r=10, t=40, b=10))
        plot_html.append(pio.to_html(fig_box, full_html=False))

    # Generate bar and pie charts for categorical columns (up to 3 columns)
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    for col in cat_cols[:3]:
        if df[col].nunique() < 20:
            freq = df[col].value_counts().reset_index()
            freq.columns = [col, 'Count']
            
            # Create a bar chart for value counts
            fig_bar = px.bar(freq, x=col, y='Count', title=f"{col} - Value Counts")
            fig_bar.update_layout(margin=dict(l=10, r=10, t=40, b=10))
            plot_html.append(pio.to_html(fig_bar, full_html=False))

            # Create a pie chart for value distribution
            fig_pie = px.pie(freq, names=col, values='Count', title=f"{col} - Distribution")
            fig_pie.update_traces(textinfo='percent+label')
            fig_pie.update_layout(margin=dict(l=10, r=10, t=40, b=10))
            plot_html.append(pio.to_html(fig_pie, full_html=False))

    # Generate a correlation heatmap if there are more than one numeric column
    if len(numeric_cols) >= 2:
        corr = df[numeric_cols].corr().round(2)
        fig_corr = px.imshow(corr, text_auto=True, title="Correlation Heatmap")
        fig_corr.update_layout(margin=dict(l=10, r=10, t=40, b=10))
        plot_html.append(pio.to_html(fig_corr, full_html=False))

    most_freq = df.mode().iloc[0].to_dict()  
    freq_percent = {col: round(df[col].value_counts(normalize=True).iloc[0] * 100, 2)
                    for col in most_freq if col in df.columns}

    # Calculate numeric column statistics: mean and range (IQR)
    average_info = {
        col: {
            "mean": round(df[col].mean(), 2),
            "range": (
                round(df[col].quantile(0.25), 2),
                round(df[col].quantile(0.75), 2)
            )
        } for col in numeric_cols
    }

    # Count the number of occurrences of each data type in the columns
    col_types = df.dtypes.apply(lambda x: x.name).value_counts().to_dict()

    # Prepare the dataset statistics to display in the report
    stats = {
        'shape': df.shape,
        'columns': df.columns.tolist(),
        'dtypes': df.dtypes.astype(str).to_dict(),
        'missing': missing_columns,  # Pass the accurate missing data here
        'missing_columns': missing_columns,
        'total_missing_columns': total_missing_columns,
        'most_freq': most_freq,
        'freq_percent': freq_percent,
        'numeric_summary': average_info,
        'col_types': col_types,
        'duplicates': duplicate_rows.shape[0],
        'summary': summary_html,
        'filename': os.path.basename(dataset.file.name),
        'upload_time': dataset.uploaded_at.strftime("%d %b %Y, %I:%M %p"),
        'num_columns': df.shape[1],
        'num_rows': df.shape[0],
    }

    # Render the report template with the dataset stats and plots
    return render(request, 'analyzer/report.html', {
        'dataset': dataset,
        'stats': stats,
        'plot_html': plot_html,
        'duplicate_rows_html': duplicate_rows_html, 
        'null_table_html': null_table_html
    })


# View for downloading the dataset report as a PDF
from django.http import HttpResponse
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from io import BytesIO

def download_pdf(request, dataset_id):
    dataset = UploadedDataset.objects.get(id=dataset_id)

    # Load dataset
    if dataset.file.name.endswith('.xlsx'):
        df = pd.read_excel(dataset.file.path)
    elif dataset.file.name.endswith('.json'):
        df = pd.read_json(dataset.file.path)
    elif dataset.file.name.endswith('.csv'):
        df = pd.read_csv(dataset.file.path)
    else:
        return HttpResponse("Unsupported file format.", status=400)

    # Clean and summarize
    obj_cols = df.select_dtypes(include=['object']).columns
    df[obj_cols] = df[obj_cols].fillna("NULL")
    
    # Remove duplicate rows from the dataframe
    duplicate_rows = df[df.duplicated()]
    df = df.drop_duplicates()

    # Generate summary statistics
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    summary = df.describe(include='all').round(2).fillna('').reset_index()
    summary_data = [summary.columns.tolist()] + summary.values.tolist()

    # Truncate long values
    def truncate(val, length=15):
        val = str(val)
        return val[:length] + ('...' if len(val) > length else '')

    truncated_summary_data = [[truncate(cell) for cell in row] for row in summary_data]

    # Additional stats
    missing_top5 = df.isnull().sum().sort_values(ascending=False).head(5).to_dict()
    most_freq = df.mode().iloc[0].to_dict()
    freq_percent = {
        col: round(df[col].value_counts(normalize=True).iloc[0] * 100, 2)
        for col in most_freq if col in df.columns
    }
    col_types = df.dtypes.apply(lambda x: x.name).value_counts().to_dict()

    # Setup PDF (portrait)
    response = HttpResponse(content_type='application/pdf')
    response['Content-Disposition'] = f'attachment; filename="report_{dataset.id}.pdf"'
    doc = SimpleDocTemplate(response, pagesize=letter, rightMargin=20, leftMargin=20, topMargin=40, bottomMargin=40)
    styles = getSampleStyleSheet()
    elements = []

    # Title
    title_style = styles['Title']
    title_style.fontName = 'Helvetica-Bold'
    elements.append(Paragraph(f"📊 AutoAnalyzer Data Report", title_style))
    elements.append(Spacer(1, 12))

    # Metadata
    elements.append(Paragraph(f"Dataset Report: {os.path.basename(dataset.file.name)}", styles['Heading1']))
    elements.append(Spacer(1, 12))
    elements.append(Paragraph(f"Uploaded: {dataset.uploaded_at.strftime('%d %b %Y, %I:%M %p')}", styles['Normal']))
    elements.append(Paragraph(f"Shape: {df.shape[0]} rows x {df.shape[1]} columns", styles['Normal']))
    elements.append(Paragraph(f"Duplicates Removed: {duplicate_rows.shape[0]}", styles['Normal']))
    elements.append(Spacer(1, 20))

    # Duplicate rows preview
    if not duplicate_rows.empty:
        elements.append(Paragraph("Sample Duplicate Rows:", styles['Heading2']))
        sample_dupes = duplicate_rows.head(30)
        dupes_data = [sample_dupes.columns.tolist()] + sample_dupes.values.tolist()
        dupes_data_trunc = [[truncate(cell) for cell in row] for row in dupes_data]

        table = Table(dupes_data_trunc, repeatRows=1)
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#ffe6e6")),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 7),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ]))
        elements.append(table)
        elements.append(Spacer(1, 20))

    # Top missing values
    elements.append(Paragraph("Top 5 Columns with Missing Values:", styles['Heading2']))
    for k, v in missing_top5.items():
        elements.append(Paragraph(f"{k}: {v}", styles['Normal']))
    elements.append(Spacer(1, 12))

    # Most frequent values
    elements.append(Paragraph("Most Frequent Values (%):", styles['Heading2']))
    for k, v in freq_percent.items():
        elements.append(Paragraph(f"{k}: {most_freq[k]} ({v}%)", styles['Normal']))
    elements.append(Spacer(1, 12))

    # Column types
    elements.append(Paragraph("Column Types:", styles['Heading2']))
    for k, v in col_types.items():
        elements.append(Paragraph(f"{k}: {v}", styles['Normal']))
    elements.append(PageBreak())

    # Summary statistics table
    elements.append(Paragraph("Summary Statistics:", styles['Heading2']))
    table = Table(truncated_summary_data, repeatRows=1)
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#f0f0f0")),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, -1), 7),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
    ]))
    elements.append(table)
    elements.append(PageBreak())

    # Plot helper with dynamic scaling
    def add_plot(fig):
        buf = BytesIO()
        fig.write_image(buf, format='png')
        buf.seek(0)
        
        # Dynamically calculate size based on the content
        img = Image(buf)
        img.drawHeight = 300  # Set height (default scaling)
        img.drawWidth = 500   # Set width (default scaling)
        
        elements.append(img)
        elements.append(Spacer(1, 12))

    # Visualizations
    elements.append(Paragraph("📊 Visualizations:", styles['Heading2']))
    elements.append(Spacer(1, 12))

    # Numeric plots
    for col in numeric_cols:  # Remove the slice [:3] to include all numeric columns
        if df[col].dropna().nunique() <= 1:
            continue
        add_plot(px.histogram(df, x=col, nbins=30, title=f"{col} - Histogram"))
        add_plot(px.box(df, y=col, title=f"{col} - Boxplot"))

    # Categorical plots
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    for col in cat_cols:  # Remove the slice [:2] to include all categorical columns
        if df[col].nunique() < 20:
            freq = df[col].value_counts().reset_index()
            freq.columns = [col, 'Count']
            add_plot(px.bar(freq, x=col, y='Count', title=f"{col} - Value Counts"))
            add_plot(px.pie(freq, names=col, values='Count', title=f"{col} - Distribution"))

    # Correlation plot
    if len(numeric_cols) > 1:
        corr = df[numeric_cols].corr().round(2)
        fig_corr = px.imshow(corr, text_auto=True, title="Correlation Heatmap")
        add_plot(fig_corr)

    # Footer
    elements.append(Spacer(1, 12))
    elements.append(Paragraph(f"Generated by AutoAnalyzer | {dataset.uploaded_at.strftime('%d %b %Y, %I:%M %p')}", styles['Normal']))
    
    # Finalize PDF
    doc.build(elements)
    return response
