import streamlit as st
import os
import shutil
from expense_parser_all import main as process_expenses  # import your main function

# Page config for wider layout
st.set_page_config(page_title="Expense Parser Tool", layout="wide")

# Title card
st.markdown(
    """
    <div style="background-color:#4CAF50;padding:20px;border-radius:10px">
        <h2 style="color:white;text-align:center;">Expense Parser Tool 📄➡️📊</h2>
        <p style="color:white;text-align:center;">Upload your bank statements (PDF) and generate expense reports quickly.</p>
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown("---")

# ------------------------------
# Upload Section (full width)
# ------------------------------
st.markdown(
    """
    <div style="background-color:#f0f2f6;padding:15px;border-radius:10px">
        <h3>📂 Upload Bank Statements</h3>        
    </div>
    """,
    unsafe_allow_html=True,
    help="Drag and drop PDF files or use the file uploader below."
)
uploaded_files = st.file_uploader(
    "Upload bank statements (PDF)",
    type=["pdf"],
    accept_multiple_files=True,
    help="Drag and drop PDF files or use the file uploader below."
)

# ------------------------------
# Output Folder Section (full width)
# ------------------------------
st.markdown(
    """
    <div style="background-color:#f0f2f6;padding:15px;border-radius:10px;margin-top:10px">
        <h3>📁 Output Folder</h3>        
    </div>
    """,
    unsafe_allow_html=True,
    help="Enter the folder name where processed reports will be saved."
)
output_dir = st.text_input(
    "Enter output folder name",
    "reports",
    help="Enter the folder name where processed reports will be saved."
)
process_button = st.button("Process")

# ------------------------------
# Processing Logic
# ------------------------------
if uploaded_files and process_button:
    os.makedirs(output_dir, exist_ok=True)

    # Save uploaded files temporarily to output folder
    for pdf in uploaded_files:
        pdf_path = os.path.join(output_dir, pdf.name)
        with open(pdf_path, "wb") as f:
            f.write(pdf.getbuffer())

    # Processing PDFs
    st.markdown(
        """
        <div style="background-color:#E3F2FD;padding:15px;border-radius:10px">
            <h3>⚡ Processing PDF Files</h3>
            <p>Generating reports from uploaded PDFs...</p>
        </div>
        """,
        unsafe_allow_html=True
    )
    parse_progress = st.progress(0)
    process_expenses(
        pdf_dir_override=output_dir,
        reports_dir_override=output_dir
    )
    parse_progress.progress(1.0)

    st.success(f"Reports generated in {os.path.abspath(output_dir)} ✅")

    # Download links in grid
    st.markdown(
        """
        <div style="background-color:#FFFDE7;padding:15px;border-radius:10px">
            <h3>⬇️ Download Generated Reports</h3>
            <p>Click the buttons below to download each report.</p>
        </div>
        """,
        unsafe_allow_html=True
    )
    report_files = [f for f in os.listdir(output_dir) if os.path.isfile(os.path.join(output_dir, f))]
    num_cols = 3
    cols = st.columns(num_cols)
    for i, report in enumerate(report_files):
        col = cols[i % num_cols]
        with col:
            report_path = os.path.join(output_dir, report)
            with open(report_path, "rb") as f:
                data = f.read()
            st.download_button(
                label=f"{report} ⬇️",
                data=data,
                file_name=report
            )

    st.info("All files processed. You can download the reports above.")

    # ------------------------------
    # Cleanup temporary files button
    # ------------------------------
    if st.button("Clear Uploaded & Generated Files"):
        shutil.rmtree(output_dir)
        st.success("All uploaded PDFs and generated reports have been deleted ✅")
