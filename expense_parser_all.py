import sys
import os
import re
import pdfplumber
import pandas as pd
from openai import OpenAI
import json
import matplotlib.pyplot as plt
import zipfile
import openai
import logging
from io import BytesIO
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from dotenv import load_dotenv

# Base path: works for both .py and .exe
if getattr(sys, 'frozen', False):  # running as exe
    base_path = os.path.dirname(sys.executable)
else:
    base_path = os.path.dirname(os.path.abspath(__file__))

# Folders relative to base path
log_dir = os.path.join(base_path, "logs")
#pdf_dir = os.environ.get("PDF_DIR", os.path.join(base_path, "data"))
pdf_dir=os.path.join(base_path, "data")
reports_dir = os.path.join(base_path, "reports")

# Ensure folders exist
os.makedirs(log_dir, exist_ok=True)
os.makedirs(pdf_dir, exist_ok=True)
os.makedirs(reports_dir, exist_ok=True)

# -----------------------------
# INITIALIZE
# -----------------------------
def main(pdf_dir_override=None, reports_dir_override=None):
    
    '''# Use passed folders or fallback to defaults
    if pdf_dir:
        os.environ["PDF_DIR"] = pdf_dir  # update environment variable if provided
    else:
        pdf_dir = os.environ.get("PDF_DIR", os.path.join(base_path, "data"))

    if reports_dir:
        # override default reports_dir
        reports_dir = reports_dir
    else:
        reports_dir = os.path.join(base_path, "reports")'''

    # Ensure folders exist
    #os.makedirs(pdf_dir, exist_ok=True)
    #os.makedirs(reports_dir, exist_ok=True)

    #log_dir = "c:/cca/logs"
    #os.makedirs(log_dir, exist_ok=True)

       # Use passed folders or fallback to defaults
    pdf_dir_local = pdf_dir_override or pdf_dir
    reports_dir_local = reports_dir_override or reports_dir

    os.makedirs(pdf_dir_local, exist_ok=True)
    os.makedirs(reports_dir_local, exist_ok=True)

    log_file = os.path.join(log_dir, "expense_parser.log")

    logging.basicConfig(
        level=logging.INFO,  # Change to DEBUG for more verbose logs
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file, mode="a", encoding="utf-8"),
            logging.StreamHandler()  # Also prints to console
        ]
    )

    logger = logging.getLogger(__name__)
    
    load_dotenv()  # this will load the .env file
    openai.api_key = os.getenv("OPENAI_API_KEY")
    os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
    
    #client = OpenAI()
    client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    processed_data ={}
    #pdf_dir = os.environ.get("PDF_DIR", "c:/cca/data")
    #reports_dir = "c:/cca/reports"
    os.makedirs(reports_dir, exist_ok=True)

    # budget (same for all categories)
    BUDGET_LIMIT = 2000

    # canonical category list + stable colors (consistent across all charts)
    CATEGORIES = [
        "Grocery", "Shopping", "Travel", "Food and Dining",
        "Entertainment", "School", "Utilities and Bills", "Fuel", "Others"
    ]
    CATEGORY_COLORS = {
        c: clr for c, clr in zip(
            CATEGORIES,
            list(plt.cm.tab20.colors[:len(CATEGORIES)])
        )
    }

    # month mapping (tolerant: "Jun", "june", "06", etc.)
    MONTH_MAP = {
        "JAN": ("January", 1), "FEB": ("February", 2), "MAR": ("March", 3), "APR": ("April", 4),
        "MAY": ("May", 5), "JUN": ("June", 6), "JUL": ("July", 7), "AUG": ("August", 8),
        "SEP": ("September", 9), "OCT": ("October", 10), "NOV": ("November", 11), "DEC": ("December", 12),
        "01": ("January", 1), "02": ("February", 2), "03": ("March", 3), "04": ("April", 4),
        "05": ("May", 5), "06": ("June", 6), "07": ("July", 7), "08": ("August", 8),
        "09": ("September", 9), "10": ("October", 10), "11": ("November", 11), "12": ("December", 12)
    }

    # -----------------------------
    # BANK CONFIGS (unchanged logic)
    # -----------------------------
    BANK_CONFIGS = {
        "FAB": {  # No credit suffix; amount column is UAE Dirham Amount Debit
            "type": "regex",
            "date_column": "Transaction date",
            "desc_column": "description",
            "amount_column": "UAE Dirham Amount Debit"
        },
        "WIO": {
            "type": "keyword",
            "start_parsing_keyword": "Transactions",
            "date_column": "date",
            "desc_column": "description",
            "amount_column": "amount",
            "credit_prefix": "+"
        },
        "ENBD": {
            "type": "regex",
            "date_column": "Transaction date",
            "desc_column": "description",
            "amount_column": "amount",
            "credit_suffix": "CR"
        }
    }

    # -----------------------------
    # EXISTING FUNCTIONS (unchanged behavior)
    # -----------------------------
    def extract_pdf_text(file_path, bank_config):
        text = ""
        if bank_config.get("type") == "keyword":
            start_parsing = False
            with pdfplumber.open(file_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if not page_text:
                        continue
                    for line in page_text.split("\n"):
                        if not start_parsing and bank_config["start_parsing_keyword"] in line:
                            start_parsing = True
                            continue
                        if start_parsing:
                            text += line + "\n"
        else:
            with pdfplumber.open(file_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
        return text

    def filter_transaction_lines(lines):
        trx_lines = []
        regex = re.compile(r"^\s*(\d{2}[/-]\d{2}(?:[/-]\d{4})?)\s+(.+?)\s+([\d,]+(?:\.\d{1,2})?)\s*$")
        for line in lines:
            if regex.match(line):
                trx_lines.append(line.strip())
        return trx_lines

    def chunk_text(lines, max_chars=3000):
        chunks, current, total_chars = [], [], 0
        for line in lines:
            if total_chars + len(line) > max_chars:
                chunks.append("\n".join(current))
                current, total_chars = [], 0
            current.append(line)
            total_chars += len(line)
        if current:
            chunks.append("\n".join(current))
        return chunks

    def parse_with_ai(text_chunk):
        prompt = f"""
    You are a financial transaction parser.
    Input lines are pre-filtered.
    Output JSON strictly as:
    {{
    "transactions": [
        {{
        "transaction_date": "DD-MM-YYYY",
        "description": "text",
        "amount": 123.45,
        "category": "Grocery/Shopping/Travel/Food and Dining/Entertainment/School/Utilities and Bills/Fuel/Others"
        }}
    ]
    }}
    Rules:
    - Use date and description exactly as in text.
    - Amount must be numeric only (remove commas).
    - Ignore credits/refunds (for WIO/ENBD handled separately).
    """
        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a strict financial data parser."},
                    {"role": "user", "content": prompt + "\n\nText:\n" + text_chunk}
                ],
                temperature=0,
                response_format={"type": "json_object"}
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"⚠️ API Error: {e}")
            logging.error(f"⚠️ API Error: {e}")
            return None

    # -----------------------------
    # Helpers
    # -----------------------------
    def parse_filename(fname):
        """
        Expected: BANKNAME_month_year.pdf (case-insensitive for month)
        Returns (bank, month_name, month_num, year) or (None, None, None, None) if not matched.
        """
        name = os.path.splitext(fname)[0]
        parts = name.split("_")
        if len(parts) < 3:
            return (None, None, None, None)
        bank = parts[0].upper()
        month_token = parts[1].strip().upper()[:3]  # Jun, JUN, june -> JUN
        year_match = re.search(r"(\d{4})", name)
        year = year_match.group(1) if year_match else None
        if month_token in MONTH_MAP:
            month_name, month_num = MONTH_MAP[month_token]
        else:
            # try full month names
            m3 = parts[1].strip().upper()
            for k, v in MONTH_MAP.items():
                if v[0].upper() == m3:
                    month_name, month_num = v
                    break
            else:
                return (None, None, None, None)
        return (bank, month_name, month_num, year)

    def autopct_pct(pct):
        # Show label only if >= 2% to avoid clutter
        return f"{pct:.1f}%" if pct >= 2 else ""

    def fig_to_rl_image(fig, width=None, height=None):
        buf = BytesIO()
        fig.savefig(buf, format="png", dpi=180, bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)
        return Image(buf, width=width, height=height)

    # -----------------------------
    # PDF Report Generation (improved charts UI)
    # -----------------------------
    def generate_pdf_report(bank_name, year, monthly_dfs, combined_df, reports_dir_local):
        pdf_file = os.path.join(reports_dir_local, f"{bank_name}_{year}_REPORT.pdf")
        doc = SimpleDocTemplate(pdf_file, pagesize=A4)
        styles = getSampleStyleSheet()
        elements = []

        elements.append(Paragraph(f"📊 {bank_name} Yearly Expense Report - {year}", styles["Title"]))
        elements.append(Spacer(1, 12))

        # monthly sections (only months present)
        months_sorted = sorted(
            monthly_dfs.keys(),
            key=lambda m: [v[1] for k, v in MONTH_MAP.items() if v[0] == m][0]
        )

        for month in months_sorted:
            df = monthly_dfs[month]
            elements.append(Paragraph(f"📌 {month}", styles["Heading2"]))
            elements.append(Spacer(1, 6))

            # --- Transactions table (as requested: data first) ---
            table_data = [list(df.columns)] + df.values.tolist()
            table = Table(table_data, repeatRows=1)
            table.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#343a40")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.whitesmoke, colors.HexColor("#f1f3f5")])
            ]))
            elements.append(table)
            elements.append(Spacer(1, 10))

            # --- Monthly summary table with budget ---
            summary = df.groupby("Category")["Amount (AED)"] \
                        .sum().reindex(CATEGORIES, fill_value=0).reset_index()
            summary["Budget Limit"] = BUDGET_LIMIT
            summary["Over Budget"] = summary["Amount (AED)"] > summary["Budget Limit"]

            smry_data = [["Category", "Amount (AED)", "Budget Limit", "Over Budget"]]
            for _, r in summary.iterrows():
                smry_data.append([r["Category"], f'{r["Amount (AED)"]:.2f}', f"{BUDGET_LIMIT}", "Yes" if r["Over Budget"] else "No"])

            smry_tbl = Table(smry_data, repeatRows=1)
            smry_tbl.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#495057")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
            ]))
            elements.append(Paragraph("Summary", styles["Heading3"]))
            elements.append(smry_tbl)
            elements.append(Spacer(1, 10))

            # --- Monthly pie chart (clean legend & labels, consistent colors) ---
            fig = plt.figure(figsize=(5, 4))
            amounts = summary["Amount (AED)"].values
            color_list = [CATEGORY_COLORS[c] for c in summary["Category"]]
            wedges, _, autotexts = plt.pie(
                amounts,
                labels=None,            # keep slices clean
                autopct=autopct_pct,    # only show >=2%
                startangle=90,
                colors=color_list
            )
            plt.title(f"{month} Spending by Category")
            # Legend in two columns below the chart (separate from slices)
            plt.legend(
                wedges,
                summary["Category"],
                title="Category",
                ncol=2,
                bbox_to_anchor=(0.5, -0.15),
                loc="upper center",
                frameon=False
            )
            plt.tight_layout()
            elements.append(fig_to_rl_image(fig, width=420, height=320))
            elements.append(Spacer(1, 18))
            elements.append(PageBreak())

        # --- Yearly combined summary ---
        elements.append(Paragraph("📌 Yearly Summary", styles["Heading2"]))
        combined_summary = combined_df.groupby("Category")["Amount (AED)"] \
                                    .sum().reindex(CATEGORIES, fill_value=0).reset_index()
        combined_summary["Budget Limit"] = BUDGET_LIMIT
        combined_summary["Over Budget"] = combined_summary["Amount (AED)"] > combined_summary["Budget Limit"]

        y_data = [["Category", "Amount (AED)", "Budget Limit", "Over Budget"]]
        for _, r in combined_summary.iterrows():
            y_data.append([r["Category"], f'{r["Amount (AED)"]:.2f}', f"{BUDGET_LIMIT}", "Yes" if r["Over Budget"] else "No"])

        y_tbl = Table(y_data, repeatRows=1)
        y_tbl.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#495057")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
        ]))
        elements.append(y_tbl)
        elements.append(Spacer(1, 10))

        # Yearly pie
        fig = plt.figure(figsize=(6, 5))
        amounts = combined_summary["Amount (AED)"].values
        color_list = [CATEGORY_COLORS[c] for c in combined_summary["Category"]]
        wedges, _, _ = plt.pie(
            amounts,
            labels=None,
            autopct=autopct_pct,
            startangle=90,
            colors=color_list
        )
        plt.title(f"{year} Spending by Category")
        plt.legend(
            wedges, combined_summary["Category"],
            title="Category", ncol=2,
            bbox_to_anchor=(0.5, -0.15), loc="upper center", frameon=False
        )
        plt.tight_layout()
        elements.append(fig_to_rl_image(fig, width=460, height=360))
        elements.append(Spacer(1, 18))

        # Trend analysis: stacked monthly bars by category
        tdf = combined_df.copy()
        tdf["MonthNum"] = pd.to_datetime(tdf["Transaction Date"], dayfirst=True, errors="coerce").dt.month
        tdf["Month"] = pd.to_datetime(tdf["Transaction Date"], dayfirst=True, errors="coerce").dt.strftime("%b")
        trend = tdf.groupby(["MonthNum", "Month", "Category"])["Amount (AED)"] \
                .sum().unstack(fill_value=0).reindex(columns=CATEGORIES)
        trend = trend.sort_index(level=0)
        if not trend.empty:
            fig = plt.figure(figsize=(7, 4.5))
            ax = trend.droplevel(0).plot(
                kind="bar",
                stacked=True,
                color=[CATEGORY_COLORS[c] for c in trend.columns],
                width=0.85,
                ax=plt.gca()
            )
            ax.set_title(f"{year} Monthly Trend by Category")
            ax.set_ylabel("Amount (AED)")
            ax.set_xlabel("Month")
            ax.legend(title="Category", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.15), loc="upper center")
            plt.xticks(rotation=0)
            plt.tight_layout()
            elements.append(fig_to_rl_image(fig, width=500, height=360))
            elements.append(Spacer(1, 16))

        doc.build(elements)
        print(f"📑 Consolidated PDF saved: {pdf_file}")
        logging.info(f"📑 Consolidated PDF saved: {pdf_file}")

    # -----------------------------
    # PROCESS FILES → EXCEL (per bank/year) + PDF
    # -----------------------------
    pdf_files = [f for f in os.listdir(pdf_dir_local) if f.lower().endswith(".pdf")]
    if not pdf_files:
        raise FileNotFoundError("No PDF files found in directory.")

    # Group files by bank -> year -> list of files
    by_bank_year = {}
    for f in pdf_files:
        bank, month_name, month_num, year = parse_filename(f)
        if not bank or not year or not month_name:
            print(f"⚠️ Skipping (name format): {f}")
            logging.info(f"⚠️ Skipping (name format): {f}")
            continue
        if bank not in BANK_CONFIGS:
            print(f"⚠️ Skipping (unknown bank): {f}")
            logging.info(f"⚠️ Skipping (unknown bank): {f}")
            continue
        by_bank_year.setdefault(bank, {}).setdefault(year, []).append((f, month_name, month_num))

    for bank_name, years in by_bank_year.items():
        for year, items in years.items():
            # Per bank-year collections
            monthly_dfs = {}   # month_name -> df
            monthly_logs = {}  # month_name -> logs df
            combined_df = pd.DataFrame()
            combined_logs = pd.DataFrame()

            # Process each provided month file ONLY
            for f, month_name, month_num in sorted(items, key=lambda x: x[2]):
                print(f"\n📄 Processing file: {f}")
                logging.info(f"\n📄 Processing file: {f}")
                BANK_CONFIG = BANK_CONFIGS[bank_name]
                pdf_path = os.path.join(pdf_dir, f)

                pdf_text = extract_pdf_text(pdf_path, BANK_CONFIG)
                lines = pdf_text.splitlines()

                # regex pre-filter for regex banks
                if BANK_CONFIG.get("type") == "regex":
                    filtered_lines = filter_transaction_lines(lines)
                else:
                    filtered_lines = lines

                chunks = chunk_text(filtered_lines)
                print(f"➡️ Split into {len(chunks)} chunks for processing.")
                logging.info(f"➡️ Split into {len(chunks)} chunks for processing.")

                transactions, logs = [], []

                for i, chunk in enumerate(chunks, 1):
                    raw = parse_with_ai(chunk)
                    if not raw:
                        logs.append({"source_file": f, "chunk": i, "status": "API Error", "reason": "AI call failed", "snippet": chunk[:500]})
                        continue
                    try:
                        parsed = json.loads(raw)
                        chunk_txns = parsed.get("transactions", [])
                        if not chunk_txns:
                            logs.append({"source_file": f, "chunk": i, "status": "Empty result", "reason": "No txns in chunk", "snippet": chunk[:500]})
                        else:
                            for t in chunk_txns:
                                amt_raw = t.get("amount", "")
                                amt_str = str(amt_raw).replace(",", "").strip()

                                # Ignore credits/refunds for WIO/ENBD only (FAB has debit column)
                                if bank_name != "FAB":
                                    if BANK_CONFIG.get("credit_suffix") and amt_str.upper().endswith(BANK_CONFIG["credit_suffix"]):
                                        logs.append({"source_file": f, "chunk": i, "status": "Ignored", "reason": f"Credit suffix {BANK_CONFIG['credit_suffix']}", "snippet": str(t)})
                                        continue
                                    if BANK_CONFIG.get("credit_prefix") and amt_str.startswith(BANK_CONFIG["credit_prefix"]):
                                        logs.append({"source_file": f, "chunk": i, "status": "Ignored", "reason": f"Credit prefix {BANK_CONFIG['credit_prefix']}", "snippet": str(t)})
                                        continue

                                if not amt_str.replace(".", "").isdigit():
                                    logs.append({"source_file": f, "chunk": i, "status": "Ignored", "reason": "Non-numeric amount", "snippet": str(t)})
                                    continue

                                amt = float(amt_str)
                                if amt > 0:
                                    transactions.append({
                                        "Transaction Date": t.get("transaction_date"),
                                        "Description": t.get("description"),
                                        "Amount (AED)": amt,
                                        "Category": t.get("category")
                                    })
                                else:
                                    logs.append({"source_file": f, "chunk": i, "status": "Ignored", "reason": "Amount <= 0", "snippet": str(t)})
                    except Exception as e:
                        logs.append({"source_file": f, "chunk": i, "status": "JSON Parse Error", "reason": str(e), "snippet": raw[:500] if raw else "No data"})

                # Build DataFrames
                df = pd.DataFrame(transactions).drop_duplicates()
                log_df = pd.DataFrame(logs)

                if not df.empty:
                    df["Transaction Date"] = pd.to_datetime(df["Transaction Date"], dayfirst=True, errors="coerce")
                    df = df.sort_values(by="Transaction Date")
                    df["Transaction Date"] = df["Transaction Date"].dt.strftime("%d-%m-%Y")

                # store in memory for Excel and PDF consolidation
                monthly_dfs[month_name] = df
                monthly_logs[month_name] = log_df
                combined_df = pd.concat([combined_df, df], ignore_index=True)
                combined_logs = pd.concat([combined_logs, log_df], ignore_index=True)
                # store processed monthly data for combined Excel later
                bank_name_clean = bank_name.strip().upper()
                processed_data.setdefault(bank_name_clean, {})[year] = monthly_dfs

            # --- Build ONE Excel per bank/year (with only the months we have) ---
            excel_file = os.path.join(reports_dir_local, f"{bank_name}_{year}_REPORT.xlsx")
            with pd.ExcelWriter(excel_file, engine="xlsxwriter") as writer:
                # per-month sheets (Transactions, Summary WITH PIE, Logs)
                for month in sorted(monthly_dfs.keys(), key=lambda m: [v[1] for k, v in MONTH_MAP.items() if v[0] == m][0]):
                    mdf = monthly_dfs[month]
                    ldf = monthly_logs.get(month, pd.DataFrame())

                    if not mdf.empty:
                        # Transactions
                        mdf.to_excel(writer, sheet_name=f"{month}_Transactions", index=False)

                        # Summary + pie chart
                        summary = mdf.groupby("Category")["Amount (AED)"] \
                                    .sum().reindex(CATEGORIES, fill_value=0).reset_index()
                        summary["percentage"] = (summary["Amount (AED)"] /
                                                summary["Amount (AED)"].sum() * 100).round(2)
                        summary.to_excel(writer, sheet_name=f"{month}_Summary", index=False)

                        wb = writer.book
                        ws = writer.sheets[f"{month}_Summary"]
                        if summary["Amount (AED)"].sum() > 0:
                            chart = wb.add_chart({"type": "pie"})
                            # categories and values range: A2:A10, B2:B10 (fixed 9 rows for CATEGORIES)
                            last_row = 1 + len(summary)
                            chart.add_series({
                                "name": f"{month} Spending by Category",
                                "categories": [f"{month}_Summary", 1, 0, last_row - 1, 0],
                                "values": [f"{month}_Summary", 1, 1, last_row - 1, 1],
                                "data_labels": {"percentage": True}
                            })
                            chart.set_title({"name": f"{month} Spending Breakdown"})
                            ws.insert_chart("F2", chart)

                        # Logs (include ignored reasons)
                        if not ldf.empty:
                            ldf.to_excel(writer, sheet_name=f"{month}_Logs", index=False)


                # Combined sheets
                if not combined_df.empty:
                    combined_df.to_excel(writer, sheet_name="Combined_Transactions", index=False)

                    csum = combined_df.groupby("Category")["Amount (AED)"] \
                                    .sum().reindex(CATEGORIES, fill_value=0).reset_index()
                    csum["percentage"] = (csum["Amount (AED)"] / csum["Amount (AED)"].sum() * 100).round(2)
                    csum.to_excel(writer, sheet_name="Combined_Summary", index=False)

                    wb = writer.book
                    ws = writer.sheets["Combined_Summary"]
                    if csum["Amount (AED)"].sum() > 0:
                        chart = wb.add_chart({"type": "pie"})
                        last_row = 1 + len(csum)
                        chart.add_series({
                            "name": f"{year} Combined Spending by Category",
                            "categories": ["Combined_Summary", 1, 0, last_row - 1, 0],
                            "values": ["Combined_Summary", 1, 1, last_row - 1, 1],
                            "data_labels": {"percentage": True}
                        })
                        chart.set_title({"name": f"{year} Combined Spending Breakdown"})
                        ws.insert_chart("F2", chart)

                if not combined_logs.empty:
                    combined_logs.to_excel(writer, sheet_name="Combined_Logs", index=False)

            print(f"✅ Excel saved: {excel_file}")
            logging.info(f"✅ Excel saved: {excel_file}")

            # --- Build ONE PDF per bank/year (monthly sections + yearly summary + trend) ---
            if not combined_df.empty:
                generate_pdf_report(bank_name, year, monthly_dfs, combined_df, reports_dir_local)



    #new start - create one consolidated excel for all records
    # ------------------- Combined Excel for all banks (yearly) -------------------
    # Collect all years from by_bank_year
    all_years = set()
    for bank_name2, years_dict in processed_data.items():
        for year_key in years_dict.keys():
            all_years.add(year_key)

    
    for year_loop in sorted(all_years):  # Loop over each year
        all_banks_data = []

        # Collect all processed monthly_dfs from each bank for this year
       #comment for bank_name2, years2 in by_bank_year.items():
            #comment if year_loop in years2:
                #comment monthly_dfs_for_bank_year = processed_data[bank_name2][year_loop]

        for bank_name2, years2 in by_bank_year.items():
            bank_name2_clean = bank_name2.strip().upper()  # normalize bank name
            if bank_name2_clean not in processed_data:
                print(f"⚠️ Skipping unknown bank in combined report: '{bank_name2}'")
                logging.info(f"⚠️ Skipping unknown bank in combined report: '{bank_name2}'")
                continue
            if year_loop not in processed_data[bank_name2_clean]:
                continue        
            monthly_dfs_for_bank_year = processed_data[bank_name2_clean][year_loop]

            for month_name, df_month in monthly_dfs_for_bank_year.items():
                if not df_month.empty:
                    df_copy = df_month.copy()
                    df_copy["Bank"] = bank_name2
                    all_banks_data.append(df_copy)

        if all_banks_data:
            final_df = pd.concat(all_banks_data, ignore_index=True)
            final_df["Transaction Date"] = pd.to_datetime(final_df["Transaction Date"], dayfirst=True, errors="coerce")
            final_df = final_df.sort_values(by="Transaction Date")
            final_df["Transaction Date"] = final_df["Transaction Date"].dt.strftime("%d-%m-%Y")

            # File path for yearly combined report
            combined_excel_file = os.path.join(reports_dir, f"expenseReport_{year_loop}.xlsx")
            with pd.ExcelWriter(combined_excel_file, engine="xlsxwriter") as writer:
                # 1️⃣ All Transactions sheet
                final_df.to_excel(writer, sheet_name="All_Data", index=False)

                # 2️⃣ Summary sheet
                summary_all = final_df.groupby("Category")["Amount (AED)"].sum().reindex(CATEGORIES, fill_value=0).reset_index()
                summary_all["Percentage"] = (summary_all["Amount (AED)"] / summary_all["Amount (AED)"].sum() * 100).round(2)
                summary_all = summary_all.sort_values(by="Amount (AED)", ascending=False)
                summary_all.to_excel(writer, sheet_name="Summary", index=False)

                # Pie chart
                wb = writer.book
                ws = writer.sheets["Summary"]
                if summary_all["Amount (AED)"].sum() > 0:
                    pie = wb.add_chart({"type": "pie"})
                    last_row = len(summary_all)
                    pie.add_series({
                        "name": f"{year_loop} Category-wise Spending",
                        "categories": ["Summary", 1, 0, last_row, 0],
                        "values": ["Summary", 1, 1, last_row, 1],
                        "data_labels": {"percentage": True}
                    })
                    pie.set_title({"name": f"{year_loop} Category Spending Breakdown"})
                    ws.insert_chart("E2", pie)

                # 3️⃣ Trend sheet (monthly stacked bar chart)
                final_df["Month"] = pd.to_datetime(final_df["Transaction Date"], dayfirst=True, errors="coerce").dt.strftime("%b")
                trend_df = final_df.groupby(["Month", "Category"])["Amount (AED)"].sum().unstack(fill_value=0).reindex(columns=CATEGORIES)

                # Ensure all months of the year are in order
                months_order = pd.date_range(start=f"1-1-{year_loop}", end=f"31-12-{year_loop}", freq="M").strftime("%b")
                trend_df = trend_df.reindex(index=months_order, fill_value=0)
                trend_df.to_excel(writer, sheet_name="Trend", index=True)

                ws_trend = writer.sheets["Trend"]
                bar = wb.add_chart({"type": "column", "subtype": "stacked"})
                for i, cat in enumerate(CATEGORIES):
                    bar.add_series({
                        "name":       ["Trend", 0, i+1],
                        "categories": ["Trend", 1, 0, len(trend_df), 0],
                        "values":     ["Trend", 1, i+1, len(trend_df), i+1],
                    })
                bar.set_title({"name": f"{year_loop} Monthly Trend by Category"})
                bar.set_y_axis({"name": "Amount (AED)"})
                bar.set_x_axis({"name": "Month"})
                ws_trend.insert_chart("G2", bar)

            print(f"✅ Yearly combined Excel saved: {combined_excel_file}")
            logging.info(f"✅ Yearly combined Excel saved: {combined_excel_file}")


    #new end
pass

if __name__ == "__main__":
    main()
