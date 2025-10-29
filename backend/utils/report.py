from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

def create_report(allocation: dict, user_info: dict, advice: str, out_path="data/uploads/report.pdf"):
    c = canvas.Canvas(out_path, pagesize=letter)
    w, h = letter
    c.setFont("Helvetica-Bold", 16)
    c.drawString(40, h-60, "Smart Investment Planner - Personalized Report")
    c.setFont("Helvetica", 10)
    c.drawString(40, h-90, f"Name: {user_info.get('name', 'N/A')}")
    c.drawString(40, h-105, f"Income: {user_info.get('income')}, Expenses: {user_info.get('expenses')}, Age: {user_info.get('age')}, Risk: {user_info.get('risk')}")
    c.drawString(40, h-130, "Allocation (absolute amounts):")
    y = h-150
    for k in ["SIP","FD","Stocks","Total"]:
        if k in allocation:
            c.drawString(50, y, f"{k}: {round(allocation[k],2)}")
            y -= 16
    y -= 8
    c.drawString(40, y, "AI Advice:")
    y -= 16
    text = advice or "No advice."
    for line in text.splitlines():
        if y < 40:
            c.showPage()
            y = h-80
        c.drawString(50, y, line[:120])
        y -= 14
    c.save()
    return out_path
