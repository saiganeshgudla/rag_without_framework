"""
create_test_pdf.py - Generate a sample text-based resume PDF for testing.
Run once: python create_test_pdf.py
"""

from fpdf import FPDF


def build_pdf(output_path="test_resume.pdf"):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_margins(20, 20, 20)
    pdf.add_page()

    W = pdf.epw  # effective page width (accounts for margins)

    def h(size, bold=False, italic=False):
        style = ("B" if bold else "") + ("I" if italic else "")
        pdf.set_font("Helvetica", style, size)

    def line(text="", size=10, bold=False, italic=False, align="L", spacing=6):
        h(size, bold, italic)
        pdf.multi_cell(W, spacing, text, align=align)

    def section(title):
        pdf.ln(2)
        h(12, bold=True)
        pdf.set_fill_color(220, 220, 220)
        pdf.multi_cell(W, 8, title, fill=True)
        pdf.ln(1)

    def bullet(text):
        h(10)
        pdf.multi_cell(W, 6, "- " + text)

    # ── Header ────────────────────────────────────────────────────────────────
    line("Aditya Pichikala", size=22, bold=True, align="C", spacing=12)
    line("aditya@example.com  |  +91 98765 43210  |  github.com/adityapichikala",
         size=9, align="C")
    pdf.ln(4)

    # ── Summary ───────────────────────────────────────────────────────────────
    section("PROFESSIONAL SUMMARY")
    line(
        "Motivated Computer Science student with hands-on experience in Python, "
        "machine learning, and full-stack development. Passionate about building "
        "intelligent systems, RAG pipelines, and deploying scalable web applications. "
        "Strong foundation in data structures, algorithms, and software engineering best practices."
    )

    # ── Education ─────────────────────────────────────────────────────────────
    section("EDUCATION")
    line("B.Tech in Computer Science and Engineering", bold=True)
    line("XYZ University  |  2022 to 2026  |  CGPA: 8.7 / 10")

    # ── Skills ────────────────────────────────────────────────────────────────
    section("TECHNICAL SKILLS")
    skills = [
        ("Languages", "Python, JavaScript, SQL, C++"),
        ("ML and AI", "Scikit-learn, PyTorch, FAISS, Sentence Transformers, Gemini API"),
        ("Web", "FastAPI, React, Node.js, REST APIs, WebSockets"),
        ("Databases", "PostgreSQL, Supabase, MongoDB"),
        ("Tools", "Git, Docker, Linux, VSCode"),
    ]
    for label, val in skills:
        line(f"{label}: {val}")

    # ── Experience ────────────────────────────────────────────────────────────
    section("WORK EXPERIENCE")
    line("ML Engineer Intern -- TechStartup Pvt. Ltd.  (Jun 2025 to Aug 2025)", bold=True)
    bullet("Built a RAG pipeline from scratch using FAISS and Sentence Transformers without LangChain.")
    bullet("Reduced document retrieval latency by 40% by optimising chunk size and overlap parameters.")
    bullet("Integrated Google Gemini API for context-grounded Q and A over internal company documents.")
    pdf.ln(3)

    line("Full-Stack Developer Intern -- WebAgency Co.  (Dec 2024 to Feb 2025)", bold=True)
    bullet("Developed a FastAPI backend with PostgreSQL and Supabase for an attendance management system.")
    bullet("Implemented attendance rectification feature with role-based access control.")
    bullet("Built React frontend with real-time updates using WebSockets.")

    # ── Projects ──────────────────────────────────────────────────────────────
    section("PROJECTS")
    line("Resume Analyser (RAG from Scratch)", bold=True)
    line("Tech: Python, FAISS, Sentence Transformers, Gemini API, pdfplumber", italic=True, size=9)
    line(
        "End-to-end RAG system that extracts text from PDF resumes, chunks and embeds them "
        "using all-MiniLM-L6-v2, indexes with FAISS, and answers job-fit questions via Gemini."
    )
    pdf.ln(2)

    line("AutoML Platform", bold=True)
    line("Tech: FastAPI, React, PyCaret, Celery, WebSockets", italic=True, size=9)
    line(
        "No-code ML platform where users upload CSV files and trigger automated model training "
        "with real-time progress updates and feature importance visualisation."
    )
    pdf.ln(2)

    line("Anime Image Generation", bold=True)
    line("Tech: Stable Diffusion Turbo, FastAPI, React", italic=True, size=9)
    line(
        "Low-latency local image generation system with domain validation to ensure "
        "prompts stay within anime-themed content."
    )

    # ── Achievements ──────────────────────────────────────────────────────────
    section("ACHIEVEMENTS")
    bullet("Ranked in top 5% of participants in a national-level ML hackathon (500+ teams).")
    bullet("Published a technical blog on RAG pipelines with 2,000+ monthly readers.")
    bullet("Open-source contributor: 3 merged PRs to popular Python ML libraries.")

    pdf.output(output_path)
    print(f"[Done] Test resume PDF created: {output_path}")


if __name__ == "__main__":
    build_pdf()
