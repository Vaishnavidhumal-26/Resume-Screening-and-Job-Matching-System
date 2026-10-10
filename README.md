# Resume Screening and Job Matching using Deep Learning & NLP

## Project Overview
This project compares a candidate's resume with a job description using skill-keyword matching and semantic text similarity. It supports PDF and DOCX resumes and presents the results in a Streamlit web app.

The app reports:
- **Skill Match (%)** — the percentage of recognized job-description skills also found in the resume.
- **Semantic Similarity (%)** — cosine similarity between sentence-transformer embeddings of the resume and job description.
- **Combined Match Score (%)** — an experimental 50/50 blend of skill match and semantic similarity.
- **Matched Skills** — recognized skills found in both texts.
- **Missing Skills / Skills to Develop** — recognized job-description skills not found in the resume.

> **Important:** These are experimental matching indicators, not hiring probabilities, validated model accuracy, or guarantees of suitability. Results depend on extracted text, the configured keyword list, and the model.

## Features
- Upload a resume in **PDF** or **DOCX** format.
- Paste a job description into the app.
- Extract resume text.
- Identify skills using a configurable keyword list.
- Compare texts using a sentence-transformer model.
- Display scores and matched/missing skills in Streamlit.

## Technologies Used
- Python
- Streamlit
- Sentence Transformers (`sentence-transformers/all-MiniLM-L6-v2`)
- scikit-learn (cosine similarity)
- PyMuPDF (PDF text extraction)
- python-docx (DOCX text extraction)
- Regular expressions (text cleaning and keyword matching)
- Pandas and NumPy are installed in the project environment but are not required by the current app's main workflow.

## Project Structure
```text
Resume_Job_Matching/
├── app.py
├── Resume_Job_Matching.ipynb
└── README.md
```

## Requirements
- Python 3.10 or a compatible Python version
- Internet access for the first model download, unless the model is already cached
- Required Python packages

Install dependencies in a terminal:
```bash
python -m pip install streamlit sentence-transformers scikit-learn pymupdf python-docx pandas numpy
```

## Run the Application
1. Open Terminal or PowerShell.
2. Change to the project folder. Update the path if your folder is elsewhere:
   ```powershell
   cd "C:\Users\dhuma\Desktop\Data Science\Deep Learning\Resume_Job_Matching"
   ```
3. Run:
   ```powershell
   python -m streamlit run app.py
   ```
4. Open the local URL shown in the terminal (usually `http://localhost:8501`).
5. Upload a PDF/DOCX resume, paste a job description, and click **Analyze Resume**.

Keep the terminal open while using the app. To stop the app, press `Ctrl+C` in the terminal.

## How Matching Works
1. Extract text from the uploaded PDF or DOCX resume.
2. Clean the resume and job-description text.
3. Find skills included in the app's `skill_keywords` list.
4. Calculate skill match as:
   `matched recognized skills / recognized required skills × 100`
5. Encode both texts with `all-MiniLM-L6-v2` and calculate cosine similarity.
6. Calculate the combined score as:
   `0.5 × skill match + 0.5 × semantic similarity`

If the job description contains no recognized skills, skill match is shown as 0%. Semantic similarity can still be calculated.

## Customizing Skills
The skill keyword list is defined in `app.py` as `skill_keywords`. Add relevant tools, skills, and phrases for the job roles you want to assess, such as MIS Executive, Data Analyst, or AI Developer Intern.

Keyword matching only recognizes terms in that list. Add common variants and test with multiple job descriptions before relying on the results.

## Limitations
- The app uses a fixed keyword list; it does not automatically recognize every skill or job domain.
- Scanned/image-only PDFs may require OCR.
- Semantic similarity is not a calibrated probability.
- The combined score uses a simple experimental weighting and has not been validated against hiring outcomes.
- Resumes may contain personal information. Only process resumes you have permission to use, and avoid sharing sensitive information unnecessarily.

## Future Improvements
- Expand the skill dictionary for more roles.
- Add OCR for scanned resumes.
- Add batch resume screening and downloadable reports.
- Evaluate results on a labeled dataset with suitable metrics.
- Improve text extraction and input validation.

## Author
Add your name and GitHub/portfolio links here before publishing.
