import logging
from typing import Optional

import pandas as pd
from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import uvicorn

# ── Logging ──
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(
    title="AI Career Navigator",
    description="Personalized career intelligence API with ML-powered predictions.",
    version="2.0.0",
)

# CORS – allow frontend origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.error(f"Unhandled error: {exc}")
    return JSONResponse(status_code=500, content={"detail": "Internal server error. Please try again."})

# 1. Load data
try:
    df = pd.read_csv('career_data.csv')
    logger.info(f"Loaded career_data.csv with {len(df)} records.")
except FileNotFoundError:
    logger.error("career_data.csv not found – API will not function correctly.")
    df = pd.DataFrame()

# 2. Setup Encoders
le_lang = LabelEncoder().fit(df['Native_Language'])
le_speed = LabelEncoder().fit(df['Learning_Speed'])
le_outcome = LabelEncoder().fit(df['Outcome_Status'])

# 3. Train Model (Using all relevant features)
# We convert text categories to numbers so the AI can process them
df['Lang_Enc'] = le_lang.transform(df['Native_Language'])
df['Speed_Enc'] = le_speed.transform(df['Learning_Speed'])

X = df[['Current_Skill_Level', 'Daily_Study_Hrs', 'Target_Salary', 'Lang_Enc', 'Speed_Enc']]
y = le_outcome.transform(df['Outcome_Status'])

model = RandomForestClassifier(n_estimators=100, random_state=42).fit(X, y)
logger.info("RandomForest model trained successfully.")

STATE_INSIGHTS = {
    "Kerala": {
        "roles": ["Data Analyst", "Fullstack Developer", "Cybersecurity"],
        "sectors": ["IT Services", "FinTech", "EdTech"],
        "outlook": "High",
        "salary_range": "$45k-$95k",
    },
    "Tamil Nadu": {
        "roles": ["Embedded Systems", "QA Engineer", "Cloud Engineer"],
        "sectors": ["Manufacturing Tech", "SaaS", "HealthTech"],
        "outlook": "High",
        "salary_range": "$50k-$100k",
    },
    "Karnataka": {
        "roles": ["AI Engineer", "Product Manager", "Data Scientist"],
        "sectors": ["Product", "AI/ML", "Enterprise"],
        "outlook": "Very High",
        "salary_range": "$70k-$150k",
    },
}

EDU_RANK = {
    "High School": 1,
    "Diploma": 2,
    "Bachelor": 3,
    "Master": 4,
}


def clamp_score(value: float) -> float:
    return round(min(100, max(0, value)), 1)


def get_state_insights(state: str) -> dict:
    return STATE_INSIGHTS.get(
        state,
        {
            "roles": ["Software Developer", "Data Analyst", "UI Designer"],
            "sectors": ["IT Services", "Public Sector", "SME Tech"],
            "outlook": "Medium",
            "salary_range": "$40k-$85k",
        },
    )


def govt_exam_eligibility(age: int, education: str) -> dict:
    edu_level = EDU_RANK.get(education, 1)
    eligible = []
    if 21 <= age <= 32 and edu_level >= 3:
        eligible.append("UPSC CSE")
    if 18 <= age <= 32 and edu_level >= 3:
        eligible.append("SSC CGL")
    if 18 <= age <= 33 and edu_level >= 1:
        eligible.append("Railways")
    if 21 <= age <= 35 and edu_level >= 3:
        eligible.append("State PSC")
    if 20 <= age <= 30 and edu_level >= 3:
        eligible.append("Banking (IBPS)")

    if not eligible:
        summary = "No major exams match current age/education. Consider skill routes or upskilling first."
        next_steps = ["Enroll in a diploma/degree", "Build a study plan", "Re-check after 6 months"]
    else:
        summary = f"Eligible for {len(eligible)} exam(s) based on age and education."
        next_steps = ["Collect syllabus", "Pick one primary exam", "Start mock tests"]

    return {"eligible": eligible, "summary": summary, "next_steps": next_steps}


def resume_readiness(skills: int, hours: int, projects: int, has_portfolio: bool) -> dict:
    score = clamp_score((skills * 0.5) + (projects * 8) + (hours * 3) + (10 if has_portfolio else 0))
    if score < 40:
        level = "Starter"
    elif score < 70:
        level = "Strong"
    else:
        level = "Excellent"

    tips = []
    if projects < 2:
        tips.append("Add 2 small projects to show proof of work")
    if not has_portfolio:
        tips.append("Create a one-page portfolio with project links")
    if hours < 2:
        tips.append("Block 90 minutes daily for focused build time")
    if not tips:
        tips.append("Keep refining impact statements and measurable results")

    return {"score": score, "level": level, "tips": tips}


def internship_readiness(skills: int, hours: int, projects: int) -> dict:
    score = clamp_score((skills * 0.4) + (projects * 10) + (hours * 4))
    if score >= 70:
        status = "Ready"
    elif score >= 45:
        status = "Almost"
    else:
        status = "Needs Foundation"

    tips = []
    if skills < 35:
        tips.append("Strengthen basics with a 2-week core module")
    if projects < 1:
        tips.append("Build a starter project with a public demo")
    if hours < 3:
        tips.append("Increase practice time to 3-4 hours/day")

    return {"score": score, "status": status, "tips": tips}


def micro_task_goals(skills: int, hours: int, speed: str) -> list:
    intensity = "light" if hours <= 2 else "focused" if hours <= 5 else "sprint"
    goals = [
        f"{intensity.title()} learning block (45-90 min)",
        "One practice problem set",
        "One portfolio update or micro-project commit",
    ]
    if speed == "Turbo":
        goals.append("One mock interview question")
    if skills < 30:
        goals.append("Revise one core concept from basics")
    return goals


def freelance_path(skills: int) -> dict:
    if skills < 30:
        starter = ["Resume clean-up", "Landing page edits", "Data entry QA"]
        portfolio = ["One mock client brief", "Before/after redesign"]
    elif skills < 60:
        starter = ["Website setup", "Dashboard cleanup", "Automation scripts"]
        portfolio = ["Case study with measurable impact", "End-to-end workflow"]
    else:
        starter = ["Productized analytics", "Growth experiments", "AI feature prototype"]
        portfolio = ["ROI case study", "Client testimonial video"]

    return {
        "starter_services": starter,
        "portfolio_ideas": portfolio,
        "first_clients": ["Local businesses", "Student startups", "Non-profits"],
    }


def parent_view_summary(feasibility: float, burnout: str) -> dict:
    summary = f"Feasibility is {feasibility}%. Burnout risk is {burnout}."
    support_actions = [
        "Set a weekly check-in routine",
        "Provide a quiet study block",
        "Encourage breaks and sleep discipline",
    ]
    if burnout in {"High", "Critical"}:
        support_actions.append("Reduce hours and add recovery days")
    return {"summary": summary, "support_actions": support_actions}


def counselor_view(skills: int, hours: int) -> dict:
    focus = ["Core fundamentals", "Portfolio depth", "Interview readiness"]
    risk_flags = []
    if skills < 20:
        risk_flags.append("Low baseline skills")
    if hours > 8:
        risk_flags.append("Overload risk")
    if not risk_flags:
        risk_flags.append("No critical risks detected")
    guidance = "Track weekly progress and assign one milestone per week."
    return {"focus_areas": focus, "risk_flags": risk_flags, "guidance": guidance}


def ngo_govt_mode(location_type: str) -> dict:
    if location_type == "Rural":
        mode = "Offline-first"
        resources = ["Printable roadmaps", "Low-bandwidth videos", "Local mentor network"]
    elif location_type == "Semi-Urban":
        mode = "Hybrid delivery"
        resources = ["Weekly lab access", "Community learning groups", "Mobile-friendly UI"]
    else:
        mode = "Online-first"
        resources = ["Live workshops", "Job fair connections", "Fast feedback loops"]
    return {"mode": mode, "resources": resources}


def impact_metrics(skills: int, hours: int) -> dict:
    employability_boost = clamp_score((skills * 0.6) + (hours * 4))
    time_to_job = max(2, round(12 - (skills / 10) - (hours / 2), 1))
    community_impact = min(500, 30 + int(hours * 25))
    return {
        "employability_boost": employability_boost,
        "time_to_job_months": time_to_job,
        "community_impact": community_impact,
    }


def career_diversity(df: pd.DataFrame, skills: int, salary: int) -> list:
    df_copy = df.copy()
    df_copy["skill_diff"] = (df_copy["Current_Skill_Level"] - skills).abs()
    df_copy["salary_diff"] = (df_copy["Target_Salary"] - salary).abs()
    df_copy["score"] = df_copy["skill_diff"] * 1.5 + df_copy["salary_diff"] / 1000
    roles = (
        df_copy.sort_values("score")["Target_Role"]
        .drop_duplicates()
        .head(3)
        .tolist()
    )
    return roles


def career_comparison(df: pd.DataFrame, career_a: str, career_b: str) -> dict:
    role_stats = df.groupby("Target_Role").agg(
        avg_salary=("Target_Salary", "mean"),
        avg_skill=("Current_Skill_Level", "mean"),
    )

    def role_info(name: str) -> dict:
        if name in role_stats.index:
            avg_salary = int(role_stats.loc[name, "avg_salary"])
            avg_skill = int(role_stats.loc[name, "avg_skill"])
            return {
                "name": name,
                "salary": f"${avg_salary}",
                "skills": avg_skill,
                "salary_value": avg_salary,
            }
        return {"name": name, "salary": "Unknown", "skills": 0, "salary_value": 0}

    a_info = role_info(career_a)
    b_info = role_info(career_b)
    growth_lead = "Balanced"
    if a_info["salary_value"] > b_info["salary_value"]:
        growth_lead = "A"
    elif b_info["salary_value"] > a_info["salary_value"]:
        growth_lead = "B"

    comparison = {
        "salary": f"{a_info['salary']} vs {b_info['salary']}",
        "skills_focus": f"{a_info['skills']} vs {b_info['skills']} skill baseline",
        "growth": growth_lead,
        "stability": "Balanced",
    }
    return {"career_a": a_info, "career_b": b_info, "comparison": comparison}

@app.get("/health")
def health_check():
    """Health check endpoint for monitoring."""
    return {"status": "ok", "model_loaded": model is not None, "records": len(df)}


@app.get("/predict")
def predict_outcome(
    skills: int = Query(..., ge=0, le=100, description="Current skill level 0-100"),
    hours: int = Query(..., ge=0, le=24, description="Daily study hours"),
    salary: int = Query(..., ge=0, description="Target salary"),
    lang: str = Query("en", description="Language code"),
    speed: str = Query("Steady", description="Learning speed"),
    state: str = Query("Kerala", description="State name"),
    location_type: str = Query("Urban", description="Urban / Semi-Urban / Rural"),
    age: int = Query(22, ge=15, le=60, description="Age"),
    education: str = Query("Bachelor", description="Education level"),
    projects: int = Query(1, ge=0, le=50, description="Projects completed"),
    portfolio: bool = Query(False, description="Has portfolio"),
    career_a: str = Query("Data Scientist", description="Career option A"),
    career_b: str = Query("UX Designer", description="Career option B"),
):
    logger.info(f"Predict request: skills={skills}, hours={hours}, salary={salary}, lang={lang}")

    try:
        l_enc = le_lang.transform([lang])[0]
        s_enc = le_speed.transform([speed])[0]
    except:
        l_enc, s_enc = 0, 0

    input_df = pd.DataFrame(
        [[skills, hours, salary, l_enc, s_enc]],
        columns=['Current_Skill_Level', 'Daily_Study_Hrs', 'Target_Salary', 'Lang_Enc', 'Speed_Enc']
    )

    probs = model.predict_proba(input_df)[0]
    prediction_enc = model.predict(input_df)[0]
    status = le_outcome.inverse_transform([prediction_enc])[0]

    roi = "High" if (salary / (101 - skills)) > 800 else "Standard"
    dropout_prob = "High" if (hours > 10 or (skills < 15 and hours < 2)) else "Low"

    # --------- LOCATION INTELLIGENCE ---------
    if location_type == "Rural":
        relocation_risk = "High"
        location_score = 60
        location_advice = "Remote or Govt-based careers recommended."
    elif location_type == "Semi-Urban":
        relocation_risk = "Medium"
        location_score = 80
        location_advice = "Hybrid career paths possible."
    else:
        relocation_risk = "Low"
        location_score = 100
        location_advice = "Full industry access available."

    explanations = {
        "en": f"AI predicts '{status}'. Burnout risk is {'Critical' if hours > 8 else 'Optimal'}.",
        "es": f"El IA predice '{status}'. Riesgo de agotamiento: {'Crítico' if hours > 8 else 'Óptimo'}.",
        "hi": f"AI '{status}' का अनुमान लगाता है। {'जोखिम अधिक है' if hours > 8 else 'सामान्य'}।",
        "fr": f"L'IA prévoit '{status}'. Risque de burnout : {'Critique' if hours > 8 else 'Optimal'}."
    }

    state_insights = get_state_insights(state)
    govt_exam = govt_exam_eligibility(age, education)
    resume_meter = resume_readiness(skills, hours, projects, portfolio)
    internship_meter = internship_readiness(skills, hours, projects)
    micro_tasks = micro_task_goals(skills, hours, speed)
    freelance_builder = freelance_path(skills)
    parent_view = parent_view_summary(round((skills * 0.45) + (hours * 5), 1),
                                      "Critical" if hours > 10 else "High" if hours > 7 else "Optimal")
    counselor_view_data = counselor_view(skills, hours)
    ngo_mode = ngo_govt_mode(location_type)
    impact = impact_metrics(skills, hours)
    diversity = career_diversity(df, skills, salary)
    comparison = career_comparison(df, career_a, career_b)

    voice_lang = {
        "en": "en-US",
        "es": "es-ES",
        "fr": "fr-FR",
        "hi": "hi-IN",
        "de": "de-DE",
    }.get(lang, "en-US")

    voice_text = (
        f"Your feasibility is {round((skills * 0.45) + (hours * 5), 1)} percent. "
        f"Focus on {micro_tasks[0].lower()} and {micro_tasks[1].lower()} today."
    )

    return {
        "status": status.replace("_", " "),
        "confidence": round(max(probs) * 100, 1),
        "feasibility": round((skills * 0.45) + (hours * 5), 1),
        "burnout": "Critical" if hours > 10 else "High" if hours > 7 else "Optimal",
        "roi": roi,
        "dropout": dropout_prob,
        "location_score": location_score,
        "relocation_risk": relocation_risk,
        "location_advice": location_advice,
        "adaptation": "Turbo" if speed == "Turbo" else "Deep Focus",
        "explanation": explanations.get(lang, explanations["en"]),
        "state_insights": state_insights,
        "govt_exam": govt_exam,
        "resume_meter": resume_meter,
        "internship_meter": internship_meter,
        "micro_tasks": micro_tasks,
        "freelance_builder": freelance_builder,
        "parent_view": parent_view,
        "counselor_view": counselor_view_data,
        "ngo_mode": ngo_mode,
        "impact": impact,
        "diversity": diversity,
        "comparison": comparison,
        "voice_text": voice_text,
        "voice_lang": voice_lang,
    }


# ── Serve frontend HTML files from the same server ──
from fastapi.responses import FileResponse
import os

FRONTEND_DIR = os.path.dirname(os.path.abspath(__file__))


@app.get("/", response_class=FileResponse)
def serve_home():
    return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))


@app.get("/index.html", response_class=FileResponse)
def serve_home_html():
    return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))


@app.get("/tier1", response_class=FileResponse)
def serve_tier1():
    return FileResponse(os.path.join(FRONTEND_DIR, "tier1.html"))


@app.get("/tier1.html", response_class=FileResponse)
def serve_tier1_html():
    return FileResponse(os.path.join(FRONTEND_DIR, "tier1.html"))


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    logger.info(f"Starting AI Career Navigator on http://localhost:{port}")
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")

