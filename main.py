# FastAPI Reddit Analysis Server - Fixed Version
# Fixes: Proper async resource management, better error handling, graceful shutdown

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.requests import Request
from pydantic import BaseModel
import asyncio
import asyncpraw
import time
import datetime
import json
import openai
from sentence_transformers import SentenceTransformer, util
import nest_asyncio
import uvicorn
import threading
from typing import List, Optional
import os
import signal
import sys
from contextlib import asynccontextmanager
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

nest_asyncio.apply()

# 🔧 GLOBAL CONFIG - Will be updated by API requests
QUESTION = "Why does everybody want to become an influencer?"
KEYWORDS = ["influencer", "content creator", "social media", "fame", "followers",
 "instagram", "tiktok", "youtube", "clout", "influence", "viral",
 "internet fame", "online career", "creator economy", "aspiration",
 "recognition", "popularity", "attention"]

EXPANDED_KEYWORDS = [
  "why people want to be influencers", "dream of being an influencer",
  "influencer lifestyle", "influencer career path", "influencer money",
  "fame on social media", "being famous online", "goal to go viral",
  "social media success stories", "influencer dreams", "making a living from content",
  "aspiring influencer struggles", "life of a content creator", "becoming a YouTuber",
  "wanting followers", "desire to go viral", "attention economy", "social media validation",
  "influencer culture", "clout chasing", "tiktok fame", "internet validation",
  "influencer trend", "everyone's an influencer", "content creation boom",
  "rise of micro-influencers", "creator burnout", "internet celebrity obsession",
  "fake influencer lifestyle", "influencer reality vs expectations",
  "pressures of content creation", "mental health of influencers",
  "why being an influencer is not glamorous", "backlash against influencers"
]

TARGET_SUBREDDITS = [
  "influencersnark", "contentcreators", "youtube", "tiktok",
  "Instagram", "marketing", "AskReddit", "TrueOffMyChest",
  "NoStupidQuestions", "InternetStars", "socialmedia",
  "casualconversation", "anticonsumption", "marketingporn"
]

# Constants
TOP_K_INITIAL_RESULTS = 100
TOP_K_FINAL = 300
DAYS_BACK = 60

# 🔑 API KEYS from environment
openai.api_key = os.getenv("OPENAI_API_KEY")
ASSISTANT_ID = os.getenv("OPENAI_ASSISTANT_ID")

# Global Reddit client - will be initialized in lifespan
reddit = None

SYSTEM_PROMPT = """
## Premium Market Research & Consumer Psychology Report Generator

You are an elite behavioral analyst and market research specialist tasked with producing institutional-grade intelligence reports from real-time social media data. Your analysis will inform strategic decisions for behavioral consultants, market researchers, and business strategists who require actionable insights worth significant investment.

**MISSION:** Transform raw Reddit discussion data into a comprehensive behavioral intelligence report that reveals deep psychological patterns, market opportunities, and strategic recommendations.

---

## REPORT STRUCTURE & REQUIREMENTS

### 1. EXECUTIVE INTELLIGENCE SUMMARY
- **Strategic Overview:** 3-4 paragraph synthesis highlighting the most commercially valuable insights
- **Key Behavioral Indicators:** Primary psychological drivers and emotional catalysts identified
- **Market Opportunity Assessment:** Immediate actionable opportunities with estimated impact potential
- **Risk Factors:** Potential challenges or negative sentiment patterns that could impact strategy

### 2. BEHAVIORAL THEME MATRIX
For each major theme identified:
- **Theme Classification:** Name and behavioral psychology category
- **Prevalence Score:** Estimate percentage of discussions focused on this theme
- **Emotional Intensity Rating:** Scale of 1-10 based on language intensity and engagement
- **Demographic Indicators:** Age, lifestyle, or psychographic patterns evident in the language
- **Commercial Relevance:** How this theme translates to market opportunities or threats
- **Supporting Evidence:** 3-5 verbatim quotes with context and significance analysis

### 3. EMOTIONAL LANDSCAPE & PSYCHOLOGICAL DRIVERS
- **Primary Emotional Clusters:** Dominant emotions with intensity mapping
- **Emotional Journey Mapping:** How emotions evolve through discussion threads
- **Trigger Point Analysis:** Specific words, concepts, or situations that amplify emotional responses
- **Behavioral Prediction Indicators:** What these emotions suggest about future actions/decisions
- **Comparative Emotional Analysis:** How emotions vary by subreddit, thread age, or discussion depth

### 4. NARRATIVE INTELLIGENCE & STORY PATTERNS
- **Dominant Story Archetypes:** Common narrative structures people use to frame their experiences
- **Hero/Villain Identification:** Who or what people position as protagonists/antagonists
- **Success/Failure Frameworks:** How people define and measure outcomes
- **Cultural Reference Points:** Shared experiences, media, or cultural touchstones that influence framing
- **Aspirational vs. Cautionary Tales:** Stories used to inspire vs. warn others

### 5. FRUSTRATION & PAIN POINT ANALYSIS
- **Primary Frustration Categories:** Organized by intensity and frequency
- **Root Cause Analysis:** What underlying needs or expectations are being violated
- **Frustration Evolution:** How complaints escalate or resolve within discussions
- **Solution-Seeking Behaviors:** What people are actively trying to do about their problems
- **Competitive Intelligence:** References to alternatives, competitors, or workarounds

### 6. QUESTIONS & INFORMATION GAPS
- **High-Priority Questions:** Most frequently asked questions with business implications
- **Knowledge Gaps:** What people wish they knew but can't find information about
- **Decision-Making Barriers:** Information needs preventing people from taking action
- **Expert vs. Novice Questioning Patterns:** How information needs vary by experience level
- **Unmet Information Needs:** Opportunities for thought leadership or content marketing

### 7. QUANTITATIVE INTELLIGENCE EXTRACTION
- **User-Reported Statistics:** Numbers, percentages, costs, timeframes mentioned by users
- **Behavioral Frequency Indicators:** How often people engage in relevant behaviors
- **Comparative Benchmarks:** User comparisons to alternatives, competitors, or past experiences
- **Success/Failure Metrics:** How users measure and quantify outcomes
- **Market Size Indicators:** References to market adoption, user base growth, or trend velocity

### 8. STRATEGIC OPPORTUNITY MATRIX
Based on your analysis, identify:
- **Immediate Market Opportunities:** Gaps or needs that could be addressed quickly
- **Product/Service Development Opportunities:** Innovations suggested by user needs
- **Content/Messaging Opportunities:** Communication strategies that would resonate
- **Partnership/Collaboration Opportunities:** Ecosystem connections users are seeking
- **Competitive Positioning Opportunities:** Ways to differentiate based on user priorities

### 9. BEHAVIORAL SEGMENTATION ANALYSIS
- **User Archetype Identification:** Distinct behavioral profiles emerging from the data
- **Decision-Making Style Patterns:** How different segments approach choices and problems
- **Communication Preference Indicators:** Language styles, information density preferences
- **Engagement Pattern Analysis:** How different segments participate in discussions
- **Conversion Readiness Assessment:** Which segments appear closest to taking action

### 10. PREDICTIVE BEHAVIORAL INDICATORS
- **Trend Velocity Signals:** Indicators of accelerating or declining interest
- **Adoption Pattern Predictions:** Early vs. late adopter behavioral markers
- **Resistance Pattern Analysis:** What might prevent broader adoption or acceptance
- **Viral Potential Assessment:** Content or concepts with high shareability indicators
- **Market Timing Intelligence:** Signals about optimal timing for market entry or messaging

---

## ANALYTICAL STANDARDS & METHODOLOGY

### Quality Benchmarks:
- **Depth Over Breadth:** Prioritize deep psychological insights over surface-level observations
- **Commercial Relevance:** Every insight must connect to actionable business intelligence
- **Evidence-Based Analysis:** All conclusions supported by specific textual evidence
- **Behavioral Science Integration:** Apply established psychological and behavioral frameworks
- **Predictive Value:** Focus on insights that enable future-oriented decision making

### Data Processing Guidelines:
- **Signal-to-Noise Optimization:** Prioritize substantive, detailed responses over one-liners
- **Contextual Intelligence:** Consider subreddit culture, thread dynamics, and user interaction patterns
- **Temporal Analysis:** Note how conversations evolve over time within threads
- **Cross-Reference Validation:** Look for patterns that appear across multiple discussions
- **Outlier Significance:** Identify when unusual responses indicate important edge cases

### Reporting Standards:
- **Executive Readiness:** Language and insights appropriate for C-suite consumption
- **Actionability Focus:** Every section must suggest specific next steps or applications
- **Risk Assessment Integration:** Balance opportunities with potential challenges
- **Quantification Emphasis:** Provide estimates, rankings, and comparative assessments wherever possible
- **Competitive Intelligence:** Extract insights about market landscape and competitive positioning

---

## INPUT VARIABLES:
- `question`: The strategic question or topic being analyzed
- `data`: JSON array containing Reddit posts, comments, and metadata

## OUTPUT REQUIREMENTS:
Generate a comprehensive markdown report that demonstrates the analytical depth and strategic value worthy of premium consulting fees. The report should read like it was produced by a top-tier behavioral consulting firm and provide insights that directly inform strategic decision-making.

**Quality Standard:** This report should deliver sufficient strategic value to justify a $500+ price point through depth of analysis, actionable recommendations, and predictive insights that competitors cannot easily replicate.
"""

# Global status tracking
current_status = "Idle"
status_messages = []

def update_status(message):
    global current_status, status_messages
    current_status = message
    status_messages.append(f"{datetime.datetime.now().strftime('%H:%M:%S')}: {message}")
    logger.info(f"📊 {message}")

# Initialize components
cutoff_timestamp = time.time() - DAYS_BACK * 86400
model = SentenceTransformer('all-MiniLM-L6-v2')

# Lifespan management for proper async resource handling
@asynccontextmanager
async def lifespan(app: FastAPI):
    global reddit
    logger.info("🚀 Starting up application...")
    
    # Initialize Reddit client
    try:
        reddit = asyncpraw.Reddit(
            client_id=os.getenv("REDDIT_CLIENT_ID"),
            client_secret=os.getenv("REDDIT_CLIENT_SECRET"),
            user_agent=os.getenv("REDDIT_USER_AGENT", "fama-reddit/0.1 by ProfessionalBison251")
        )
        logger.info("✅ Reddit client initialized")
    except Exception as e:
        logger.error(f"❌ Failed to initialize Reddit client: {e}")
        raise
    
    yield  # Application runs here
    
    # Cleanup on shutdown
    logger.info("🔄 Shutting down application...")
    if reddit:
        try:
            await reddit.close()
            logger.info("✅ Reddit client closed properly")
        except Exception as e:
            logger.error(f"⚠️ Error closing Reddit client: {e}")

# FastAPI app with lifespan management
app = FastAPI(
    title="Fama Reddit Insight Pipeline", 
    version="1.0.0",
    lifespan=lifespan
)

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://fama.patronuslabs.org"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# Request models
class AnalysisRequest(BaseModel):
    question: str
    keywords: List[str]
    expanded_keywords: List[str]
    subreddits: List[str]

class StatusResponse(BaseModel):
    current_status: str
    messages: List[str]

# Core analysis functions with better error handling
async def fetch_initial_posts():
    results = []
    for sub_name in TARGET_SUBREDDITS:
        try:
            subreddit = await reddit.subreddit(sub_name)
            for keyword in EXPANDED_KEYWORDS:
                try:
                    async for submission in subreddit.search(
                        keyword, 
                        sort="relevance", 
                        time_filter="month", 
                        limit=TOP_K_INITIAL_RESULTS // len(TARGET_SUBREDDITS)
                    ):
                        if submission.created_utc < cutoff_timestamp:
                            continue
                        results.append(submission)
                        # Rate limiting
                        await asyncio.sleep(0.1)
                except Exception as e:
                    logger.warning(f"Error searching {sub_name} for '{keyword}': {e}")
                    continue
        except Exception as e:
            logger.warning(f"Error accessing subreddit {sub_name}: {e}")
            continue
    return results

async def extract_post_meta(submission):
    try:
        await submission.load()
        return {
            "id": submission.id,
            "title": submission.title,
            "selftext": submission.selftext,
            "url": f"https://reddit.com{submission.permalink}",
            "score": submission.score,
            "subreddit": str(submission.subreddit),
            "created": datetime.datetime.utcfromtimestamp(submission.created_utc).strftime("%Y-%m-%d")
        }
    except Exception as e:
        logger.warning(f"Error extracting post meta for {submission.id}: {e}")
        return None

async def tier_two_search(subreddit_names):
    final_results = []
    for name in set(subreddit_names):
        try:
            subreddit = await reddit.subreddit(name)
            for keyword in EXPANDED_KEYWORDS:
                try:
                    async for submission in subreddit.search(
                        keyword, 
                        sort="relevance", 
                        time_filter="month", 
                        limit=10
                    ):
                        if submission.created_utc < cutoff_timestamp:
                            continue
                        final_results.append(submission)
                        await asyncio.sleep(0.1)  # Rate limiting
                except Exception as e:
                    logger.warning(f"Error in tier 2 search for {name}/{keyword}: {e}")
                    continue
                await asyncio.sleep(1.0)
        except Exception as e:
            logger.warning(f"Error in tier 2 search for subreddit {name}: {e}")
            continue
    return final_results

async def extract_thread(submission):
    try:
        await submission.load()
        await submission.comments.replace_more(limit=0)
        thread = await extract_post_meta(submission)
        if thread is None:
            return None
            
        thread["comments"] = []

        for comment in submission.comments:
            try:
                if hasattr(comment, "body") and comment.body:
                    replies = []
                    for reply in comment.replies:
                        try:
                            if hasattr(reply, "body") and reply.body:
                                replies.append({
                                    "reply": reply.body.strip(), 
                                    "score": reply.score
                                })
                        except Exception as e:
                            logger.warning(f"Error processing reply: {e}")
                            continue
                    
                    thread["comments"].append({
                        "comment": comment.body.strip(), 
                        "score": comment.score, 
                        "replies": replies
                    })
            except Exception as e:
                logger.warning(f"Error processing comment: {e}")
                continue
        return thread
    except Exception as e:
        logger.warning(f"Error extracting thread {submission.id}: {e}")
        return None

async def upload_to_assistant_and_analyze(file_path):
    try:
        with open(file_path, "rb") as f:
            uploaded_file = openai.files.create(file=f, purpose="assistants")

        file_id = uploaded_file.id
        update_status(f"File uploaded. File ID: {file_id}")

        thread = openai.beta.threads.create()
        openai.beta.threads.messages.create(
            thread_id=thread.id,
            role="user",
            content=QUESTION,
            attachments=[{
                "file_id": file_id,
                "tools": [{"type": "file_search"}]
            }]
        )

        run = openai.beta.threads.runs.create(
            assistant_id=ASSISTANT_ID,
            thread_id=thread.id,
            instructions=SYSTEM_PROMPT
        )

        update_status("Waiting for assistant response...")
        max_wait_time = 300  # 5 minutes max
        start_time = time.time()
        
        while time.time() - start_time < max_wait_time:
            status = openai.beta.threads.runs.retrieve(thread_id=thread.id, run_id=run.id)
            if status.status == "completed":
                break
            elif status.status == "failed":
                raise RuntimeError(f"❌ Assistant run failed: {status.last_error}")
            await asyncio.sleep(2)
        else:
            raise RuntimeError("❌ Assistant run timed out")

        messages = openai.beta.threads.messages.list(thread_id=thread.id)
        result = messages.data[0].content[0].text.value
        update_status("Analysis complete!")
        return {
            "result": result,
            "file_id": file_id,
        }
    except Exception as e:
        logger.error(f"Error in OpenAI analysis: {e}")
        raise

async def run_analysis():
    global QUESTION, KEYWORDS, EXPANDED_KEYWORDS, TARGET_SUBREDDITS

    try:
        update_status("Tier 1 search starting...")
        posts = await fetch_initial_posts()
        update_status(f"Tier 1 complete. Found {len(posts)} posts.")

        if not posts:
            raise RuntimeError("No posts found in tier 1 search")

        subreddits = [p.subreddit.display_name for p in posts if p.subreddit.display_name.lower() != "all"]
        update_status("Tier 2 search starting...")
        tier2_posts = await tier_two_search(subreddits)
        update_status(f"Tier 2 complete. Found {len(tier2_posts)} posts.")

        combined = list({p.id: p for p in posts + tier2_posts}.values())
        update_status(f"Total unique posts: {len(combined)}")

        if not combined:
            raise RuntimeError("No posts found after combining tiers")

        question_embedding = model.encode(QUESTION, convert_to_tensor=True)
        post_texts = []
        for p in combined:
            text = p.title + " " + (p.selftext or "")
            post_texts.append(text)
        
        post_embeddings = model.encode(post_texts, convert_to_tensor=True)
        similarities = util.cos_sim(question_embedding, post_embeddings)[0]

        scored_posts = list(zip(combined, similarities.tolist()))
        top_posts = sorted(scored_posts, key=lambda x: x[1], reverse=True)

        MAX_EXTRACTED = 100
        extracted = []
        update_status("Extracting thread details...")
        
        for i, (submission, sim) in enumerate(top_posts):
            if i >= MAX_EXTRACTED:
                break
            thread = await extract_thread(submission)
            if thread:
                thread["similarity"] = float(sim)  # Ensure JSON serializable
                extracted.append(thread)

        if not extracted:
            raise RuntimeError("No threads extracted successfully")

        output_file = f"fama_reddit_output_{int(time.time())}.json"
        with open(output_file, "w") as f:
            json.dump(extracted, f, indent=2)

        update_status(f"Threads saved to {output_file}")
        result = await upload_to_assistant_and_analyze(output_file)
        
        # Clean up the temporary file
        try:
            os.remove(output_file)
        except Exception as e:
            logger.warning(f"Could not remove temp file {output_file}: {e}")
            
        return result
    except Exception as e:
        logger.error(f"Error in run_analysis: {e}")
        raise

# API Endpoints
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy", 
        "timestamp": datetime.datetime.now().isoformat(),
        "reddit_connected": reddit is not None
    }

@app.get("/status", response_model=StatusResponse)
async def get_status():
    """Get current analysis status and messages"""
    return StatusResponse(
        current_status=current_status, 
        messages=status_messages[-20:]  # Last 20 messages
    )

@app.post("/analyze")
async def analyze_reddit(request: AnalysisRequest):
    """Run Reddit analysis with provided parameters"""
    global QUESTION, KEYWORDS, EXPANDED_KEYWORDS, TARGET_SUBREDDITS, status_messages

    # Validate environment variables
    required_env_vars = ["OPENAI_API_KEY", "OPENAI_ASSISTANT_ID", "REDDIT_CLIENT_ID", "REDDIT_CLIENT_SECRET"]
    missing_vars = [var for var in required_env_vars if not os.getenv(var)]
    if missing_vars:
        raise HTTPException(
            status_code=500, 
            detail=f"Missing required environment variables: {', '.join(missing_vars)}"
        )

    # Clear previous status messages
    status_messages = []

    # Update global variables from request
    QUESTION = request.question
    KEYWORDS = request.keywords
    EXPANDED_KEYWORDS = request.expanded_keywords
    TARGET_SUBREDDITS = request.subreddits

    update_status(f"Starting analysis for question: {QUESTION}")
    update_status(f"Using {len(KEYWORDS)} keywords and {len(EXPANDED_KEYWORDS)} expanded keywords")
    update_status(f"Targeting {len(TARGET_SUBREDDITS)} subreddits")

    try:
        result = await run_analysis()
        return {
            "status": "success",
            "question": QUESTION,
            "analysis_result": result if result else "No result returned.",
            "file_id": result["file_id"] if result else None,
            "timestamp": datetime.datetime.now().isoformat()
        }
    except Exception as e:
        error_msg = f"Analysis failed: {str(e)}"
        update_status(error_msg)
        logger.error(error_msg, exc_info=True)
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": error_msg,
                "question": QUESTION,
                "timestamp": datetime.datetime.now().isoformat()
            }
        )


@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Fama Reddit Insight Pipeline API",
        "endpoints": {
            "health": "/health - Health check",
            "status": "/status - Current analysis status",
            "analyze": "/analyze - Run Reddit analysis (POST)"
        },
        "version": "1.0.0"
    }

# Graceful shutdown handler
def signal_handler(signum, frame):
    logger.info(f"Received signal {signum}, shutting down gracefully...")
    sys.exit(0)

signal.signal(signal.SIGTERM, signal_handler)
signal.signal(signal.SIGINT, signal_handler)

if __name__ == "__main__":
    # For production deployment
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8000, 
        log_level="info",
        access_log=True
    )