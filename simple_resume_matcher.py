#!/usr/bin/env python3
"""
Simple Resume Matcher - A simplified version of the Resume Matcher project
This script demonstrates the core functionality without the complex architecture.

Usage:
    python simple_resume_matcher.py resume.pdf "job description text"
    python simple_resume_matcher.py resume.pdf https://example.com/job-posting
    python simple_resume_matcher.py resume.pdf job_description.pdf
    python simple_resume_matcher.py resume.pdf job_description.pdf \
        --interactive
"""

import sys
import os
import json
import argparse
import re
import time
import threading
from itertools import cycle
from typing import Dict, List, Tuple, Optional
from urllib.parse import urlparse
import requests

try:
    from markitdown import MarkItDown
except ImportError:
    print(
        "❌ Error: markitdown not installed. "
        "Install with: pip install markitdown[all]"
    )
    sys.exit(1)

try:
    from bs4 import BeautifulSoup
except ImportError:
    print(
        "❌ Error: beautifulsoup4 not installed. Install with: pip install beautifulsoup4"
    )
    sys.exit(1)


class ProgressSpinner:
    """Animated spinner for showing progress during long operations"""

    def __init__(self, message="Processing..."):
        self.message = message
        self.spinner = cycle(["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"])
        self.running = False
        self.thread = None

    def start(self):
        """Start the spinner animation"""
        self.running = True
        self.thread = threading.Thread(target=self._spin)
        self.thread.daemon = True  # Allow main thread to exit
        self.thread.start()

    def stop(self):
        """Stop the spinner and clear the line"""
        self.running = False
        if self.thread:
            self.thread.join(timeout=0.1)  # Wait briefly for thread to finish
        print(f"\r{' ' * 60}\r", end="", flush=True)  # Clear line

    def _spin(self):
        """Internal method to run the spinner animation"""
        while self.running:
            print(f"\r{next(self.spinner)} {self.message}", end="", flush=True)
            time.sleep(0.1)


class ProgressTracker:
    """Track and display progress for multi-step operations"""

    def __init__(self, total_steps, operation_name="Operation"):
        self.total_steps = total_steps
        self.current_step = 0
        self.operation_name = operation_name
        self.start_time = time.time()

    def update(self, step_name=None):
        """Update progress to next step"""
        self.current_step += 1
        elapsed = time.time() - self.start_time

        if step_name:
            print(f"✅ Step {self.current_step}/{self.total_steps}: {step_name}")
        else:
            print(f"✅ Step {self.current_step}/{self.total_steps} complete")

        if self.current_step == self.total_steps:
            total_time = time.time() - self.start_time
            print(f"🎉 {self.operation_name} completed in {total_time:.1f} seconds")

    def estimate_time_remaining(self):
        """Estimate time remaining based on current progress"""
        if self.current_step == 0:
            return "Unknown"

        elapsed = time.time() - self.start_time
        avg_time_per_step = elapsed / self.current_step
        remaining_steps = self.total_steps - self.current_step
        estimated_remaining = avg_time_per_step * remaining_steps

        if estimated_remaining < 60:
            return f"{estimated_remaining:.0f} seconds"
        else:
            return f"{estimated_remaining/60:.1f} minutes"


def estimate_processing_time(content_length):
    """Estimate processing time based on content length"""
    base_time = 10  # seconds
    chars_per_second = 1000  # rough estimate
    estimated_time = base_time + (content_length / chars_per_second)

    if estimated_time < 30:
        return f"~{estimated_time:.0f} seconds"
    elif estimated_time < 60:
        return f"~{estimated_time/60:.1f} minutes"
    else:
        return f"~{estimated_time/60:.0f} minutes"


class LLMProvider:
    """Base class for LLM providers"""
    
    def __init__(self, config: dict):
        self.config = config
    
    def generate(self, prompt: str, **kwargs) -> dict:
        """Generate text from prompt, return dict with 'response' key"""
        raise NotImplementedError
    
    def get_available_models(self) -> list:
        """Return list of available model names"""
        raise NotImplementedError
    
    def get_default_model(self) -> str:
        """Return the default model for this provider"""
        raise NotImplementedError


class OllamaProvider(LLMProvider):
    """Provider for Ollama local LLM server"""
    
    def __init__(self, config: dict):
        super().__init__(config)
        self.base_url = config.get('base_url', 'http://localhost:11434')
        self.session = requests.Session()
        self.session.headers.update({
            'Content-Type': 'application/json'
        })
    
    def get_available_models(self) -> list:
        """Get list of available models from Ollama"""
        try:
            response = self.session.get(f"{self.base_url}/api/tags", timeout=30)
            response.raise_for_status()
            models_data = response.json()
            
            # Extract model names from the response
            models = [model["name"] for model in models_data.get("models", [])]
            return models
        except Exception as e:
            print(f"❌ Error getting available models from Ollama: {e}")
            return []
    
    def get_default_model(self) -> str:
        """Get the first available model, or fallback to llama3.1"""
        models = self.get_available_models()
        if models:
            # Prefer common models
            preferred_models = ["llama3.1", "gemma2", "mistral", "llama2"]
            for preferred in preferred_models:
                if preferred in models:
                    return preferred
            return models[0]
        else:
            return "llama3.1"
    
    def generate(self, prompt: str, **kwargs) -> dict:
        """Generate text using Ollama API"""
        model = kwargs.get('model', self.get_default_model())
        url = f"{self.base_url}/api/generate"
        
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": kwargs.get('temperature', 0.1),
                "num_predict": kwargs.get('max_tokens', 2000)
            }
        }
        
        try:
            response = self.session.post(url, json=payload, timeout=300)
            response.raise_for_status()
            result = response.json()
            return {"response": result.get("response", "")}
        except Exception as e:
            print(f"❌ Error calling Ollama: {e}")
            raise


class OpenAIProvider(LLMProvider):
    """Provider for OpenAI API"""
    
    def __init__(self, config: dict):
        super().__init__(config)
        self.api_key = config.get('api_key')
        if not self.api_key:
            self.api_key = os.getenv('OPENAI_API_KEY')
        if not self.api_key:
            raise ValueError("OpenAI API key not provided. Set OPENAI_API_KEY environment variable or pass api_key in config.")
        
        self.base_url = config.get('base_url', 'https://api.openai.com/v1')
        self.session = requests.Session()
        self.session.headers.update({
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json'
        })
    
    def get_available_models(self) -> list:
        """Get list of available models from OpenAI"""
        try:
            response = self.session.get(f"{self.base_url}/models", timeout=30)
            response.raise_for_status()
            models_data = response.json()
            
            # Filter for chat models
            chat_models = [
                model["id"] for model in models_data.get("data", [])
                if "gpt" in model["id"].lower() or "claude" in model["id"].lower()
            ]
            return chat_models
        except Exception as e:
            print(f"❌ Error getting available models from OpenAI: {e}")
            return []
    
    def get_default_model(self) -> str:
        """Return default OpenAI model"""
        return "gpt-4"
    
    def generate(self, prompt: str, **kwargs) -> dict:
        """Generate text using OpenAI API"""
        model = kwargs.get('model', self.get_default_model())
        url = f"{self.base_url}/chat/completions"
        
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": kwargs.get('temperature', 0.1),
            "max_tokens": kwargs.get('max_tokens', 2000)
        }
        
        try:
            response = self.session.post(url, json=payload, timeout=300)
            response.raise_for_status()
            result = response.json()
            return {"response": result["choices"][0]["message"]["content"]}
        except Exception as e:
            print(f"❌ Error calling OpenAI: {e}")
            raise


class GeminiProvider(LLMProvider):
    """Provider for Google Gemini API"""
    
    def __init__(self, config: dict):
        super().__init__(config)
        self.api_key = config.get('api_key')
        if not self.api_key:
            self.api_key = os.getenv('GEMINI_API_KEY')
        if not self.api_key:
            raise ValueError("Gemini API key not provided. Set GEMINI_API_KEY environment variable or pass api_key in config.")
        
        self.base_url = "https://generativelanguage.googleapis.com/v1beta"
        self.session = requests.Session()
        self.session.headers.update({
            'Content-Type': 'application/json'
        })
    
    def get_available_models(self) -> list:
        """Get list of available models from Gemini"""
        # Gemini has a limited set of models
        return ["gemini-pro", "gemini-1.5-flash", "gemini-1.5-pro"]
    
    def get_default_model(self) -> str:
        """Return default Gemini model"""
        return "gemini-pro"
    
    def generate(self, prompt: str, **kwargs) -> dict:
        """Generate text using Gemini API"""
        model = kwargs.get('model', self.get_default_model())
        url = f"{self.base_url}/models/{model}:generateContent?key={self.api_key}"
        
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": kwargs.get('temperature', 0.1),
                "maxOutputTokens": kwargs.get('max_tokens', 2000)
            }
        }
        
        try:
            response = self.session.post(url, json=payload, timeout=300)
            response.raise_for_status()
            result = response.json()
            
            # Extract text from Gemini response
            if 'candidates' in result and len(result['candidates']) > 0:
                content = result['candidates'][0]['content']
                if 'parts' in content and len(content['parts']) > 0:
                    return {"response": content['parts'][0]['text']}
            
            return {"response": ""}
        except Exception as e:
            print(f"❌ Error calling Gemini: {e}")
            raise


class LMStudioProvider(LLMProvider):
    """Provider for LM Studio local server"""
    
    def __init__(self, config: dict):
        super().__init__(config)
        self.base_url = config.get('base_url', 'http://localhost:1234')
        self.session = requests.Session()
    
    def get_available_models(self) -> list:
        """Get list of available models from LM Studio"""
        try:
            response = self.session.get(f"{self.base_url}/v1/models", timeout=30)
            response.raise_for_status()
            models_data = response.json()
            
            # Extract model IDs from the response
            models = [model["id"] for model in models_data.get("data", [])]
            
            # Filter out embedding models
            chat_models = [model for model in models if "embed" not in model.lower()]
            return chat_models
        except Exception as e:
            print(f"❌ Error getting available models from LM Studio: {e}")
            return []
    
    def get_default_model(self) -> str:
        """Get the first available chat model, or fallback to local-model"""
        models = self.get_available_models()
        if models:
            # Prefer Gemma first, then other models
            preferred_models = [
                "google/gemma-3-27b",
                "mistralai/mistral-small-3.2",
                "phi-4-reasoning-plus-mlx",
            ]
            
            for preferred in preferred_models:
                if preferred in models:
                    return preferred
            
            # If no preferred model found, use the first available
            return models[0]
        else:
            return "local-model"
    
    def generate(self, prompt: str, **kwargs) -> dict:
        """Generate text using LM Studio API"""
        model = kwargs.get('model', self.get_default_model())
        url = f"{self.base_url}/v1/chat/completions"
        
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": kwargs.get('temperature', 0.1),
            "max_tokens": kwargs.get('max_tokens', 2000),
            "stream": False,
        }
        
        try:
            response = self.session.post(url, json=payload, timeout=300)
            response.raise_for_status()
            result = response.json()
            return {"response": result["choices"][0]["message"]["content"]}
        except Exception as e:
            print(f"❌ Error calling LM Studio: {e}")
            raise


class LLMClient:
    """Unified client for multiple LLM providers"""
    
    def __init__(self, provider_type: str = "ollama", config: dict = None):
        self.provider_type = provider_type
        self.config = config or {}
        self.provider = self._create_provider(provider_type, self.config)
    
    def _create_provider(self, provider_type: str, config: dict) -> LLMProvider:
        """Create provider instance based on type"""
        if provider_type == "ollama":
            return OllamaProvider(config)
        elif provider_type == "openai":
            return OpenAIProvider(config)
        elif provider_type == "gemini":
            return GeminiProvider(config)
        elif provider_type == "lmstudio":
            return LMStudioProvider(config)
        else:
            raise ValueError(f"Unknown provider: {provider_type}")
    
    def get_available_models(self) -> list:
        """Get available models from current provider"""
        return self.provider.get_available_models()
    
    def get_default_model(self) -> str:
        """Get default model from current provider"""
        return self.provider.get_default_model()
    
    def generate(self, prompt: str, **kwargs) -> dict:
        """Generate text using current provider"""
        return self.provider.generate(prompt, **kwargs)
    
    def validate_connection(self) -> bool:
        """Test provider connectivity and model availability"""
        try:
            models = self.get_available_models()
            default_model = self.get_default_model()
            print(f"✅ {self.provider_type} connected, {len(models)} models available")
            print(f"🤖 Default model: {default_model}")
            return True
        except Exception as e:
            print(f"❌ {self.provider_type} connection failed: {e}")
            return False


class ResumeImprovementAdvisor:
    """Interactive resume improvement advisor using LLM and best practices"""

    def __init__(
        self,
        llm_client: LLMClient,
        model_name: str = None,
        max_tokens: int = 12000,
    ):
        self.llm_client = llm_client
        self.max_tokens = max_tokens
        # Auto-detect model if none specified
        if model_name is None:
            self.model_name = llm_client.get_default_model()
            print(f"🤖 Using auto-detected model: {self.model_name}")
        else:
            self.model_name = model_name

        # Resume improvement best practices from research
        self.improvement_strategies = {
            "quantification": "Add specific metrics, percentages, and numbers to achievements",
            "action_verbs": "Use powerful action verbs to start bullet points",
            "keywords": "Incorporate relevant industry keywords naturally",
            "skills_gap": "Add missing skills that match job requirements",
            "achievements": "Focus on accomplishments rather than just responsibilities",
            "summary": "Create compelling professional summary",
            "formatting": "Improve overall structure and readability",
            "relevance": "Remove irrelevant information and focus on job requirements",
        }

        # Resume template definitions based on expert analysis
        self.resume_templates = {
            "classic_ats": {
                "name": "Classic ATS-Optimized",
                "description": "Universal template optimized for Applicant Tracking Systems",
                "best_for": "All technology professionals, high-volume applications",
                "features": [
                    "Single-column design",
                    "Standard fonts",
                    "Clear headings",
                    "No graphics",
                ],
            },
            "skills_forward": {
                "name": "Skills-Forward Hybrid",
                "description": "Prominent skills section with chronological work history",
                "best_for": "Recent graduates, career changers, diverse backgrounds",
                "features": [
                    "Skills section at top",
                    "Reverse-chronological history",
                    "Transferable skills focus",
                ],
            },
            "accomplishments": {
                "name": "Accomplishments-Centric",
                "description": "Heavy emphasis on measurable impact and achievements",
                "best_for": "Mid-to-senior-level professionals with clear progression",
                "features": [
                    "Quantified achievements",
                    "Strong action verbs",
                    "Career story focus",
                ],
            },
            "dual_column": {
                "name": "Dual-Column Modern",
                "description": "Structured two-column layout for high information density",
                "best_for": "Candidates with many projects, skills, and certifications",
                "features": [
                    "Two-column format",
                    "Visual balance",
                    "High information density",
                ],
            },
            "minimalist": {
                "name": "Minimalist LaTeX/Plain-Text",
                "description": "Clean, technical format favored by developers",
                "best_for": "Software engineers, developers, data scientists",
                "features": [
                    "LaTeX/plain-text",
                    "Technical elegance",
                    "Function over form",
                ],
            },
        }

    def analyze_resume_gaps(
        self,
        resume_text: str,
        job_description: str,
        resume_keywords: List[str],
        job_keywords: List[str],
    ) -> Dict:
        """Analyze gaps between resume and job requirements"""

        print("🔍 Analyzing resume gaps and improvement opportunities...")

        # Estimate processing time
        total_content = len(resume_text) + len(job_description)
        estimated_time = estimate_processing_time(total_content)
        print(f"⏱️  Estimated processing time: {estimated_time}")

        prompt = f"""
        Analyze this resume against the job description and identify specific improvement opportunities.
        
        Job Description:
        {job_description}
        
        Job Keywords: {', '.join(job_keywords)}
        
        Current Resume:
        {resume_text}
        
        Current Resume Keywords: {', '.join(resume_keywords)}
        
        Based on resume improvement best practices, identify the top 3-5 specific areas for improvement:
        1. Quantification gaps (missing metrics/numbers)
        2. Missing skills or keywords
        3. Weak action verbs or descriptions
        4. Missing achievements or accomplishments
        5. Formatting or structure issues
        
        Return a JSON object with:
        - improvement_areas: array of specific improvement categories
        - missing_keywords: array of important job keywords not in resume
        - quantification_opportunities: array of bullet points that need metrics
        - skill_gaps: array of missing skills
        - questions_needed: array of specific questions to ask the user
        
        Only return valid JSON, no other text.
        """

        # Show spinner during LLM call
        spinner = ProgressSpinner("🤖 AI is analyzing your resume...")
        spinner.start()

        try:
            response = self.llm_client.generate(
                model=self.model_name, prompt=prompt, temperature=0.2
            )
            spinner.stop()
            print("✅ Gap analysis complete!")

            # Extract JSON from response
            json_start = response["response"].find("{")
            json_end = response["response"].rfind("}") + 1
            json_str = response["response"][json_start:json_end]

            return json.loads(json_str)
        except Exception as e:
            spinner.stop()
            print(f"❌ Error analyzing gaps: {e}")
            print("🔄 Trying fallback to smaller model...")

            # Try with a smaller model as fallback
            try:
                fallback_model = "mistralai/mistral-small-3.2"
                if fallback_model != self.model_name:
                    spinner = ProgressSpinner("🤖 Retrying with fallback model...")
                    spinner.start()

                    response = self.llm_client.generate(
                        model=fallback_model, prompt=prompt, temperature=0.2
                    )
                    spinner.stop()
                    print(f"✅ Successfully used fallback model: {fallback_model}")

                    # Extract JSON from response
                    json_start = response["response"].find("{")
                    json_end = response["response"].rfind("}") + 1
                    json_str = response["response"][json_start:json_end]

                    return json.loads(json_str)
                else:
                    print("❌ Fallback model is same as current model")
                    return {"improvement_areas": [], "questions_needed": []}
            except Exception as fallback_error:
                spinner.stop()
                print(f"❌ Fallback model also failed: {fallback_error}")
                return {"improvement_areas": [], "questions_needed": []}

    def generate_targeted_questions(
        self,
        gap_analysis: Dict,
        resume_text: str,
        job_description: str,
        round_num: int = 1,
        previous_responses: Dict = None,
    ) -> List[Dict]:
        """Generate targeted questions based on gap analysis and round number"""

        if round_num == 1:
            return self._generate_round1_questions(gap_analysis)
        else:
            return self._generate_dynamic_questions(
                gap_analysis,
                round_num,
                previous_responses,
                resume_text,
                job_description,
            )

    def _generate_round1_questions(self, gap_analysis: Dict) -> List[Dict]:
        """Generate initial foundation questions"""
        questions = []

        # Generate questions for missing keywords
        if gap_analysis.get("missing_keywords"):
            keywords = gap_analysis["missing_keywords"][:3]  # Limit to top 3
            questions.append(
                {
                    "type": "keywords",
                    "question": f"Do you have experience with any of these technologies/skills: {', '.join(keywords)}? If yes, please describe your experience level and any projects where you used them.",
                    "context": "These keywords are important for the job but missing from your resume",
                }
            )

        # Generate questions for quantification
        if gap_analysis.get("quantification_opportunities"):
            questions.append(
                {
                    "type": "quantification",
                    "question": "Can you provide specific metrics for your achievements? For example: 'Increased sales by X%', 'Reduced costs by $X', 'Managed team of X people', 'Completed X projects'",
                    "context": "Adding specific numbers makes your achievements more impactful",
                }
            )

        # Generate questions for skills
        if gap_analysis.get("skill_gaps"):
            skills = gap_analysis["skill_gaps"][:3]  # Limit to top 3
            questions.append(
                {
                    "type": "skills",
                    "question": f"Can you add any experience with: {', '.join(skills)}? Include any certifications, courses, or practical experience.",
                    "context": "These skills are highly valued for this position",
                }
            )

        # Generate questions for achievements
        questions.append(
            {
                "type": "achievements",
                "question": "What are your top 2-3 professional achievements that demonstrate your impact? Focus on results and outcomes rather than just responsibilities.",
                "context": "Highlighting achievements helps you stand out from other candidates",
            }
        )

        # Generate questions for summary
        questions.append(
            {
                "type": "summary",
                "question": "What makes you uniquely qualified for this specific role? Consider your unique combination of skills, experience, and achievements.",
                "context": "A compelling summary can significantly improve your resume's impact",
            }
        )

        return questions

    def _generate_dynamic_questions(
        self,
        gap_analysis: Dict,
        round_num: int,
        previous_responses: Dict,
        resume_text: str,
        job_description: str,
    ) -> List[Dict]:
        """Generate dynamic questions using LLM based on previous responses"""

        print(f"🤖 Generating personalized questions for Round {round_num}...")

        # Prepare context for LLM
        context = f"""
        Round {round_num} of resume improvement. Previous responses from user:
        {self._format_previous_responses(previous_responses)}
        
        Current gap analysis:
        - Missing keywords: {gap_analysis.get('missing_keywords', [])}
        - Skill gaps: {gap_analysis.get('skill_gaps', [])}
        - Quantification opportunities: {gap_analysis.get('quantification_opportunities', False)}
        - Job context: {job_description[:500]}...
        """

        # Generate questions based on round number
        if round_num == 2:
            prompt = f"""
            {context}
            
            Generate 3-5 specific, targeted questions for Round 2 that focus on:
            1. Deepening the user's responses with more specific details
            2. Exploring industry-specific experience relevant to the job
            3. Identifying transferable skills that could apply to this role
            4. Getting more quantified achievements and metrics
            5. Understanding their leadership and team management style
            
            Focus on questions that build upon their previous responses and dig deeper into relevant areas.
            Make questions specific and actionable.
            
            Return only the questions, one per line, no numbering.
            """
        else:  # Round 3+
            prompt = f"""
            {context}
            
            Generate 3-5 advanced questions for Round {round_num} that focus on:
            1. Unique differentiators and competitive advantages
            2. Specific examples of innovation or problem-solving
            3. Growth mindset and learning ability
            4. Cultural fit and values alignment
            5. Future goals and career trajectory alignment with this role
            
            These should be sophisticated questions that help the user stand out from other candidates.
            Focus on what makes them uniquely qualified beyond just skills and experience.
            
            Return only the questions, one per line, no numbering.
            """

        # Show spinner during LLM call
        spinner = ProgressSpinner("🤖 AI is generating personalized questions...")
        spinner.start()

        try:
            response = self.llm_client.generate(prompt, model=self.model_name)
            spinner.stop()
            print("✅ Dynamic questions generated!")
            questions_text = response.get("response", "").strip()

            # Parse questions from response
            question_lines = [
                q.strip() for q in questions_text.split("\n") if q.strip()
            ]

            # Convert to question format
            questions = []
            for i, question in enumerate(question_lines[:5]):  # Limit to 5 questions
                questions.append(
                    {
                        "type": f"dynamic_round{round_num}",
                        "question": question,
                        "context": f"Round {round_num} targeted question to improve your resume",
                    }
                )

            # Fallback if LLM fails or returns too few questions
            if len(questions) < 3:
                questions.extend(
                    self._generate_fallback_questions(round_num, gap_analysis)
                )

            return questions[:5]  # Ensure max 5 questions

        except Exception as e:
            print(f"❌ Error generating dynamic questions: {e}")
            return self._generate_fallback_questions(round_num, gap_analysis)

    def _format_previous_responses(self, previous_responses: Dict) -> str:
        """Format previous responses for LLM context"""
        if not previous_responses:
            return "No previous responses available."

        formatted = []
        for question_num, response in previous_responses.items():
            # Truncate long responses for context
            truncated_response = (
                response[:500] + "..." if len(response) > 500 else response
            )
            formatted.append(f"Q{question_num}: {truncated_response}")

        return "\n".join(formatted)

    def _generate_fallback_questions(
        self, round_num: int, gap_analysis: Dict
    ) -> List[Dict]:
        """Generate fallback questions if LLM fails"""
        if round_num == 2:
            return [
                {
                    "type": "leadership",
                    "question": "Can you provide more specific examples of your technical leadership experience?",
                    "context": "Understanding your leadership style and approach",
                },
                {
                    "type": "metrics",
                    "question": "What metrics or KPIs did you track in your previous management roles?",
                    "context": "Quantifying your management impact",
                },
                {
                    "type": "mentoring",
                    "question": "How do you approach mentoring and developing junior engineers?",
                    "context": "Demonstrating your people development skills",
                },
                {
                    "type": "problem_solving",
                    "question": "Can you describe a challenging technical problem you solved as a team lead?",
                    "context": "Showcasing your technical problem-solving abilities",
                },
                {
                    "type": "trends",
                    "question": "What industry trends or technologies are you most excited about?",
                    "context": "Showing your forward-thinking mindset",
                },
            ]
        else:
            return [
                {
                    "type": "differentiation",
                    "question": "What unique perspective or approach do you bring to technical leadership?",
                    "context": "Identifying your competitive advantages",
                },
                {
                    "type": "learning",
                    "question": "How do you stay current with emerging technologies and industry best practices?",
                    "context": "Demonstrating continuous learning",
                },
                {
                    "type": "influence",
                    "question": "Can you describe a time when you had to influence stakeholders without direct authority?",
                    "context": "Showing your influence and communication skills",
                },
                {
                    "type": "philosophy",
                    "question": "What's your philosophy on balancing technical excellence with business needs?",
                    "context": "Understanding your strategic thinking",
                },
                {
                    "type": "success",
                    "question": "How do you measure success for your teams and projects?",
                    "context": "Defining your success metrics and values",
                },
            ]

    def generate_resume_template(
        self, resume_data: Dict, template_type: str = "classic_ats", matcher=None
    ) -> str:
        """Generate a formatted resume using the specified template"""

        if template_type not in self.resume_templates:
            print(f"❌ Unknown template type: {template_type}")
            print(f"Available templates: {', '.join(self.resume_templates.keys())}")
            template_type = "classic_ats"  # Default fallback

        template_info = self.resume_templates[template_type]
        print(f"📄 Generating {template_info['name']} template...")

        # Extract basic information from resume data
        improved_resume = resume_data.get("improved_resume", "")
        improved_keywords = resume_data.get("improved_keywords", [])
        original_keywords = resume_data.get("original_keywords", [])

        # Extract LinkedIn URL and certifications from resume text
        original_resume = resume_data.get("original_resume", "")
        linkedin_url = self._extract_linkedin_url(original_resume)
        certifications = self._extract_certifications(original_resume)

        # Create a simple, reliable data structure
        template_data = {
            "name": resume_data.get("structured_resume", {}).get(
                "name", "Cesar Delgado"
            ),
            "contact": {
                "email": resume_data.get("structured_resume", {}).get(
                    "email", "beettlle@gmail.com"
                ),
                "phone": resume_data.get("structured_resume", {}).get(
                    "phone", "(402) 617-6049"
                ),
                "location": "Damascus, OR",  # Not in structured data, keeping default
                "linkedin": linkedin_url,  # Extracted from resume text
            },
            "summary": (
                improved_resume[:500] + "..."
                if len(improved_resume) > 500
                else improved_resume
            ),
            "skills": improved_keywords,
            "experience": self._extract_experience_from_structured_data(resume_data),
            "education": {
                "degree": "Bachelor of Science in Biochemistry",
                "institution": "University of Nebraska, Lincoln",
                "dates": "2004",
            },
            "certifications": certifications,  # Extracted from resume text
        }

        # Generate template based on type
        if template_type == "classic_ats":
            return self._generate_classic_ats_template(template_data)
        elif template_type == "skills_forward":
            return self._generate_skills_forward_template(template_data)
        elif template_type == "accomplishments":
            return self._generate_accomplishments_template(template_data)
        elif template_type == "dual_column":
            return self._generate_dual_column_template(template_data)
        elif template_type == "minimalist":
            return self._generate_minimalist_template(template_data)
        else:
            return self._generate_classic_ats_template(
                template_data
            )  # Default fallback

    def _generate_classic_ats_template(self, data: Dict) -> str:
        """Generate Classic ATS-Optimized template"""
        template = f"""
{data.get('name', 'YOUR NAME').upper()}
{data.get('contact', {}).get('email', 'email@example.com')} | {data.get('contact', {}).get('phone', '(555) 123-4567')}
{data.get('contact', {}).get('location', 'City, State')} | {data.get('contact', {}).get('linkedin', 'linkedin.com/in/yourprofile')}

PROFESSIONAL SUMMARY
{data.get('summary', 'Experienced technology professional with proven track record of delivering results.')}

TECHNICAL SKILLS
{', '.join(data.get('skills', []))}

PROFESSIONAL EXPERIENCE

"""

        for job in data.get("experience", []):
            achievements = job.get("achievements", [])

            if achievements:
                # Full format for jobs with achievements
                template += f"""
{job.get('title', 'Job Title')} | {job.get('company', 'Company Name')}
{job.get('dates', 'Dates')} | {job.get('location', 'Location')}

"""
                for achievement in achievements:
                    template += f"• {achievement}\n"
                template += "\n"
            else:
                # Compact format for jobs without achievements
                template += f"{job.get('title', 'Job Title')} | {job.get('company', 'Company Name')} | {job.get('dates', 'Dates')}\n\n"

        template += f"""
EDUCATION
{data.get('education', {}).get('degree', 'Degree')} | {data.get('education', {}).get('institution', 'Institution')}
{data.get('education', {}).get('dates', 'Dates')}

CERTIFICATIONS
"""
        for cert in data.get('certifications', []):
            template += f"• {cert}\n"

        return template.strip()

    def _generate_skills_forward_template(self, data: Dict) -> str:
        """Generate Skills-Forward Hybrid template"""
        template = f"""
{data.get('name', 'YOUR NAME').upper()}
{data.get('contact', {}).get('email', 'email@example.com')} | {data.get('contact', {}).get('phone', '(555) 123-4567')}
{data.get('contact', {}).get('location', 'City, State')} | {data.get('contact', {}).get('linkedin', 'linkedin.com/in/yourprofile')}

CORE COMPETENCIES
{', '.join(data.get('skills', [])[:10])}

TECHNICAL EXPERTISE
{', '.join(data.get('skills', [])[10:])}

PROFESSIONAL SUMMARY
{data.get('summary', 'Experienced technology professional with proven track record of delivering results.')}

PROFESSIONAL EXPERIENCE

"""

        for job in data.get("experience", []):
            achievements = job.get("achievements", [])

            if achievements:
                # Full format for jobs with achievements
                template += f"""
{job.get('title', 'Job Title')} | {job.get('company', 'Company Name')}
{job.get('dates', 'Dates')} | {job.get('location', 'Location')}

"""
                for achievement in achievements:
                    template += f"• {achievement}\n"
                template += "\n"
            else:
                # Compact format for jobs without achievements
                template += f"{job.get('title', 'Job Title')} | {job.get('company', 'Company Name')} | {job.get('dates', 'Dates')}\n\n"

        template += f"""
EDUCATION & CERTIFICATIONS
{data.get('education', {}).get('degree', 'Degree')} | {data.get('education', {}).get('institution', 'Institution')}
{data.get('education', {}).get('dates', 'Dates')}

Certifications:
"""
        for cert in data.get('certifications', []):
            template += f"• {cert}\n"

        return template.strip()

    def _generate_accomplishments_template(self, data: Dict) -> str:
        """Generate Accomplishments-Centric template"""
        template = f"""
{data.get('name', 'YOUR NAME').upper()}
{data.get('contact', {}).get('email', 'email@example.com')} | {data.get('contact', {}).get('phone', '(555) 123-4567')}
{data.get('contact', {}).get('location', 'City, State')} | {data.get('contact', {}).get('linkedin', 'linkedin.com/in/yourprofile')}

EXECUTIVE SUMMARY
{data.get('summary', 'Experienced technology professional with proven track record of delivering results.')}

KEY ACHIEVEMENTS
"""

        # Extract quantified achievements
        achievements = []
        for job in data.get("experience", []):
            for achievement in job.get("achievements", []):
                if any(char.isdigit() for char in achievement):  # Contains numbers
                    achievements.append(achievement)

        for achievement in achievements[:5]:  # Top 5 quantified achievements
            template += f"• {achievement}\n"

        template += f"""

TECHNICAL SKILLS
{', '.join(data.get('skills', []))}

PROFESSIONAL EXPERIENCE

"""

        for job in data.get("experience", []):
            achievements = job.get("achievements", [])

            if achievements:
                # Full format for jobs with achievements
                template += f"""
{job.get('title', 'Job Title')} | {job.get('company', 'Company Name')}
{job.get('dates', 'Dates')} | {job.get('location', 'Location')}

"""
                for achievement in achievements:
                    template += f"• {achievement}\n"
                template += "\n"
            else:
                # Compact format for jobs without achievements
                template += f"{job.get('title', 'Job Title')} | {job.get('company', 'Company Name')} | {job.get('dates', 'Dates')}\n\n"

        template += f"""
EDUCATION
{data.get('education', {}).get('degree', 'Degree')} | {data.get('education', {}).get('institution', 'Institution')}
{data.get('education', {}).get('dates', 'Dates')}

CERTIFICATIONS
"""
        for cert in data.get('certifications', []):
            template += f"• {cert}\n"

        return template.strip()

    def _generate_dual_column_template(self, data: Dict) -> str:
        """Generate Dual-Column Modern template"""
        template = f"""
{data.get('name', 'YOUR NAME').upper()}
{data.get('contact', {}).get('email', 'email@example.com')} | {data.get('contact', {}).get('phone', '(555) 555-5555')}
{data.get('contact', {}).get('location', 'City, State')} | {data.get('contact', {}).get('linkedin', 'linkedin.com/in/yourprofile')}

PROFESSIONAL SUMMARY
{data.get('summary', 'Experienced technology professional with proven track record of delivering results.')}

TECHNICAL SKILLS
{', '.join(data.get('skills', []))}

EDUCATION
{data.get('education', {}).get('degree', 'Degree')} | {data.get('education', {}).get('institution', 'Institution')}
{data.get('education', {}).get('dates', 'Dates')}

CERTIFICATIONS
{', '.join(data.get('certifications', []))}

PROFESSIONAL EXPERIENCE

"""

        for job in data.get("experience", []):
            template += f"""
{job.get('title', 'Job Title')} | {job.get('company', 'Company Name')}
{job.get('dates', 'Dates')} | {job.get('location', 'Location')}

"""
            for achievement in job.get("achievements", []):
                template += f"• {achievement}\n"
            template += "\n"

        return template.strip()

    def _generate_minimalist_template(self, data: Dict) -> str:
        """Generate Minimalist LaTeX/Plain-Text template"""
        template = f"""
{data.get('name', 'YOUR NAME')}
{data.get('contact', {}).get('email', 'email@example.com')} | {data.get('contact', {}).get('phone', '(555) 123-4567')}
{data.get('contact', {}).get('location', 'City, State')} | {data.get('contact', {}).get('linkedin', 'linkedin.com/in/yourprofile')}

{data.get('summary', 'Experienced technology professional with proven track record of delivering results.')}

SKILLS
{', '.join(data.get('skills', []))}

EXPERIENCE

"""

        for job in data.get("experience", []):
            achievements = job.get("achievements", [])

            if achievements:
                # Full format for jobs with achievements
                template += f"""
{job.get('title', 'Job Title')} at {job.get('company', 'Company Name')}
{job.get('dates', 'Dates')}

"""
                for achievement in achievements:
                    template += f"  {achievement}\n"
                template += "\n"
            else:
                # Compact format for jobs without achievements
                template += f"{job.get('title', 'Job Title')} at {job.get('company', 'Company Name')} | {job.get('dates', 'Dates')}\n\n"

        template += f"""
EDUCATION
{data.get('education', {}).get('degree', 'Degree')} from {data.get('education', {}).get('institution', 'Institution')}
{data.get('education', {}).get('dates', 'Dates')}

CERTIFICATIONS
"""
        for cert in data.get('certifications', []):
            template += f"• {cert}\n"

        return template.strip()

    def _extract_linkedin_url(self, resume_text: str) -> str:
        """Extract LinkedIn URL from resume text"""
        import re

        # Common LinkedIn URL patterns
        linkedin_patterns = [
            r"linkedin\.com/in/[a-zA-Z0-9\-_]+",  # Standard LinkedIn profile URLs
            r"linkedin\.com/company/[a-zA-Z0-9\-_]+",  # Company pages
            r"www\.linkedin\.com/in/[a-zA-Z0-9\-_]+",  # With www
            r"https?://linkedin\.com/in/[a-zA-Z0-9\-_]+",  # With protocol
            r"https?://www\.linkedin\.com/in/[a-zA-Z0-9\-_]+",  # Full URL
        ]

        for pattern in linkedin_patterns:
            match = re.search(pattern, resume_text, re.IGNORECASE)
            if match:
                linkedin_url = match.group()
                # Ensure it has protocol
                if not linkedin_url.startswith("http"):
                    linkedin_url = "https://" + linkedin_url
                return linkedin_url

        return "linkedin.com/in/yourprofile"  # Default placeholder

    def _extract_certifications(self, resume_text: str) -> List[str]:
        """Extract certifications from resume text"""
        import re
        
        certifications = []
        
        # Look for certification section with various formats
        cert_patterns = [
            r"C\s*E\s*R\s*T\s*I\s*F\s*I\s*C\s*A\s*T\s*I\s*O\s*N\s*S?\s*\n(.*?)(?=\n\n|\n[A-Z]|$)",  # C E R T I F I C A T I O N S
            r"CERTIFICATIONS?\s*\n(.*?)(?=\n\n|\n[A-Z]|$)",  # CERTIFICATIONS
            r"Certifications?\s*\n(.*?)(?=\n\n|\n[A-Z]|$)",  # Certifications
        ]
        
        for pattern in cert_patterns:
            match = re.search(pattern, resume_text, re.IGNORECASE | re.DOTALL)
            if match:
                cert_section = match.group(1)
                # Split by bullet points or newlines
                lines = cert_section.split('\n')
                for line in lines:
                    line = line.strip()
                    # Remove bullet points and clean up
                    if line.startswith('•') or line.startswith('-') or line.startswith('*'):
                        line = line.lstrip('•-* ').strip()
                    if line and len(line) > 3:  # Minimum length to be a real certification
                        certifications.append(line)
                break
        
        # If no structured section found, look for certification keywords
        if not certifications:
            cert_keywords = [
                r'certified\s+[a-zA-Z\s]+',
                r'certification\s+[a-zA-Z\s]+',
                r'[A-Z]+\s+certified',
                r'[A-Z]+\s+certification',
            ]
            
            for keyword in cert_keywords:
                matches = re.findall(keyword, resume_text, re.IGNORECASE)
                for match in matches:
                    if len(match.strip()) > 5:  # Minimum length
                        certifications.append(match.strip())
        
        return certifications

    def _extract_experience_from_structured_data(self, resume_data: Dict) -> List[Dict]:
        """Extract experience information from structured resume data"""
        experience = []

        # Check if we have structured resume data
        if (
            "structured_resume" in resume_data
            and "experience" in resume_data["structured_resume"]
        ):
            structured_experience = resume_data["structured_resume"]["experience"]

            for job in structured_experience:
                # Extract achievements from description
                achievements = []
                if "description" in job:
                    # Split description by newlines and look for bullet points
                    lines = job["description"].split("\n")
                    for line in lines:
                        line = line.strip()
                        if (
                            line.startswith("•")
                            or line.startswith("-")
                            or line.startswith("*")
                        ):
                            achievement = line.lstrip("•-* ").strip()
                            if achievement:
                                achievements.append(achievement)
                        elif (
                            line
                            and not line.startswith("•")
                            and not line.startswith("-")
                            and not line.startswith("*")
                        ):
                            # If no bullet points, treat each line as an achievement
                            achievements.append(line)

                experience.append(
                    {
                        "title": job.get("title", "Job Title"),
                        "company": job.get("company", "Company Name"),
                        "dates": job.get("duration", "Dates"),
                        "location": "Location",  # Not available in structured data
                        "achievements": achievements,
                    }
                )
        else:
            # Fallback to text parsing if no structured data
            improved_resume = resume_data.get("improved_resume", "")
            experience = self._extract_experience_from_text(improved_resume)

        return experience

    def _is_recent_job(self, duration: str) -> bool:
        """Check if a job is recent (within last 10 years)"""
        if not duration:
            return False

        # Look for year patterns like "2024", "2023", etc.
        import re

        year_match = re.search(r"20[12]\d", duration)
        if year_match:
            year = int(year_match.group())
            return year >= 2014  # Within last 10 years

        # Look for "Present" or "Current" which indicates recent job
        if any(word in duration.lower() for word in ["present", "current", "now"]):
            return True

        return False

    def _extract_experience_from_text(self, resume_text: str) -> List[Dict]:
        """Extract experience information from resume text using simple parsing"""
        experience = []

        # Look for common job title patterns
        lines = resume_text.split("\n")
        current_job = None

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Look for job titles (usually in bold or all caps)
            if any(
                keyword in line.upper()
                for keyword in ["ENGINEER", "MANAGER", "ARCHITECT", "LEAD", "DIRECTOR"]
            ):
                if current_job:
                    experience.append(current_job)

                # Extract job info
                parts = line.split(";")
                if len(parts) >= 2:
                    title_company = parts[0].strip()
                    location_dates = parts[1].strip()

                    current_job = {
                        "title": (
                            title_company.split(";")[0]
                            if ";" in title_company
                            else title_company
                        ),
                        "company": "Company Name",
                        "dates": location_dates,
                        "location": "Location",
                        "achievements": [],
                    }

            # Look for bullet points (achievements)
            elif line.startswith("•") or line.startswith("-") or line.startswith("*"):
                if current_job:
                    achievement = line.lstrip("•-* ").strip()
                    if achievement:
                        current_job["achievements"].append(achievement)

        # Add the last job
        if current_job:
            experience.append(current_job)

        return experience

    def list_available_templates(self) -> None:
        """Display available resume templates"""
        print("\n📋 AVAILABLE RESUME TEMPLATES")
        print("=" * 50)

        for key, template in self.resume_templates.items():
            print(f"\n🎯 {template['name']}")
            print(f"   Description: {template['description']}")
            print(f"   Best For: {template['best_for']}")
            print(f"   Key Features: {', '.join(template['features'])}")
            print(f"   Template ID: {key}")

    def integrate_user_responses(
        self,
        resume_text: str,
        user_responses: Dict,
        job_description: str,
        job_keywords: List[str],
    ) -> str:
        """Intelligently integrate user responses into the resume"""

        print("🔄 Integrating your responses into resume improvements...")

        # Check if we need chunk processing based on estimated tokens
        total_content_length = (
            len(job_description) + len(resume_text) + len(str(user_responses))
        )

        # Estimate tokens (roughly 4 characters per token)
        estimated_tokens = total_content_length // 4

        print(f"📊 Content length analysis: {total_content_length} characters")
        print(f"📊 Estimated tokens: {estimated_tokens}")

        # Use configured token limit
        token_limit = self.max_tokens

        # If content fits within token limit, use full content
        if estimated_tokens < token_limit:
            print(
                f"✅ Content fits within token limit ({estimated_tokens} < {token_limit}), using full content"
            )
            processed_job_desc = job_description
            processed_resume = resume_text
            processed_responses = "\n\n".join(
                [f"{k}: {v}" for k, v in user_responses.items()]
            )
        else:
            print(
                f"⚠️ Content exceeds token limit ({estimated_tokens} > {token_limit}), using chunk processing..."
            )
            # Use chunk processing to preserve all information
            processed_job_desc = self.process_job_description_chunks(
                job_description, job_keywords
            )
            processed_resume = self.process_resume_chunks(resume_text)
            processed_responses = self.process_user_responses_chunks(user_responses)

        prompt = f"""
        You are an expert resume writer. Integrate the user's responses into their resume to improve it for the target job.
        
        Job Description:
        {processed_job_desc}
        
        Job Keywords: {', '.join(job_keywords)}
        
        Current Resume:
        {processed_resume}
        
        User Responses:
        {processed_responses}
        
        Instructions:
        1. Naturally incorporate the user's responses into the appropriate sections
        2. Add missing keywords and skills where relevant
        3. Quantify achievements with the provided metrics
        4. Improve the professional summary with unique qualifications
        5. Maintain professional formatting and avoid redundancy
        6. Keep the same overall structure but enhance content
        7. Focus on achievements and results, not just responsibilities
        8. Use action verbs and industry-specific terminology
        
        Return only the improved resume text, no explanations.
        """

        # Show spinner during LLM call
        spinner = ProgressSpinner("🤖 AI is integrating your responses...")
        spinner.start()

        try:
            response = self.llm_client.generate(
                model=self.model_name, prompt=prompt, temperature=0.3
            )
            spinner.stop()
            print("✅ Resume integration complete!")
            return response["response"].strip()
        except Exception as e:
            spinner.stop()
            print(f"❌ Error integrating responses: {e}")
            print("🔄 Trying fallback to smaller model...")

            # Try with a smaller model as fallback
            try:
                fallback_model = "mistralai/mistral-small-3.2"
                if fallback_model != self.model_name:
                    spinner = ProgressSpinner("🤖 Retrying with fallback model...")
                    spinner.start()

                    response = self.llm_client.generate(
                        model=fallback_model, prompt=prompt, temperature=0.3
                    )
                    spinner.stop()
                    print(f"✅ Successfully used fallback model: {fallback_model}")
                    return response["response"].strip()
                else:
                    print("❌ Fallback model is same as current model")
                    return resume_text
            except Exception as fallback_error:
                spinner.stop()
                print(f"❌ Fallback model also failed: {fallback_error}")
                return resume_text

    def conduct_interactive_improvement(
        self,
        resume_text: str,
        job_description: str,
        resume_keywords: List[str],
        job_keywords: List[str],
        max_rounds: int = 3,
    ) -> Tuple[str, List[float]]:
        """Conduct interactive improvement sessions with the user"""

        print("\n" + "=" * 60)
        print("🎯 INTERACTIVE RESUME IMPROVEMENT SESSION")
        print("=" * 60)

        current_resume = resume_text
        scores = []
        user_responses = {}

        for round_num in range(1, max_rounds + 1):
            print(f"\n📋 ROUND {round_num} OF {max_rounds}")
            print("-" * 40)

            # Analyze current gaps
            gap_analysis = self.analyze_resume_gaps(
                current_resume, job_description, resume_keywords, job_keywords
            )

            # Generate questions based on round number and previous responses
            questions = self.generate_targeted_questions(
                gap_analysis, current_resume, job_description, round_num, user_responses
            )

            if not questions:
                print("✅ No more improvement opportunities identified!")
                break

            # Ask questions and collect responses
            round_responses = self.ask_questions(questions, round_num)
            user_responses.update(round_responses)

            # Integrate responses
            print("\n🔄 Integrating your responses...")
            improved_resume = self.integrate_user_responses(
                current_resume, user_responses, job_description, job_keywords
            )

            # Calculate new score
            new_keywords = self.extract_keywords(improved_resume, "improved resume")
            new_score = self.calculate_similarity(new_keywords, job_keywords)
            scores.append(new_score)

            print(f"📊 Round {round_num} Score: {new_score:.2%}")

            # Ask if user wants to continue
            if round_num < max_rounds:
                continue_choice = (
                    input(f"\n🤔 Continue to round {round_num + 1}? (y/n): ")
                    .lower()
                    .strip()
                )
                if continue_choice not in ["y", "yes"]:
                    print("✅ Stopping improvement process.")
                    break

            current_resume = improved_resume

        return current_resume, scores

    def ask_questions(self, questions: List[Dict], round_num: int) -> Dict:
        """Ask questions to the user and collect responses"""

        responses = {}

        print(f"\n❓ I have {len(questions)} questions to help improve your resume:")

        for i, q in enumerate(questions, 1):
            print(f"\n{i}. {q['context']}")
            print(f"   {q['question']}")

            response = input(f"\n   Your response: ").strip()

            if response.lower() in ["skip", "none", "no", "n/a"]:
                print("   ⏭️  Skipped")
                continue
            elif response:
                responses[q["type"]] = response
                print("   ✅ Recorded")
            else:
                print("   ⏭️  No response provided")

        return responses

    def extract_keywords(self, text: str, context: str = "resume") -> List[str]:
        """Extract keywords from text using LLM"""

        prompt = f"""
        Extract the most important keywords and skills from this {context} text.
        Focus on technical skills, tools, technologies, and key competencies.

        {text}

        Return only a comma-separated list of keywords, no other text.
        """

        try:
            response = self.llm_client.generate(
                model=self.model_name, prompt=prompt, temperature=0.1
            )

            keywords = [kw.strip() for kw in response["response"].split(",")]
            return [kw for kw in keywords if kw and len(kw) > 2]
        except Exception as e:
            print(f"❌ Error extracting keywords: {e}")
            return []

    def calculate_similarity(
        self, resume_keywords: List[str], job_keywords: List[str]
    ) -> float:
        """Calculate similarity between resume and job keywords"""

        if not resume_keywords or not job_keywords:
            return 0.0

        # Simple Jaccard similarity
        resume_set = set(resume_keywords)
        job_set = set(job_keywords)

        intersection = len(resume_set.intersection(job_set))
        union = len(resume_set.union(job_set))

        return intersection / union if union > 0 else 0.0


class JobDescriptionExtractor:
    """Extract job descriptions from various sources (text, URL, PDF)"""

    def __init__(self):
        self.md = MarkItDown(enable_plugins=False)
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
            }
        )

    def is_url(self, text: str) -> bool:
        """Check if the input is a URL"""
        try:
            result = urlparse(text)
            return all([result.scheme, result.netloc])
        except:
            return False

    def is_pdf_file(self, text: str) -> bool:
        """Check if the input is a PDF file path"""
        return text.lower().endswith(".pdf") and os.path.exists(text)

    def extract_from_url(self, url: str) -> str:
        """Extract job description from a web page"""
        print(f"🌐 Extracting job description from URL: {url}")

        try:
            response = self.session.get(url, timeout=30)
            response.raise_for_status()

            soup = BeautifulSoup(response.content, "html.parser")

            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()

            # Common selectors for job descriptions
            selectors = [
                '[class*="job-description"]',
                '[class*="job-details"]',
                '[class*="description"]',
                '[id*="job-description"]',
                '[id*="job-details"]',
                '[id*="description"]',
                "article",
                ".content",
                "#content",
                "main",
                ".main-content",
            ]

            # Try to find job description content
            content = None
            for selector in selectors:
                elements = soup.select(selector)
                if elements:
                    # Get the largest text block
                    largest_element = max(elements, key=lambda x: len(x.get_text()))
                    if len(largest_element.get_text().strip()) > 100:
                        content = largest_element
                        break

            if not content:
                # Fallback to body content
                content = soup.find("body") or soup

            # Extract text and clean it up
            text = content.get_text()
            text = self.clean_text(text)

            if len(text.strip()) < 50:
                raise ValueError(
                    "Could not extract meaningful content from the webpage"
                )

            print(f"✅ Extracted {len(text)} characters from webpage")
            return text

        except Exception as e:
            print(f"❌ Error extracting from URL: {e}")
            raise

    def extract_from_pdf(self, pdf_path: str) -> str:
        """Extract job description from a PDF file"""
        print(f"📄 Extracting job description from PDF: {pdf_path}")

        try:
            result = self.md.convert(pdf_path)
            text = result.text_content
            text = self.clean_text(text)

            if len(text.strip()) < 50:
                raise ValueError("Could not extract meaningful content from the PDF")

            print(f"✅ Extracted {len(text)} characters from PDF")
            return text

        except Exception as e:
            print(f"❌ Error extracting from PDF: {e}")
            raise

    def clean_text(self, text: str) -> str:
        """Clean and normalize extracted text"""
        # Remove extra whitespace
        text = re.sub(r"\s+", " ", text)

        # Remove common web artifacts
        text = re.sub(
            r"Cookie|Privacy Policy|Terms of Service|©.*?\.",
            "",
            text,
            flags=re.IGNORECASE,
        )

        # Remove navigation elements
        text = re.sub(
            r"Home|About|Contact|Careers|Jobs|Login|Sign Up",
            "",
            text,
            flags=re.IGNORECASE,
        )

        # Clean up line breaks
        text = re.sub(r"\n\s*\n", "\n\n", text)

        return text.strip()

    def extract_job_description(self, source: str) -> str:
        """Extract job description from various sources"""
        if self.is_url(source):
            return self.extract_from_url(source)
        elif self.is_pdf_file(source):
            return self.extract_from_pdf(source)
        else:
            # Assume it's plain text
            return source.strip()


class SimpleResumeMatcher:
    def __init__(
        self,
        provider_type: str = "ollama",
        model_name: str = None,
        config: dict = None,
    ):
        """
        Initialize the resume matcher with any supported LLM provider.

        Args:
            provider_type: Type of LLM provider ("ollama", "openai", "gemini", "lmstudio")
            model_name: Name of the model to use (auto-detected if None)
            config: Provider-specific configuration
        """
        self.provider_type = provider_type
        self.model_name = model_name
        self.md = MarkItDown(enable_plugins=False)
        self.job_extractor = JobDescriptionExtractor()
        
        # Create LLM client with provider
        self.llm_client = LLMClient(provider_type, config or {})
        self.improvement_advisor = ResumeImprovementAdvisor(
            self.llm_client, self.model_name
        )

        # Validate provider connection
        if not self.llm_client.validate_connection():
            self._show_provider_setup_instructions(provider_type)
            sys.exit(1)
    
    def _show_provider_setup_instructions(self, provider_type: str):
        """Show setup instructions for the specified provider"""
        print(f"❌ Error: {provider_type} not running or not accessible")
        print("💡 To fix this:")
        
        if provider_type == "ollama":
            print("   1. Install Ollama from https://ollama.ai/")
            print("   2. Start Ollama: ollama serve")
            print("   3. Pull a model: ollama pull llama3.1")
            print("   4. Run this script again")
        elif provider_type == "openai":
            print("   1. Get an API key from https://platform.openai.com/")
            print("   2. Set environment variable: export OPENAI_API_KEY='your-key'")
            print("   3. Or pass --api-key your-key")
            print("   4. Run this script again")
        elif provider_type == "gemini":
            print("   1. Get an API key from https://makersuite.google.com/app/apikey")
            print("   2. Set environment variable: export GEMINI_API_KEY='your-key'")
            print("   3. Or pass --api-key your-key")
            print("   4. Run this script again")
        elif provider_type == "lmstudio":
            print("   1. Download and install LM Studio from https://lmstudio.ai/")
            print("   2. Open LM Studio and start the local server")
            print("   3. Make sure it's running on http://localhost:1234")
            print("   4. Download a model (like 'google/gemma-3-27b')")
            print("   5. Run this script again")
        
        print(f"   💡 Or try a different provider with: --provider ollama")

    def extract_text_from_file(self, file_path: str) -> str:
        """
        Extract text from PDF or DOCX file.

        Args:
            file_path: Path to the resume file

        Returns:
            Extracted text content
        """
        print(f"📄 Extracting text from {file_path}...")

        try:
            result = self.md.convert(file_path)
            return result.text_content
        except Exception as e:
            print(f"❌ Error extracting text: {e}")
            sys.exit(1)

    def extract_resume_structure(self, resume_text: str) -> Dict:
        """
        Extract structured data from resume text using LLM.

        Args:
            resume_text: Raw resume text

        Returns:
            Structured resume data
        """
        print("🤖 Extracting structured resume data...")

        prompt = f"""
        Extract structured information from this resume and return as JSON:

        {resume_text}

        Return a JSON object with these fields:
        - name: string
        - email: string
        - phone: string (optional)
        - summary: string
        - experience: array of objects with title, company, duration, description
        - education: array of objects with degree, institution, year
        - skills: array of strings

        Only return valid JSON, no other text.
        """

        try:
            response = self.llm_client.generate(
                model=self.model_name, prompt=prompt, temperature=0.1
            )

            # Extract JSON from response
            json_start = response["response"].find("{")
            json_end = response["response"].rfind("}") + 1
            json_str = response["response"][json_start:json_end]

            result = json.loads(json_str)
            return result
        except Exception as e:
            print(f"❌ Error extracting structure: {e}")
            return {"error": "Failed to extract structured data"}

    def extract_keywords(self, text: str, context: str = "resume") -> List[str]:
        """
        Extract keywords from text using LLM.

        Args:
            text: Text to extract keywords from
            context: Context (resume or job)

        Returns:
            List of keywords
        """
        print(f"🔑 Extracting keywords from {context}...")

        prompt = f"""
        Extract the most important keywords and skills from this {context} text.
        Focus on technical skills, tools, technologies, and key competencies.

        {text}

        Return only a comma-separated list of keywords, no other text.
        """

        try:
            response = self.llm_client.generate(
                model=self.model_name, prompt=prompt, temperature=0.1
            )

            keywords = [kw.strip() for kw in response["response"].split(",")]
            return [kw for kw in keywords if kw and len(kw) > 2]
        except Exception as e:
            print(f"❌ Error extracting keywords: {e}")
            return []

    def calculate_similarity(
        self, resume_keywords: List[str], job_keywords: List[str]
    ) -> float:
        """
        Calculate similarity between resume and job keywords.

        Args:
            resume_keywords: Keywords from resume
            job_keywords: Keywords from job description

        Returns:
            Similarity score (0-1)
        """
        if not resume_keywords or not job_keywords:
            return 0.0

        # Simple Jaccard similarity
        resume_set = set(resume_keywords)
        job_set = set(job_keywords)

        intersection = len(resume_set.intersection(job_set))
        union = len(resume_set.union(job_set))

        return intersection / union if union > 0 else 0.0

    def improve_resume(
        self,
        resume_text: str,
        job_description: str,
        resume_keywords: List[str],
        job_keywords: List[str],
    ) -> Tuple[str, float]:
        """
        Improve resume to better match job description.

        Args:
            resume_text: Original resume text
            job_description: Job description text
            resume_keywords: Current resume keywords
            job_keywords: Job keywords

        Returns:
            Tuple of (improved_resume_text, new_similarity_score)
        """
        print("🚀 Improving resume to match job requirements...")

        prompt = f"""
        You are an expert resume writer. Improve this resume to better match the job description.

        Job Description:
        {job_description}

        Job Keywords: {', '.join(job_keywords)}

        Current Resume:
        {resume_text}

        Current Resume Keywords: {', '.join(resume_keywords)}

        Instructions:
        1. Naturally incorporate relevant job keywords into the resume
        2. Rewrite sections to better align with job requirements
        3. Add relevant skills and experiences if implied
        4. Maintain professional tone and avoid keyword stuffing
        5. Keep the same overall structure and length
        6. Focus on quantifiable achievements where possible

        Return only the improved resume text, no explanations.
        """

        try:
            response = self.llm_client.generate(
                model=self.model_name, prompt=prompt, temperature=0.3
            )

            improved_resume = response["response"].strip()

            # Calculate new similarity score
            new_keywords = self.extract_keywords(improved_resume, "improved resume")
            new_score = self.calculate_similarity(new_keywords, job_keywords)

            return improved_resume, new_score
        except Exception as e:
            print(f"❌ Error improving resume: {e}")
            return resume_text, 0.0

    def format_resume_preview(self, structured_data: Dict) -> str:
        """
        Format structured resume data for display.

        Args:
            structured_data: Structured resume data

        Returns:
            Formatted resume text
        """
        if "error" in structured_data:
            return "Failed to extract structured data"

        preview = []

        # Header
        if structured_data.get("name"):
            preview.append(f"# {structured_data['name']}")

        if structured_data.get("email"):
            preview.append(f"📧 {structured_data['email']}")

        if structured_data.get("phone"):
            preview.append(f"📞 {structured_data['phone']}")

        preview.append("")  # Empty line

        # Summary
        if structured_data.get("summary"):
            preview.append("## Summary")
            preview.append(structured_data["summary"])
            preview.append("")

        # Experience
        if structured_data.get("experience"):
            preview.append("## Experience")
            for exp in structured_data["experience"]:
                preview.append(f"### {exp.get('title', 'N/A')}")
                preview.append(
                    f"**{exp.get('company', 'N/A')}** | "
                    f"{exp.get('duration', 'N/A')}"
                )
                if exp.get("description"):
                    preview.append(exp["description"])
                preview.append("")

        # Education
        if structured_data.get("education"):
            preview.append("## Education")
            for edu in structured_data["education"]:
                preview.append(f"### {edu.get('degree', 'N/A')}")
                preview.append(
                    f"**{edu.get('institution', 'N/A')}** | "
                    f"{edu.get('year', 'N/A')}"
                )
                preview.append("")

        # Skills
        if structured_data.get("skills"):
            preview.append("## Skills")
            preview.append(", ".join(structured_data["skills"]))

        return "\n".join(preview)

    def chunk_content(
        self, text: str, max_chunk_size: int = 3000, overlap: int = 200
    ) -> List[str]:
        """Split text into overlapping chunks for processing"""
        if len(text) <= max_chunk_size:
            return [text]

        chunks = []
        start = 0

        while start < len(text):
            end = start + max_chunk_size

            # If this isn't the last chunk, try to break at a sentence boundary
            if end < len(text):
                # Look for sentence endings in the overlap area
                overlap_start = max(start, end - overlap)
                for i in range(end, overlap_start, -1):
                    if text[i - 1] in ".!?\n":
                        end = i
                        break

            chunks.append(text[start:end])
            start = end - overlap if end < len(text) else end

        return chunks

    def process_job_description_chunks(
        self, job_description: str, job_keywords: List[str]
    ) -> str:
        """Process job description in chunks and extract key information"""
        print("📋 Processing job description in chunks...")

        chunks = self.chunk_content(job_description, max_chunk_size=3000, overlap=200)

        if len(chunks) == 1:
            return job_description

        # Process each chunk to extract key information
        key_info_parts = []

        for i, chunk in enumerate(chunks):
            prompt = f"""
            Extract the most important information from this job description chunk.
            Focus on: requirements, responsibilities, qualifications, and key details.
            
            Chunk {i+1} of {len(chunks)}:
            {chunk}
            
            Return only the key information in a concise format, no explanations.
            """

            try:
                response = self.llm_client.generate(
                    model=self.model_name, prompt=prompt, temperature=0.1
                )
                key_info_parts.append(response["response"].strip())
            except Exception as e:
                print(f"❌ Error processing chunk {i+1}: {e}")
                # Fallback: use the chunk as-is
                key_info_parts.append(
                    chunk[:1000] + "..." if len(chunk) > 1000 else chunk
                )

        # Combine the key information
        combined_info = "\n\n".join(key_info_parts)

        # Create a final summary
        summary_prompt = f"""
        Create a comprehensive but concise summary of this job description information.
        Include all requirements, responsibilities, and key qualifications.
        
        Information:
        {combined_info}
        
        Job Keywords: {', '.join(job_keywords)}
        
        Return a well-structured job description summary.
        """

        try:
            response = self.llm_client.generate(
                model=self.model_name, prompt=summary_prompt, temperature=0.1
            )
            return response["response"].strip()
        except Exception as e:
            print(f"❌ Error creating summary: {e}")
            return combined_info

    def process_resume_chunks(self, resume_text: str) -> str:
        """Process resume in chunks and extract key information"""
        print("📄 Processing resume in chunks...")

        chunks = self.chunk_content(resume_text, max_chunk_size=3000, overlap=200)

        if len(chunks) == 1:
            return resume_text

        # Process each chunk to extract key information
        key_info_parts = []

        for i, chunk in enumerate(chunks):
            prompt = f"""
            Extract the most important information from this resume chunk.
            Focus on: experience, achievements, skills, and qualifications.
            
            Chunk {i+1} of {len(chunks)}:
            {chunk}
            
            Return only the key information in a concise format, no explanations.
            """

            try:
                response = self.llm_client.generate(
                    model=self.model_name, prompt=prompt, temperature=0.1
                )
                key_info_parts.append(response["response"].strip())
            except Exception as e:
                print(f"❌ Error processing chunk {i+1}: {e}")
                # Fallback: use the chunk as-is
                key_info_parts.append(
                    chunk[:1000] + "..." if len(chunk) > 1000 else chunk
                )

        # Combine the key information
        combined_info = "\n\n".join(key_info_parts)

        # Create a final summary
        summary_prompt = f"""
        Create a comprehensive but concise summary of this resume information.
        Maintain the structure: header, summary, experience, education, skills.
        
        Information:
        {combined_info}
        
        Return a well-structured resume summary.
        """

        try:
            response = self.llm_client.generate(
                model=self.model_name, prompt=summary_prompt, temperature=0.1
            )
            return response["response"].strip()
        except Exception as e:
            print(f"❌ Error creating summary: {e}")
            return combined_info

    def process_user_responses_chunks(self, user_responses: Dict) -> str:
        """Process user responses in chunks and create a summary"""
        print("💬 Processing user responses in chunks...")

        # Combine all responses into one text
        combined_responses = "\n\n".join(
            [f"{k}: {v}" for k, v in user_responses.items()]
        )

        if len(combined_responses) <= 2000:
            return combined_responses

        chunks = self.chunk_content(
            combined_responses, max_chunk_size=2000, overlap=100
        )

        # Process each chunk to extract key information
        key_info_parts = []

        for i, chunk in enumerate(chunks):
            prompt = f"""
            Extract the most important information from this user response chunk.
            Focus on: skills, achievements, metrics, and key qualifications.
            
            Chunk {i+1} of {len(chunks)}:
            {chunk}
            
            Return only the key information in a concise format, no explanations.
            """

            try:
                response = self.llm_client.generate(
                    model=self.model_name, prompt=prompt, temperature=0.1
                )
                key_info_parts.append(response["response"].strip())
            except Exception as e:
                print(f"❌ Error processing chunk {i+1}: {e}")
                # Fallback: use the chunk as-is
                key_info_parts.append(
                    chunk[:800] + "..." if len(chunk) > 800 else chunk
                )

        # Combine the key information
        combined_info = "\n\n".join(key_info_parts)

        # Create a final summary
        summary_prompt = f"""
        Create a comprehensive summary of the user's responses.
        Organize by: skills, achievements, metrics, and qualifications.
        
        Information:
        {combined_info}
        
        Return a well-structured summary of user responses.
        """

        try:
            response = self.llm_client.generate(
                model=self.model_name, prompt=summary_prompt, temperature=0.1
            )
            return response["response"].strip()
        except Exception as e:
            print(f"❌ Error creating summary: {e}")
            return combined_info

    def run_analysis(
        self, resume_file: str, job_source: str, previous_results: Dict = None
    ) -> Dict:
        """
        Run complete resume analysis and improvement.

        Args:
            resume_file: Path to resume file
            job_source: Job description (text, URL, or PDF file)

        Returns:
            Analysis results
        """
        print("🎯 Starting Resume Matcher Analysis...")
        print("=" * 50)

        # Step 1: Extract text from resume (or use previous results)
        if previous_results and "original_resume" in previous_results:
            resume_text = previous_results["original_resume"]
            print(
                f"📂 Using resume text from previous analysis ({len(resume_text)} characters)"
            )
        else:
            resume_text = self.extract_text_from_file(resume_file)
            print(f"✅ Extracted {len(resume_text)} characters from resume")

        # Step 2: Extract job description from various sources (or use previous results)
        if previous_results and "job_description" in previous_results:
            job_description = previous_results["job_description"]
            print(
                f"📂 Using job description from previous analysis ({len(job_description)} characters)"
            )
        else:
            job_description = self.job_extractor.extract_job_description(job_source)
            print(f"✅ Extracted {len(job_description)} characters from job source")

        # Step 3: Extract structured data
        structured_data = self.extract_resume_structure(resume_text)

        # Step 4: Extract keywords (or use previous results)
        if previous_results and "resume_keywords" in previous_results:
            resume_keywords = previous_results["resume_keywords"]
            print(
                f"📂 Using resume keywords from previous analysis ({len(resume_keywords)} keywords)"
            )
        else:
            resume_keywords = self.extract_keywords(resume_text, "resume")
            print(f"✅ Extracted {len(resume_keywords)} resume keywords")

        if previous_results and "job_keywords" in previous_results:
            job_keywords = previous_results["job_keywords"]
            print(
                f"📂 Using job keywords from previous analysis ({len(job_keywords)} keywords)"
            )
        else:
            job_keywords = self.extract_keywords(job_description, "job")
            print(f"✅ Extracted {len(job_keywords)} job keywords")

        # Step 5: Calculate initial similarity
        initial_score = self.calculate_similarity(resume_keywords, job_keywords)
        print(f"📊 Initial match score: {initial_score:.2%}")

        # Step 6: Improve resume
        improved_resume, new_score = self.improve_resume(
            resume_text, job_description, resume_keywords, job_keywords
        )

        print(f"📈 Improved match score: {new_score:.2%}")
        print(f"📈 Score improvement: {new_score - initial_score:.2%}")

        # Step 7: Format results
        improved_keywords = self.extract_keywords(improved_resume, "improved resume")

        results = {
            "original_score": initial_score,
            "improved_score": new_score,
            "score_improvement": new_score - initial_score,
            "resume_keywords": resume_keywords,
            "job_keywords": job_keywords,
            "improved_keywords": improved_keywords,
            "structured_resume": structured_data,
            "original_resume": resume_text,
            "improved_resume": improved_resume,
            "job_description": job_description,
            "formatted_preview": self.format_resume_preview(structured_data),
        }

        return results

    def conduct_interactive_improvement(
        self, resume_file: str, job_source: str, previous_results: Dict = None
    ) -> Dict:
        """Conduct an interactive improvement session for a specific resume and job."""
        print("\n" + "=" * 60)
        print("🎯 INTERACTIVE RESUME IMPROVEMENT SESSION")
        print("=" * 60)

        # Use previous results if available, otherwise extract fresh data
        if previous_results and "original_resume" in previous_results:
            resume_text = previous_results["original_resume"]
            print(
                f"📂 Using resume text from previous analysis ({len(resume_text)} characters)"
            )
        else:
            resume_text = self.extract_text_from_file(resume_file)
            print(f"✅ Extracted {len(resume_text)} characters from resume")

        if previous_results and "job_description" in previous_results:
            job_description = previous_results["job_description"]
            print(
                f"📂 Using job description from previous analysis ({len(job_description)} characters)"
            )
        else:
            job_description = self.job_extractor.extract_job_description(job_source)
            print(f"✅ Extracted {len(job_description)} characters from job source")

        if previous_results and "resume_keywords" in previous_results:
            resume_keywords = previous_results["resume_keywords"]
            print(
                f"📂 Using resume keywords from previous analysis ({len(resume_keywords)} keywords)"
            )
        else:
            resume_keywords = self.extract_keywords(resume_text, "resume")
            print(f"✅ Extracted {len(resume_keywords)} resume keywords")

        if previous_results and "job_keywords" in previous_results:
            job_keywords = previous_results["job_keywords"]
            print(
                f"📂 Using job keywords from previous analysis ({len(job_keywords)} keywords)"
            )
        else:
            job_keywords = self.extract_keywords(job_description, "job")
            print(f"✅ Extracted {len(job_keywords)} job keywords")

        improved_resume, scores = (
            self.improvement_advisor.conduct_interactive_improvement(
                resume_text, job_description, resume_keywords, job_keywords
            )
        )

        print(f"\n📈 Final Improved Match Score: {scores[-1]:.2%}")
        print(f"📈 Total Score Improvement: {scores[-1] - scores[0]:.2%}")

        final_keywords = self.extract_keywords(improved_resume, "improved resume")

        results = {
            "original_keywords": resume_keywords,
            "job_keywords": job_keywords,
            "improved_keywords": final_keywords,
            "structured_resume": self.extract_resume_structure(improved_resume),
            "original_resume": resume_text,
            "improved_resume": improved_resume,
            "job_description": job_description,
            "formatted_preview": self.format_resume_preview(
                self.extract_resume_structure(improved_resume)
            ),
        }

        return results


def main():
    parser = argparse.ArgumentParser(description="Simple Resume Matcher")
    parser.add_argument(
        "resume_file", nargs="?", help="Path to resume file (PDF or DOCX)"
    )
    parser.add_argument(
        "job_source",
        nargs="?",
        help="Job description source: text, URL, or PDF file path",
    )
    parser.add_argument(
        "--provider",
        choices=["ollama", "openai", "gemini", "lmstudio"],
        default="ollama",
        help="LLM provider to use (default: ollama)",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Model name (auto-detected if not specified)",
    )
    parser.add_argument(
        "--api-key",
        help="API key for cloud providers (OpenAI, Gemini)",
    )
    parser.add_argument(
        "--provider-url",
        help="Custom URL for provider (e.g., Ollama server URL)",
    )
    parser.add_argument("--output", help="Output file for results (JSON)")
    parser.add_argument(
        "--input", help="Input file with previous analysis results (JSON)"
    )
    parser.add_argument("--preview", action="store_true", help="Show formatted preview")
    parser.add_argument(
        "--interactive", action="store_true", help="Run interactive improvement session"
    )
    parser.add_argument(
        "--template",
        choices=[
            "classic_ats",
            "skills_forward",
            "accomplishments",
            "dual_column",
            "minimalist",
        ],
        default="classic_ats",
        help="Resume template to use for output (default: classic_ats)",
    )
    parser.add_argument(
        "--list-templates", action="store_true", help="List available resume templates"
    )
    parser.add_argument(
        "--list-providers", action="store_true", help="List available LLM providers and setup instructions"
    )
    parser.add_argument(
        "--template-output", help="Output file for formatted resume template"
    )
    parser.add_argument(
        "--generate-template",
        choices=[
            "classic_ats",
            "skills_forward",
            "accomplishments",
            "dual_column",
            "minimalist",
        ],
        help="Generate specific template from input file (requires --input)",
    )

    args = parser.parse_args()

    # Show provider information if requested
    if args.list_providers:
        print("🤖 AVAILABLE LLM PROVIDERS")
        print("=" * 50)
        print("\n1. Ollama (Local - Recommended for privacy)")
        print("   - Free, runs locally on your machine")
        print("   - Setup: Install from https://ollama.ai/")
        print("   - Usage: ollama serve && ollama pull llama3.1")
        print("   - Default URL: http://localhost:11434")
        
        print("\n2. LM Studio (Local)")
        print("   - Free, runs locally on your machine")
        print("   - Setup: Install from https://lmstudio.ai/")
        print("   - Usage: Start local server in LM Studio app")
        print("   - Default URL: http://localhost:1234")
        
        print("\n3. OpenAI (Cloud)")
        print("   - Paid service, requires API key")
        print("   - Setup: Get key from https://platform.openai.com/")
        print("   - Usage: export OPENAI_API_KEY='your-key'")
        print("   - Models: gpt-4, gpt-3.5-turbo")
        
        print("\n4. Gemini (Cloud)")
        print("   - Free tier available, requires API key")
        print("   - Setup: Get key from https://makersuite.google.com/app/apikey")
        print("   - Usage: export GEMINI_API_KEY='your-key'")
        print("   - Models: gemini-pro, gemini-1.5-flash")
        
        print("\n💡 For privacy and cost, we recommend starting with Ollama!")
        sys.exit(0)

    # Build provider configuration
    config = {}
    if args.api_key:
        config['api_key'] = args.api_key
    if args.provider_url:
        config['base_url'] = args.provider_url

    # List templates if requested (before initializing matcher)
    if args.list_templates:
        # Initialize matcher only for template listing
        matcher = SimpleResumeMatcher(args.provider, args.model, config)
        matcher.improvement_advisor.list_available_templates()
        sys.exit(0)

    # Handle template generation from input file
    if args.generate_template:
        if not args.input:
            print("❌ Error: --generate-template requires --input file")
            print("Use --help for more information")
            sys.exit(1)

        # Load the input file
        if not os.path.exists(args.input):
            print(f"❌ Error: Input file '{args.input}' not found")
            sys.exit(1)

        try:
            with open(args.input, "r") as f:
                input_data = json.load(f)
            print(f"📂 Loaded analysis data from {args.input}")

            # Validate required data
            required_keys = ["improved_resume"]
            missing_keys = [key for key in required_keys if key not in input_data]
            if missing_keys:
                print(f"❌ Error: Input file missing required keys: {missing_keys}")
                sys.exit(1)

            # Initialize matcher for template generation
            matcher = SimpleResumeMatcher(args.provider, args.model, config)

            # Generate the template
            print(f"📄 Generating {args.generate_template} template...")
            formatted_resume = matcher.improvement_advisor.generate_resume_template(
                input_data, args.generate_template, matcher
            )

            # Save to file if specified, otherwise show preview
            if args.template_output:
                with open(args.template_output, "w") as f:
                    f.write(formatted_resume)
                print(f"💾 Template saved to {args.template_output}")
            else:
                print(f"\n📄 {args.generate_template.upper()} TEMPLATE:")
                print("=" * 50)
                print(formatted_resume)

            sys.exit(0)

        except json.JSONDecodeError as e:
            print(f"❌ Error: Invalid JSON in input file: {e}")
            sys.exit(1)
        except Exception as e:
            print(f"❌ Error generating template: {e}")
            sys.exit(1)

    # Check if resume file and job source are provided (unless using input file)
    if not args.input and (not args.resume_file or not args.job_source):
        print("❌ Error: Both resume_file and job_source are required")
        print("Use --input to load previous analysis results")
        print("Use --list-templates to see available resume templates")
        print("Use --list-providers to see available LLM providers")
        print("Use --help for more information")
        sys.exit(1)

    # Initialize matcher for actual processing
    matcher = SimpleResumeMatcher(args.provider, args.model, config)

    # Load previous results if input file specified
    previous_results = None
    if args.input:
        if not os.path.exists(args.input):
            print(f"❌ Error: Input file '{args.input}' not found")
            sys.exit(1)

        try:
            with open(args.input, "r") as f:
                previous_results = json.load(f)
            print(f"📂 Loaded previous analysis from {args.input}")

            # Validate the loaded data structure
            required_keys = ["original_resume", "job_description", "job_keywords"]
            missing_keys = [key for key in required_keys if key not in previous_results]
            if missing_keys:
                print(f"❌ Error: Input file missing required keys: {missing_keys}")
                sys.exit(1)

        except json.JSONDecodeError as e:
            print(f"❌ Error: Invalid JSON in input file: {e}")
            sys.exit(1)
        except Exception as e:
            print(f"❌ Error loading input file: {e}")
            sys.exit(1)

    # Check if resume file exists (only if not using input file)
    if not args.input and not os.path.exists(args.resume_file):
        print(f"❌ Error: Resume file '{args.resume_file}' not found")
        sys.exit(1)

    # Show provider information
    print(f"🤖 Using {args.provider} provider")
    if args.provider in ["openai", "gemini"] and not args.api_key and not os.getenv(f"{args.provider.upper()}_API_KEY"):
        print(f"⚠️  Warning: No API key provided for {args.provider}. Set {args.provider.upper()}_API_KEY environment variable or use --api-key")

    if args.interactive:
        results = matcher.conduct_interactive_improvement(
            args.resume_file, args.job_source, previous_results
        )
    else:
        results = matcher.run_analysis(
            args.resume_file, args.job_source, previous_results
        )

    # Display results
    print("\n" + "=" * 50)
    print("📋 ANALYSIS RESULTS")
    print("=" * 50)

    if args.interactive:
        # Interactive mode results structure
        print(
            f"📊 Final Improved Resume Keywords: {len(results['improved_keywords'])} keywords"
        )
        print(
            f"📈 Original Resume Keywords: {len(results['original_keywords'])} keywords"
        )

        print(f"\n🔑 Original Keywords: {', '.join(results['original_keywords'][:10])}")
        print(f"🔑 Job Keywords: {', '.join(results['job_keywords'][:10])}")
        print(f"🔑 Improved Keywords: {', '.join(results['improved_keywords'][:10])}")

        if args.preview:
            print(f"\n📄 IMPROVED RESUME PREVIEW:")
            print("-" * 30)
            print(results["formatted_preview"])
    else:
        # Regular analysis mode results structure
        print(f"📊 Original Match Score: {results['original_score']:.2%}")
        print(f"📊 Improved Match Score: {results['improved_score']:.2%}")
        print(f"📈 Score Improvement: {results['score_improvement']:.2%}")

        print(f"\n🔑 Resume Keywords: {', '.join(results['resume_keywords'][:10])}")
        print(f"🔑 Job Keywords: {', '.join(results['job_keywords'][:10])}")
        print(f"🔑 New Keywords: {', '.join(results['improved_keywords'][:10])}")

        if args.preview:
            print(f"\n📄 RESUME PREVIEW:")
            print("-" * 30)
            print(results["formatted_preview"])

    # Save results if output file specified
    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n💾 Results saved to {args.output}")

    # Generate formatted resume template if requested
    if args.template_output or args.template != "classic_ats":
        print(f"\n📄 Generating {args.template} template...")

        # Generate the template
        formatted_resume = matcher.improvement_advisor.generate_resume_template(
            results, args.template
        )

        # Save to file if specified
        if args.template_output:
            with open(args.template_output, "w") as f:
                f.write(formatted_resume)
            print(f"💾 Formatted resume saved to {args.template_output}")
        else:
            # Show preview
            print(f"\n📄 {args.template.upper()} TEMPLATE PREVIEW:")
            print("=" * 50)
            print(formatted_resume)

    print(f"\n✅ Analysis complete!")


if __name__ == "__main__":
    main()
