# Simple Resume Matcher

A Python script that matches resumes against job descriptions and provides improvement suggestions using AI with support for multiple LLM providers.

> **Note**: This is a simplified, standalone version inspired by the [Resume-Matcher](https://github.com/srbhr/Resume-Matcher) project. The main Resume-Matcher is a full-stack web application with advanced features, while this version provides the core functionality in a single Python script for easier setup and use.

## Why This Project Exists

I built this application because I wanted a simple way to take my resume, compare it to a job description, and interact with both in a conversational way. Having an LLM look at both documents and ask targeted questions to help improve the quality of my resume is incredibly powerful.

Resume matching and optimization isn't something we do every day, so having an AI-powered tool that can:

- **Analyze gaps** between your resume and job requirements
- **Ask intelligent questions** to uncover missing experiences and achievements
- **Provide specific improvement suggestions** based on the job description
- **Guide you through an interactive improvement process**

...is incredibly valuable when you need to tailor your resume for a specific role.

This tool bridges the gap between generic resume advice and the specific requirements of each job, making the resume optimization process more targeted and effective.

## Features

- Extract text from PDF and DOCX resume files
- Extract job descriptions from text, URLs, or PDF files
- Analyze keyword matching between resume and job description
- Provide AI-powered resume improvement suggestions
- Interactive improvement sessions with targeted questions
- Generate formatted resume templates with 5 different styles
- **Multi-provider LLM support**: Ollama, OpenAI, Gemini, and LM Studio
- **Advanced logging system** with multiple verbosity levels (quiet, normal, verbose, debug)
- **Professional progress tracking** with spinners and detailed status updates
- **Flexible input/output** with JSON save/load capabilities
- **Template generation** from previous analysis results
- **Comprehensive error handling** with troubleshooting guidance

## Installation

1. Install Python dependencies:

```bash
pip install -r requirements.txt
```

2. Set up your preferred LLM provider (see [Provider Setup](#provider-setup) below)

## Provider Setup

### Ollama (Local - Recommended) 🏠

**Free, private, runs locally on your machine**

```bash
# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Start server
ollama serve

# Pull a model
ollama pull llama3.1
```

### LM Studio (Local) 🏠

**Free, private, runs locally on your machine**

1. Download from https://lmstudio.ai/
2. Start local server in LM Studio app
3. Download a model (e.g., `google/gemma-3-27b`)

### OpenAI (Cloud) ☁️

**Paid service, requires API key**

1. Get API key from https://platform.openai.com/
2. Set environment variable: `export OPENAI_API_KEY="your-key"`

### Gemini (Cloud) ☁️

**Free tier available, requires API key**

1. Get API key from https://makersuite.google.com/app/apikey
2. Set environment variable: `export GEMINI_API_KEY="your-key"`

## Usage

### Basic Usage

```bash
# Using Ollama (default, recommended)
python simple_resume_matcher.py resume.pdf "job description text"
python simple_resume_matcher.py resume.pdf https://example.com/job-posting
python simple_resume_matcher.py resume.pdf job_description.pdf

# Using OpenAI
python simple_resume_matcher.py resume.pdf job_description.pdf --provider openai

# Using Gemini
python simple_resume_matcher.py resume.pdf job_description.pdf --provider gemini

# Using LM Studio
python simple_resume_matcher.py resume.pdf job_description.pdf --provider lmstudio
```

### Interactive Mode

```bash
python simple_resume_matcher.py resume.pdf job_description.pdf --interactive
```

### Generate Formatted Resume

```bash
python simple_resume_matcher.py resume.pdf job_description.pdf --template-output improved_resume.txt
```

### Advanced Usage with Logging

```bash
# Verbose output for detailed progress
python simple_resume_matcher.py resume.pdf job_description.pdf --verbose

# Quiet mode (errors only)
python simple_resume_matcher.py resume.pdf job_description.pdf --quiet

# Debug mode (maximum verbosity)
python simple_resume_matcher.py resume.pdf job_description.pdf --debug
```

### Save and Load Analysis Results

```bash
# Save analysis results to JSON
python simple_resume_matcher.py resume.pdf job_description.pdf --output analysis.json

# Load previous results and generate template
python simple_resume_matcher.py --input analysis.json --generate-template accomplishments --template-output final_resume.txt
```

### List Available Options

```bash
# List available resume templates
python simple_resume_matcher.py --list-templates

# List available LLM providers with setup instructions
python simple_resume_matcher.py --list-providers
```

## Command Line Options

### Provider Selection

- `--provider {ollama,openai,gemini,lmstudio}`: Choose LLM provider (default: ollama)
- `--model`: Specify model name (auto-detected if not specified)
- `--api-key`: API key for cloud providers (OpenAI, Gemini)
- `--provider-url`: Custom URL for provider server

### Logging and Verbosity

- `--verbose, -v`: Show detailed progress information
- `--quiet, -q`: Show only errors and critical messages
- `--debug`: Show debug information (very verbose)

### Input/Output Options

- `--output`: Save analysis results to JSON file
- `--input`: Load previous analysis results from JSON file
- `--preview`: Show formatted resume preview
- `--interactive`: Run interactive improvement session

### Template Options

- `--template`: Choose resume template (classic_ats, skills_forward, accomplishments, dual_column, minimalist)
- `--template-output`: Save formatted resume to file
- `--generate-template`: Generate specific template from input file (requires --input)
- `--list-templates`: Show available resume templates
- `--list-providers`: Show available LLM providers and setup instructions

## Resume Templates

The tool includes 5 professionally designed resume templates, each optimized for different career stages and industries:

### 1. Classic ATS-Optimized (Default)

- **Best For**: All technology professionals, high-volume applications
- **Features**: Single-column design, standard fonts, clear headings, no graphics
- **Use Case**: Universal template that works with most Applicant Tracking Systems

### 2. Skills-Forward Hybrid

- **Best For**: Recent graduates, career changers, diverse backgrounds
- **Features**: Skills section at top, reverse-chronological history, transferable skills focus
- **Use Case**: When your skills are more relevant than your work history

### 3. Accomplishments-Centric

- **Best For**: Mid-to-senior-level professionals with clear progression
- **Features**: Quantified achievements, strong action verbs, career story focus
- **Use Case**: When you have impressive metrics and career growth to highlight

### 4. Dual-Column Modern

- **Best For**: Candidates with many projects, skills, and certifications
- **Features**: Two-column format, visual balance, high information density
- **Use Case**: When you need to fit a lot of information in a compact format

### 5. Minimalist LaTeX/Plain-Text

- **Best For**: Software engineers, developers, data scientists
- **Features**: LaTeX/plain-text, technical elegance, function over form
- **Use Case**: Technical roles where clean, simple formatting is preferred

## Examples

### Basic Analysis

```bash
# Simple analysis with Ollama (default)
python simple_resume_matcher.py resume.pdf job.pdf

# Interactive improvement session
python simple_resume_matcher.py resume.pdf job.pdf --interactive
```

### Provider-Specific Examples

```bash
# Using OpenAI with API key
python simple_resume_matcher.py resume.pdf job.pdf --provider openai --api-key sk-...

# Using Gemini with environment variable
export GEMINI_API_KEY="your-key"
python simple_resume_matcher.py resume.pdf job.pdf --provider gemini

# Using LM Studio with custom URL
python simple_resume_matcher.py resume.pdf job.pdf --provider lmstudio --provider-url http://localhost:1234

# Using Ollama with custom URL
python simple_resume_matcher.py resume.pdf job.pdf --provider ollama --provider-url http://192.168.1.100:11434
```

### Advanced Workflow Examples

```bash
# Save analysis results for later use
python simple_resume_matcher.py resume.pdf job.pdf --output analysis.json --verbose

# Generate different templates from saved analysis
python simple_resume_matcher.py --input analysis.json --generate-template accomplishments --template-output accomplishments_resume.txt
python simple_resume_matcher.py --input analysis.json --generate-template minimalist --template-output minimalist_resume.txt

# Complete workflow: analyze, save, and generate multiple templates
python simple_resume_matcher.py resume.pdf job.pdf --interactive --output analysis.json --template-output improved_resume.txt
```

### Debugging and Troubleshooting

```bash
# Debug mode for troubleshooting issues
python simple_resume_matcher.py resume.pdf job.pdf --debug

# Quiet mode for automated scripts
python simple_resume_matcher.py resume.pdf job.pdf --quiet --output results.json

# Verbose mode to see detailed progress
python simple_resume_matcher.py resume.pdf job.pdf --verbose --interactive
```

### Template-Specific Examples

```bash
# Generate skills-forward template for career changers
python simple_resume_matcher.py resume.pdf job.pdf --template skills_forward --template-output skills_resume.txt

# Generate accomplishments template for senior professionals
python simple_resume_matcher.py resume.pdf job.pdf --template accomplishments --template-output accomplishments_resume.txt

# Generate minimalist template for developers
python simple_resume_matcher.py resume.pdf job.pdf --template minimalist --template-output dev_resume.txt
```

## Provider Comparison

| Provider | Privacy | Cost | Setup Difficulty | Performance | Recommended For |
|----------|---------|------|------------------|-------------|-----------------|
| Ollama | ✅ Local | ✅ Free | 🟢 Easy | 🟡 Good | Privacy-conscious users |
| LM Studio | ✅ Local | ✅ Free | 🟡 Medium | 🟢 Excellent | Power users |
| OpenAI | ❌ Cloud | 💰 Paid | 🟢 Easy | 🟢 Excellent | Professional use |
| Gemini | ❌ Cloud | 🟡 Free tier | 🟢 Easy | 🟡 Good | Budget-conscious users |

## Requirements

- Python 3.7+
- One of the supported LLM providers (see [Provider Setup](#provider-setup))
- Internet connection (for URL job descriptions and cloud providers)

## Dependencies

- `markitdown[all]`: PDF/DOCX text extraction
- `requests`: HTTP requests
- `beautifulsoup4`: Web scraping

## Troubleshooting

### Ollama Issues

- **"Connection refused"**: Make sure `ollama serve` is running
- **"Model not found"**: Run `ollama pull llama3.1` to download a model
- **"Permission denied"**: Run `sudo ollama serve` on Linux

### OpenAI Issues

- **"API key not provided"**: Set `export OPENAI_API_KEY="your-key"`
- **"Rate limit exceeded"**: Wait a moment and try again
- **"Model not found"**: Check if the model name is correct

### Gemini Issues

- **"API key not provided"**: Set `export GEMINI_API_KEY="your-key"`
- **"Quota exceeded"**: Check your Google Cloud quota

### LM Studio Issues

- **"Connection refused"**: Make sure LM Studio local server is running
- **"Model not loaded"**: Load a model in LM Studio app

## Benefits of Multi-Provider Support

1. **Flexibility**: Choose the provider that best fits your needs
2. **Privacy**: Local providers keep your data on your machine
3. **Cost Control**: Free local options vs paid cloud services
4. **Reliability**: Fallback options if one provider is unavailable
5. **Performance**: Different providers have different strengths
6. **Accessibility**: More users can use the tool regardless of their setup

## Comparison with Main Resume-Matcher Project

This Simple Resume Matcher is inspired by the [main Resume-Matcher project](https://github.com/srbhr/Resume-Matcher) but designed for different use cases:

| Feature | Simple Resume Matcher | Main Resume-Matcher |
|---------|----------------------|-------------------|
| **Architecture** | Single Python script | Full-stack web app (FastAPI + Next.js) |
| **Setup** | Simple pip install | Docker/containerized setup |
| **UI** | Command-line interface | Modern web interface |
| **Database** | None (in-memory processing) | SQLite with structured models |
| **Deployment** | Run locally | Web application |
| **Features** | Core matching & improvement | Advanced ATS analysis, templates, collaboration |
| **Use Case** | Quick analysis, learning, CLI users | Production use, teams, web interface |

**Choose Simple Resume Matcher if you want:**

- Quick setup and immediate use
- Command-line workflow
- Local processing only
- Learning the core algorithms
- Simple deployment

**Choose Main Resume-Matcher if you want:**

- Full web application experience
- Advanced ATS compatibility analysis
- Team collaboration features
- Production-ready deployment
- Comprehensive resume templates
