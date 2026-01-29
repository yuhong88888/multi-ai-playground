# 🎭 Multi-AI Playground

_A platform where multiple AI agents discuss topics with each other and welcome human participants to join the conversation._

## 🌟 Overview

Multi-AI Playground is an interactive system that enables multiple AI agents with different personalities and perspectives to engage in meaningful discussions on various topics. Human users can join these conversations, contributing their own insights and perspectives alongside the AI agents.

### ✨ Key Features

- **🤖 Multiple AI Agents**: Each agent has a unique personality (optimist, skeptic, analyst, creative, pragmatist)
- **👥 Human Participation**: Humans can join discussions and interact with AI agents in real-time
- **💬 Dynamic Conversations**: Turn-based discussion system with context-aware responses
- **🎯 Topic-Focused**: All discussions center around a specific topic chosen by the user
- **📊 Discussion Analytics**: Track participation, message counts, and conversation flow

## 🚀 Quick Start

### Prerequisites

- Python 3.7 or higher

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yuhong88888/multi-ai-playground.git
cd multi-ai-playground
```

2. (Optional) Install any additional dependencies:
```bash
pip install -r requirements.txt
```

### Running Your First Discussion

**Option 1: Using the main CLI**
```bash
python main.py "The future of artificial intelligence"
```

**Option 2: Run an example**
```bash
python example_ai_only.py
```

## 📖 Usage Guide

### Command-Line Interface

The main CLI provides several options:

```bash
# Basic usage with default settings (includes human participation)
python main.py "Your topic here"

# AI-only discussion (no human)
python main.py "Climate change solutions" --no-human

# Custom number of turns
python main.py "Space exploration" --turns 8

# Custom setup wizard
python main.py "Education technology" --custom
```

### Human Participation

When you join a discussion as a human participant:

1. **Your Turn**: When it's your turn, you'll be prompted to enter your message
2. **Skip**: Type `skip` to pass your turn without contributing
3. **Quit**: Type `quit` to exit the discussion (AI agents will continue)

### Example Programs

Four example programs are included:

1. **`example_ai_only.py`** - Simple AI-only discussion
2. **`example_with_human.py`** - Discussion with human participation
3. **`example_diverse.py`** - Large discussion with diverse AI personalities
4. **`example_bayesian_modeling.py`** - Comprehensive discussion on Bayesian modeling in Python

Run any example:
```bash
python example_ai_only.py
python example_with_human.py
python example_diverse.py
python example_bayesian_modeling.py
```

## 🎯 AI Agent Personalities

Each AI agent has a unique personality that influences their perspective:

| Personality | Description | Example Response Style |
|------------|-------------|----------------------|
| **Optimist** | Positive, enthusiastic, sees potential | "This has tremendous potential!" |
| **Skeptic** | Critical, questioning, cautious | "We should consider the challenges..." |
| **Analyst** | Data-driven, logical, methodical | "Looking at this analytically..." |
| **Creative** | Innovative, imaginative, unconventional | "What if we approach this differently?" |
| **Pragmatist** | Practical, action-oriented, realistic | "The practical approach would be..." |

## 🔧 Programmatic Usage

You can also use the Discussion API directly in your Python code:

```python
from discussion import DiscussionBuilder

# Create a custom discussion
discussion = (
    DiscussionBuilder("Your topic here")
    .add_ai_agent("Alice", "optimist")
    .add_ai_agent("Bob", "skeptic")
    .add_ai_agent("Charlie", "analyst")
    .add_human_agent("You")
    .set_max_turns(5)
    .build()
)

# Run the discussion
discussion.run()
```

## 📁 Project Structure

```
multi-ai-playground/
├── agents.py                      # Agent classes (AI and Human)
├── discussion.py                  # Discussion manager and builder
├── main.py                       # Main CLI application
├── example_ai_only.py            # Example: AI-only discussion
├── example_with_human.py         # Example: Human participation
├── example_diverse.py            # Example: Diverse agent discussion
├── example_bayesian_modeling.py  # Example: Bayesian modeling discussion
├── tutorials/                    # Educational guides and tutorials
│   └── bayesian_modeling_guide.md # Comprehensive Bayesian modeling guide
├── requirements.txt              # Python dependencies
└── README.md                    # This file
```

## 📚 Tutorials & Guides

### Bayesian Modeling and Computation in Python

A comprehensive guide covering Bayesian inference concepts, popular Python libraries, and practical applications:

**📖 [Bayesian Modeling Guide](tutorials/bayesian_modeling_guide.md)**

This tutorial includes:
- **Key Concepts**: Prior distributions, likelihood, posterior, MCMC
- **Popular Libraries**: PyMC, Stan, TensorFlow Probability, Pyro, ArviZ
- **Code Examples**: Linear regression, hierarchical models, time series
- **Practical Applications**: A/B testing, machine learning, finance, NLP
- **Best Practices**: Prior selection, model diagnostics, validation
- **Resources**: Books, courses, documentation, and community links

**Run the interactive discussion:**
```bash
python example_bayesian_modeling.py
```

This example features a multi-agent discussion exploring Bayesian methods from different perspectives (analyst, pragmatist, optimist, skeptic, creative), demonstrating how AI agents can collaboratively explore complex data science topics.

## 🎨 Customization

### Creating Custom Agents

You can extend the `AIAgent` class to create agents with custom behaviors:

```python
from agents import AIAgent

class CustomAgent(AIAgent):
    def respond(self, topic, conversation_history):
        # Your custom logic here
        return "Custom response"
```

### Adjusting Discussion Parameters

- **Number of turns**: Control how long the discussion runs
- **Number of agents**: Add as many agents as you want
- **Agent mix**: Choose different combinations of personalities
- **Human participation**: Include or exclude human participants

## 🤝 Contributing

Contributions are welcome! Feel free to:

- Report bugs or suggest features via issues
- Submit pull requests with improvements
- Share interesting discussion topics or agent configurations

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🌐 Use Cases

- **Education**: Explore different perspectives on complex topics (e.g., Bayesian modeling, machine learning)
- **Data Science Learning**: Understand statistical concepts through multi-perspective discussions
- **Brainstorming**: Generate diverse ideas through multi-perspective discussion
- **Research**: Study conversation dynamics and argument patterns
- **Technical Exploration**: Deep-dive into programming concepts, libraries, and best practices
- **Entertainment**: Create engaging dialogues on interesting subjects
- **Training**: Practice moderation and discussion facilitation skills

## 💡 Example Discussion Output

```
🎭 MULTI-AGENT DISCUSSION STARTING
📌 Topic: The future of artificial intelligence
👥 Participants: Alex (AI-optimist), Blake (AI-skeptic), Casey (AI-analyst), You (human)
🔄 Maximum turns: 3

============================================================
🗣️  Alex [AI-optimist]:
============================================================
I think The future of artificial intelligence has tremendous 
potential! it could revolutionize how we think about collaboration
============================================================

[... and so on ...]
```

## 🙏 Acknowledgments

Built with Python and designed to foster meaningful multi-perspective discussions on any topic.

---

**Ready to start?** Run `python main.py "Your topic"` and join the conversation! 🚀
