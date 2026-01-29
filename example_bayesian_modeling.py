#!/usr/bin/env python3
"""
Example: Multi-Agent Discussion on Bayesian Modeling and Computation in Python

This example demonstrates a discussion among AI agents with different perspectives
on Bayesian modeling, covering key concepts, popular libraries, and practical applications.

The discussion covers:
- Introduction to Bayesian inference and its advantages
- Popular Python libraries (PyMC, Stan, TensorFlow Probability)
- Practical applications in data science and machine learning
- Best practices and common pitfalls
- Statistical analysis and data visualization techniques

Related to: tutorials/bayesian_modeling_guide.md
"""

from discussion import DiscussionBuilder


def main():
    print("\n" + "="*70)
    print("📊 BAYESIAN MODELING & COMPUTATION IN PYTHON")
    print("="*70)
    print("A comprehensive discussion on Bayesian methods, libraries, and applications")
    print("="*70 + "\n")
    
    # Create a rich discussion about Bayesian modeling
    discussion = (
        DiscussionBuilder("Bayesian modeling and computation in Python: concepts, libraries, and applications")
        .add_ai_agent("Dr. Bayes", "analyst")      # Data-driven statistical perspective
        .add_ai_agent("PyMC Expert", "pragmatist") # Practical implementation focus
        .add_ai_agent("ML Enthusiast", "optimist") # Machine learning applications
        .add_ai_agent("Code Reviewer", "skeptic")  # Critical analysis of approaches
        .add_ai_agent("Innovator", "creative")     # Novel applications and ideas
        .set_max_turns(6)
        .build()
    )
    
    # Run the discussion
    discussion.run()
    
    # Print additional information
    print("\n" + "="*70)
    print("📚 LEARN MORE")
    print("="*70)
    print("For a comprehensive guide to Bayesian modeling in Python, see:")
    print("➡️  tutorials/bayesian_modeling_guide.md")
    print("\nKey topics covered:")
    print("  • Bayes' Theorem and key concepts")
    print("  • Popular Python libraries (PyMC, Stan, TFP, Pyro, ArviZ)")
    print("  • Practical code examples and applications")
    print("  • Best practices and common pitfalls")
    print("  • Resources for further learning")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
