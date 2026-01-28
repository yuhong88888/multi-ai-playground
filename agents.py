"""
Agent classes for multi-agent discussion system.
Supports both AI agents and human participants.
"""

import random
from typing import List, Dict
from datetime import datetime


class Agent:
    """Base class for all agents (AI and Human)"""
    
    def __init__(self, name: str, role: str = "participant"):
        self.name = name
        self.role = role
        self.message_history: List[Dict] = []
    
    def respond(self, topic: str, conversation_history: List[Dict]) -> str:
        """Generate a response based on the topic and conversation history"""
        raise NotImplementedError("Subclasses must implement respond method")
    
    def add_to_history(self, message: str, speaker: str):
        """Add a message to this agent's history"""
        self.message_history.append({
            "speaker": speaker,
            "message": message,
            "timestamp": datetime.now().isoformat()
        })


class AIAgent(Agent):
    """AI Agent that simulates conversation on a topic"""
    
    # Predefined response templates for different agent personalities
    PERSONALITY_TEMPLATES = {
        "optimist": [
            "I think {topic} has tremendous potential! {insight}",
            "That's a great point! Building on that, {insight}",
            "I'm excited about {topic} because {insight}",
            "This is wonderful! {insight}",
        ],
        "skeptic": [
            "While I see the merit, we should consider the challenges with {topic}. {insight}",
            "I have some concerns about {topic}. {insight}",
            "Let me play devil's advocate here - {insight}",
            "We need to be realistic about {topic}. {insight}",
        ],
        "analyst": [
            "Looking at {topic} analytically, {insight}",
            "The data suggests that {insight}",
            "From a research perspective on {topic}, {insight}",
            "If we break down {topic}, we can see that {insight}",
        ],
        "creative": [
            "What if we approach {topic} from a completely different angle? {insight}",
            "Here's an innovative idea about {topic}: {insight}",
            "Let me share a creative perspective - {insight}",
            "Imagine if {topic} could be reimagined as {insight}",
        ],
        "pragmatist": [
            "For {topic}, the practical approach would be {insight}",
            "Let's focus on actionable steps for {topic}. {insight}",
            "In real-world terms, {topic} means {insight}",
            "From an implementation standpoint, {insight}",
        ]
    }
    
    INSIGHTS = [
        "it could revolutionize how we think about collaboration",
        "we need to consider both technical and social implications",
        "the key is finding the right balance between innovation and practicality",
        "we should look at successful examples from other domains",
        "the most important factor is user experience and accessibility",
        "we must ensure ethical considerations are at the forefront",
        "scalability and sustainability are crucial factors",
        "this could foster more inclusive and diverse perspectives",
        "we should leverage existing technologies and frameworks",
        "the community feedback will be essential for success",
        "we need robust testing and validation processes",
        "this opens up new possibilities for creative expression",
        "we should prioritize transparency and open communication",
        "the integration with existing systems is a key challenge",
        "we can learn from both successes and failures in this space",
    ]
    
    def __init__(self, name: str, personality: str = "analyst"):
        super().__init__(name, role=f"AI-{personality}")
        self.personality = personality
        self.response_count = 0
    
    def respond(self, topic: str, conversation_history: List[Dict]) -> str:
        """Generate a response based on personality and conversation context"""
        self.response_count += 1
        
        # Select template based on personality
        templates = self.PERSONALITY_TEMPLATES.get(
            self.personality, 
            self.PERSONALITY_TEMPLATES["analyst"]
        )
        
        # Add variation based on conversation progress
        if self.response_count == 1:
            # First response - introduce perspective
            template = random.choice(templates)
        elif len(conversation_history) > 5:
            # Later in conversation - reference previous points
            if conversation_history:
                last_msg = conversation_history[-1]
                references = [
                    f"Building on what {last_msg['speaker']} said, ",
                    f"I agree with {last_msg['speaker']} that ",
                    f"To add to {last_msg['speaker']}'s point, ",
                    "Considering what's been discussed, ",
                ]
                prefix = random.choice(references)
                template = prefix + random.choice(templates).split("{insight}")[0] + "{insight}"
            else:
                template = random.choice(templates)
        else:
            template = random.choice(templates)
        
        # Generate insight
        insight = random.choice(self.INSIGHTS)
        
        # Format the response
        response = template.format(topic=topic, insight=insight)
        
        return response


class HumanAgent(Agent):
    """Human participant in the discussion"""
    
    def __init__(self, name: str = "Human"):
        super().__init__(name, role="human")
    
    def respond(self, topic: str, conversation_history: List[Dict]) -> str:
        """Get human input for response"""
        print(f"\n{self.name}, it's your turn to contribute to the discussion about '{topic}'")
        print("Enter your message (or 'skip' to pass this turn, 'quit' to exit):")
        
        user_input = input("> ").strip()
        
        if user_input.lower() == 'quit':
            return "QUIT"
        elif user_input.lower() == 'skip' or not user_input:
            return f"[{self.name} is listening]"
        else:
            return user_input
