"""
Discussion Manager for orchestrating multi-agent conversations.
Manages turn-taking, message flow, and human participation.
"""

from typing import List
from agents import Agent, AIAgent, HumanAgent
import time


class Discussion:
    """Manages a multi-agent discussion on a specific topic"""
    
    def __init__(self, topic: str, agents: List[Agent], max_turns: int = 10):
        self.topic = topic
        self.agents = agents
        self.max_turns = max_turns
        self.conversation_history: List[dict] = []
        self.current_turn = 0
    
    def add_message(self, speaker: str, message: str):
        """Add a message to the conversation history"""
        self.conversation_history.append({
            "turn": self.current_turn,
            "speaker": speaker,
            "message": message
        })
        
        # Update all agents' histories
        for agent in self.agents:
            agent.add_to_history(message, speaker)
    
    def display_message(self, speaker: str, message: str, role: str = ""):
        """Display a message in a formatted way"""
        role_tag = f" [{role}]" if role else ""
        print(f"\n{'='*60}")
        print(f"🗣️  {speaker}{role_tag}:")
        print(f"{'='*60}")
        print(f"{message}")
        print(f"{'='*60}")
    
    def run(self):
        """Run the discussion"""
        print("\n" + "🎭 " * 20)
        print(f"🎯 MULTI-AGENT DISCUSSION STARTING")
        print(f"📌 Topic: {self.topic}")
        print(f"👥 Participants: {', '.join([f'{a.name} ({a.role})' for a in self.agents])}")
        print(f"🔄 Maximum turns: {self.max_turns}")
        print("🎭 " * 20 + "\n")
        
        # Opening statement
        self.display_message(
            "Moderator",
            f"Welcome everyone! Today we'll be discussing: '{self.topic}'\n\n"
            f"We have {len(self.agents)} participants including AI agents and humans. "
            f"Let's have an engaging and thoughtful conversation!",
            "System"
        )
        
        time.sleep(1)
        
        # Main discussion loop
        for turn in range(self.max_turns):
            self.current_turn = turn + 1
            
            print(f"\n{'🔹'*30}")
            print(f"   TURN {self.current_turn}/{self.max_turns}")
            print(f"{'🔹'*30}\n")
            
            # Create a copy to iterate over to avoid modification during iteration
            agents_to_process = self.agents.copy()
            
            # Each agent gets a turn
            for agent in agents_to_process:
                # Skip if agent was already removed
                if agent not in self.agents:
                    continue
                    
                # Get agent's response
                try:
                    response = agent.respond(self.topic, self.conversation_history)
                    
                    # Check if human wants to quit
                    if response == "QUIT":
                        print("\n👋 Human participant has left the discussion.")
                        print("💬 Continuing with AI agents...\n")
                        self.agents.remove(agent)
                        continue
                    
                    # Display and record the message
                    self.display_message(agent.name, response, agent.role)
                    self.add_message(agent.name, response)
                    
                    # Small delay for readability
                    time.sleep(0.5)
                    
                except KeyboardInterrupt:
                    print("\n\n⚠️  Discussion interrupted by user.")
                    self.conclude()
                    return
                except Exception as e:
                    print(f"\n❌ Error with agent {agent.name}: {e}")
                    continue
            
            # Check if we should continue
            if not self.agents:
                print("\n⚠️  No more participants. Ending discussion.")
                break
        
        self.conclude()
    
    def conclude(self):
        """Conclude the discussion and show summary"""
        print("\n" + "🎬 " * 20)
        print("📊 DISCUSSION SUMMARY")
        print("🎬 " * 20 + "\n")
        
        print(f"Topic: {self.topic}")
        print(f"Total messages: {len(self.conversation_history)}")
        print(f"Participants: {len(set(msg['speaker'] for msg in self.conversation_history))}")
        
        # Count messages per participant
        print("\n💬 Message Count by Participant:")
        message_counts = {}
        for msg in self.conversation_history:
            speaker = msg['speaker']
            message_counts[speaker] = message_counts.get(speaker, 0) + 1
        
        for speaker, count in sorted(message_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"   • {speaker}: {count} messages")
        
        print("\n✅ Discussion concluded successfully!")
        print("🎬 " * 20 + "\n")


class DiscussionBuilder:
    """Builder class for creating discussions with various configurations"""
    
    def __init__(self, topic: str):
        self.topic = topic
        self.agents: List[Agent] = []
        self.max_turns = 10
    
    def add_ai_agent(self, name: str, personality: str = "analyst"):
        """Add an AI agent with a specific personality"""
        valid_personalities = ["optimist", "skeptic", "analyst", "creative", "pragmatist"]
        if personality not in valid_personalities:
            raise ValueError(f"Invalid personality '{personality}'. Must be one of: {', '.join(valid_personalities)}")
        self.agents.append(AIAgent(name, personality))
        return self
    
    def add_human_agent(self, name: str = "Human"):
        """Add a human participant"""
        self.agents.append(HumanAgent(name))
        return self
    
    def set_max_turns(self, turns: int):
        """Set maximum number of discussion turns"""
        if turns < 1:
            raise ValueError(f"Number of turns must be at least 1, got {turns}")
        self.max_turns = turns
        return self
    
    def build(self) -> Discussion:
        """Build and return the discussion"""
        if not self.agents:
            raise ValueError("Discussion must have at least one agent")
        return Discussion(self.topic, self.agents, self.max_turns)
