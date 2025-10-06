# src/memory/session_memory.py
"""
SessionMemory: Simple persistent memory for chat sessions.
Stores messages in a JSON file with role and content.
"""

import json
import os

class SessionMemory:
    def __init__(self, file_path: str = "session_memory.json"):
        """
        Initialize session memory.
        If the file exists, load previous session data.

        Parameters:
        - file_path: Path to the JSON file for storing messages.
        """
        self.file_path = file_path
        self.data = []

        if os.path.exists(file_path):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    self.data = json.load(f)
            except (json.JSONDecodeError, IOError):
                self.data = []

    def add(self, role: str, content: str):
        """
        Add a message to session memory and persist it.

        Parameters:
        - role: "user" or "tutor"
        - content: Message text
        """
        self.data.append({"role": role, "content": content})
        self.save()

    def save(self):
        """Save session memory to the JSON file."""
        try:
            with open(self.file_path, "w", encoding="utf-8") as f:
                json.dump(self.data, f, indent=2, ensure_ascii=False)
        except IOError as e:
            print(f"Error saving session memory: {e}")

    def show(self):
        """Return the current session memory as a list of messages."""
        return self.data
