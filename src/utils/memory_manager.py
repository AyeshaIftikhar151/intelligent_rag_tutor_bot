# src/utils/memory_manager.py
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain.memory import ConversationBufferMemory

class MemoryManager:
    """
    Handles per-subject conversation memory and stores user's name.
    """

    def __init__(self):
        # Initialize per-subject memory using ConversationBufferMemory
        self.memories = {
            "english": ConversationBufferMemory(
                memory_key="chat_history", return_messages=True, chat_memory=ChatMessageHistory()
            ),
            "physics": ConversationBufferMemory(
                memory_key="chat_history", return_messages=True, chat_memory=ChatMessageHistory()
            ),
            "biology": ConversationBufferMemory(
                memory_key="chat_history", return_messages=True, chat_memory=ChatMessageHistory()
            ),
            "pakistan_studies": ConversationBufferMemory(
                memory_key="chat_history", return_messages=True, chat_memory=ChatMessageHistory()
            ),
        }
        self.user_name = None

    def get_memory(self, subject: str) -> ConversationBufferMemory:
        """Return memory object for a given subject."""
        return self.memories.get(subject)

    def set_user_name(self, name: str):
        """Store the user's name for the session."""
        self.user_name = name

    def get_user_name(self) -> str:
        """Return the stored user name."""
        return self.user_name
