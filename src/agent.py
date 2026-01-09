from a2a.server.tasks import TaskUpdater
from a2a.types import Message, TaskState, Part, TextPart
from a2a.utils import get_message_text, new_agent_text_message

from messenger import Messenger
import litellm
from dotenv import load_dotenv
import json

RESPONSE_KEY = "action"
CONTENT_KEY = "content"

load_dotenv()

class Agent:
    def __init__(self):
        self.messenger = Messenger()
        # Initialize other state here
        self.messages = []

    async def run(self, message: Message, updater: TaskUpdater) -> None:
        """Implement your agent logic here.

        Args:
            message: The incoming message
            updater: Report progress (update_status) and results (add_artifact)

        Use self.messenger.talk_to_agent(message, url) to call other agents.
        """

        input_text = get_message_text(message)
        print(f"input > {input_text}")

        await updater.update_status(
            TaskState.working, new_agent_text_message("Thinking...")
        )

        self.messages.append({"content": input_text, "role": "user"})
        completion = litellm.completion(
            model="openrouter/qwen/qwen3-coder:free", #"openrouter/z-ai/glm-4.5-air:free"
            messages=self.messages
        )
        response = completion.choices[0].message.content
        self.messages.append({"content": response, "role": "assistant"})
        print("response > ", response)
  
        response_json = json.loads(response)
        await updater.add_artifact(
            name=response_json[RESPONSE_KEY],
            parts=[Part(root=TextPart(text=response_json[CONTENT_KEY]))],
        )
