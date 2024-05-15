import openai
from message.config import get_settings


class OpenAIKeys(str):
    ROLE = "role"
    CONTENT = "content"
    ANSWER = "answer"


class ChatModel:
    def __init__(self):
        settings = get_settings()
        openai.api_key = settings.OPENAI_API_KEY

    def get_completion(
        self,
        **kwargs,
    ) -> str:
        """Creates a new chat completion for the provided messages and parameters.

        Returns
        -------
        str
            The chat completion response.
        """

        chat_completion = openai.ChatCompletion.create(**kwargs)

        return chat_completion.choices[0].message[OpenAIKeys.CONTENT]
