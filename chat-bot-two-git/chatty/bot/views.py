# views.py
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
import openai

openai.api_key = "api_key_goes_here"


class ChatAPIView(APIView):
    def post(self, request):
        user_message = request.data.get("message")
        if not user_message:
            return Response({"error": "Message not provided"}, status=status.HTTP_400_BAD_REQUEST)

        # Call the OpenAI API
        response = openai.ChatCompletion.create(
            model="gpt-4-turbo",
            messages=[{"role": "user", "content": user_message}]
        )
        bot_message = response.choices[0].message.content.strip()
        return Response({"message": bot_message}, status=status.HTTP_200_OK)
