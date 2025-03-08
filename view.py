from rest_framework.response import Response

# class ChattingWithCobase(CreateAPIView):
#     serializer_class = ChattingWithCobaseSerializer
#     permission_classes = [IsAuthenticated]

#     def post(self, request, *args, **kwargs):
#         serializer = self.get_serializer(data=request.data)
#         serializer.is_valid(raise_exception=True)
#         data = serializer.validated_data

#         chatbot = Chatbot(codebase_name=data["codebase_name"])

#         def stream():
#             for chunk in chatbot.get_streaming_response(data["question"]):
#                 yield f"{chunk}"

#         return StreamingHttpResponse(stream(), content_type="text/plain")


class ChatHistory(CreateAPIView):
    serializer_class = ChattingWithCobaseSerializer
    permission_classes = [IsAuthenticated]

    def perform_create(self, serializer):
        serializer.save(user=self.request.user)

    def create(self, request, *args, **kwargs):
        response = super().create(request, *args, **kwargs)
        chat_history = response.data["id"]
        return Response({"id": chat_history})
