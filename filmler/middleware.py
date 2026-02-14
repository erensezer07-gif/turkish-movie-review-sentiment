class SecurityHeadersMiddleware:
    """
    YouTube embed'ler ve genel güvenlik için HTTP header'ları ekler.
    """

    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        response = self.get_response(request)
        response["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response["X-Content-Type-Options"] = "nosniff"
        return response
