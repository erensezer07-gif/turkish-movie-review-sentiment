from django import template

register = template.Library()

@register.filter(name='convert_to_embed')
def convert_to_embed(value):
    """
    Converts a standard YouTube watch URL to an embed URL.
    Example: https://www.youtube.com/watch?v=VIDEO_ID -> https://www.youtube.com/embed/VIDEO_ID
    OR: https://youtu.be/VIDEO_ID -> https://www.youtube.com/embed/VIDEO_ID
    If already embed, returns as is.
    """
    if not value:
        return ""
    
    # Clean string
    url = str(value).strip()
    
    # If already embed
    if "youtube.com/embed/" in url:
        return url
        
    # Replace watch?v= with embed/
    if "watch?v=" in url:
        return url.replace("watch?v=", "embed/")
    
    # Handle youtu.be short links
    if "youtu.be/" in url:
        return url.replace("youtu.be/", "www.youtube.com/embed/")
        
    return url
