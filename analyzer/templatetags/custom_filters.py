from django import template

# Register a new template filter library so Django knows where to find custom filters
register = template.Library()

@register.filter
def get_item(dictionary, key):
    """
    Custom template filter to retrieve a value from a dictionary using a key.
    
    Usage in Django templates:
        {{ my_dict|get_item:"my_key" }}
    
    Why it's needed:
        - Django templates don't support dictionary-style access like my_dict["my_key"]
        - This filter allows you to dynamically access dictionary values when the key is a variable
        - Especially useful when looping over data and needing to look up related values

    Example:
        {% for col, val in stats.most_freq.items %}
            {{ stats.freq_percent|get_item:col }}%
        {% endfor %}
    """
    return dictionary.get(key)