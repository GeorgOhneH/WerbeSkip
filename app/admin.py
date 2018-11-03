from django.contrib import admin
from .models import Room, Channel, Logo


admin.site.register(
    Room,
    list_display=["id", "title", "staff_only"],
    list_display_links=["id", "title"],
)
admin.site.register(
    Channel,
    list_display=["id", "name", "teleboy_id"],
    list_display_links=["id", "name"],
)
admin.site.register(
    Logo,
    list_display=["id", "status", "channel"],
    list_display_links=["id", "status", "channel"],
)
