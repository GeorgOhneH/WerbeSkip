from channels.db import database_sync_to_async

from .exceptions import ClientError
from .models import Room, Channel, Logo

from django.forms.models import model_to_dict

import time


# This decorator turns this function from a synchronous function into an async one
# we can call from our async consumers, that handles Django DBs correctly.
# For more, see http://channels.readthedocs.io/en/latest/topics/databases.html
@database_sync_to_async
def get_room_or_error(room_name):
    """
    Tries to fetch a room for the user, checking permissions along the way.
    """
    # Find the room they requested (by ID)
    try:
        room = Room.objects.get(title=room_name)
    except Room.DoesNotExist:
        raise ClientError("ROOM_INVALID")
    return room


@database_sync_to_async
def do_inti_db(channels):
    for channel_name in channels.keys():
        try:
            Channel.objects.get(name=channel_name)
            continue
        except Channel.DoesNotExist:
            pass
        channel = Channel.objects.create(name=channel_name, teleboy_id=channels[channel_name]["id"])
        channel.save()


async def update_db(channels, last_channel):
    for channel_name in channels.keys():
        last_status = last_channel.get(channel_name, {}).get("ad", None)
        status = channels[channel_name]["ad"]
        if last_status == status:
            continue
        await _update_db(channel_name, status)


@database_sync_to_async
def _update_db(channel_name, status):
        try:
            channel = Channel.objects.get(name=channel_name)
        except Channel.DoesNotExist:
            raise ClientError("CHANNEL NOT INIT")

        logo = channel.logo_set.create(status=status, timestamp=int(time.time()))
        logo.save()


@database_sync_to_async
def get_db(room_name, duration):
    result = {}
    for channel in Channel.objects.all():
        result[channel.name] = []
        for logo in channel.logo_set.filter(timestamp__gt=int(time.time())-duration):
            if len(result[channel.name]) == 0:
                result[channel.name].append(
                    {"ad": logo.status, "x": -duration, "id": channel.teleboy_id})
            result[channel.name].append({"ad": logo.status, "x": logo.timestamp-int(time.time()), "id": channel.teleboy_id})
    return result


