# Generated by Django 2.1.2 on 2018-11-03 19:10

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('app', '0003_channel_logo'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='logo',
            name='date',
        ),
        migrations.AddField(
            model_name='logo',
            name='timestamp',
            field=models.IntegerField(default=0),
        ),
    ]
