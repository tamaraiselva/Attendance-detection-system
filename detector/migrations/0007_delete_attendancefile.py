# Generated by Django 4.2.5 on 2023-12-25 04:58

from django.db import migrations


class Migration(migrations.Migration):
    dependencies = [
        ("detector", "0006_attendancefile"),
    ]

    operations = [
        migrations.DeleteModel(
            name="AttendanceFile",
        ),
    ]
