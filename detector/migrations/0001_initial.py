# Generated by Django 4.2.5 on 2023-12-19 06:01

from django.db import migrations, models


class Migration(migrations.Migration):
    initial = True

    dependencies = []

    operations = [
        migrations.CreateModel(
            name="add",
            fields=[
                (
                    "id",
                    models.BigAutoField(
                        auto_created=True,
                        primary_key=True,
                        serialize=False,
                        verbose_name="ID",
                    ),
                ),
                ("newusername", models.CharField(max_length=255)),
                ("newuserid", models.CharField(max_length=255)),
            ],
        ),
    ]
