# Generated by Django 5.1.3 on 2024-11-23 14:25

import django.db.models.deletion
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("src", "0003_alter_crudebirthrate_birth_rate_and_more"),
    ]

    operations = [
        migrations.CreateModel(
            name="Population",
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
                ("population", models.IntegerField(blank=True, null=True)),
                ("entity", models.CharField(blank=True, max_length=100, null=True)),
                (
                    "year",
                    models.ForeignKey(
                        blank=True,
                        null=True,
                        on_delete=django.db.models.deletion.CASCADE,
                        to="src.years",
                    ),
                ),
            ],
        ),
    ]
