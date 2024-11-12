from django.db import models


class Years(models.Model):
    year = models.IntegerField(unique=True, primary_key=True)

    def __str__(self) -> str:
        return f"{str(self.year)}"


class CrudeBirthRate(models.Model):
    year = models.ForeignKey(Years, on_delete=models.CASCADE, null=True, blank=True)
    entity = models.CharField(max_length=100, blank=True, null=True)
    birth_rate = models.DecimalField(null=True, blank=True, decimal_places=3, max_digits=10)

    def __str__(self) -> str:
        return f"{str(self.year)} - {str(self.entity)} - {str(self.birth_rate)}"


class PopulationAndDemography(models.Model):
    year = models.ForeignKey(Years, on_delete=models.CASCADE, null=True, blank=True)
    entity = models.CharField(max_length=100, blank=True, null=True)
    population = models.IntegerField(null=True, blank=True)

    def __str__(self) -> str:
        return f"{str(self.year)} - {str(self.entity)} - {str(self.population)}"


class PoliticalRegieme(models.Model):
    year = models.ForeignKey(Years, on_delete=models.CASCADE, null=True, blank=True)
    entity = models.CharField(max_length=100, blank=True, null=True)
    political_regime = models.IntegerField(null=True, blank=True)

    def __str__(self) -> str:
        return f"{str(self.year)} - {str(self.entity)} - {str(self.political_regime)}"


#! We also have a religious wide but for now it serves our purpose
class ReligiousLarge(models.Model):
    year = models.ForeignKey(Years, on_delete=models.CASCADE, null=True, blank=True)
    entity = models.CharField(max_length=100, blank=True, null=True)
    group_name = models.CharField(max_length=100, blank=True, null=True)
    group_proportion = models.DecimalField(null=True, blank=True, decimal_places=3, max_digits=10)
    group_estimate = models.DecimalField(null=True, blank=True, decimal_places=3, max_digits=10)
    outlier = models.IntegerField(null=True, blank=True)
    anchor = models.IntegerField(null=True, blank=True)
    independent_country = models.BooleanField(null=True, blank=True)

    def __str__(self) -> str:
        return f"{str(self.year)} - {str(self.entity)} - {str(self.group_name)} - {str(self.group_proportion)}"
