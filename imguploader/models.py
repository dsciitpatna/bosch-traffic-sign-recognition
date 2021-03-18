from django.db import models
import random
import os
from django.db.models.signals import pre_save, post_save
from django.db.models import Q

from .utils import unique_slug_generator


def get_filename_ext(filepath):
    base_name = os.path.basename(filepath)
    name, ext = os.path.splitext(base_name)
    return name, ext


def upload_image_path(instance, filename):
    print(instance)
    new_filename = random.randint(1, 3910209312)
    name, ext = get_filename_ext(filename)
    final_filename = '{new_filename}{ext}'.format(
        new_filename=new_filename, ext=ext)
    return "products/{new_filename}/{final_filename}".format(
        new_filename=new_filename,
        final_filename=final_filename
    )


class ImageQuerySet(models.query.QuerySet):
    def active(self):
        return self.filter(active=True)

    def search(self, query):
        lookups = (Q(slug__icontains=query)
                   )
        return self.filter(lookups).distinct()


# Query parameters decoder. Its splitted in manager
class ImageManager(models.Manager):
    def get_queryset(self):
        return ImageQuerySet(self.model, using=self._db)

    def all(self):
        return self.get_queryset().active()

    def get_by_id(self, id):
        qs = self.get_queryset().filter(id=id)
        if qs.count() == 1:
            return qs.first()
        return None

    def search(self, query):
        return self.get_queryset().active().search(query)


class ImgUp(models.Model):
    # Slug for trversing between images usign custom url
    slug = models.SlugField(blank=True, unique=True)
    image = models.ImageField(
        upload_to=upload_image_path, null=True, blank=True)
    active = models.BooleanField(default=True)

    objects = ImageManager()  # Object of ImageManager

    def get_absolute_url(self):
        return "/img/view/{slug}/".format(slug=self.slug)
        # return reverse("products:detail", kwargs={"slug": self.slug})

    def __str__(self):
        return self.slug

    def __unicode__(self):
        return self.slug


# its is a function thats is called by the pre_sav funtion automatically
# when a new image is uploaded
def product_pre_save_receiver(sender, instance, *args, **kwargs):
    if not instance.slug:
        instance.slug = unique_slug_generator(instance)


# It gets executed automatically whenever new image is uploaded.
pre_save.connect(product_pre_save_receiver, sender=ImgUp)
