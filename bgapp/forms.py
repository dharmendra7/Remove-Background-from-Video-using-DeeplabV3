from cProfile import label
from django  import forms
from bgapp.models import *


class Uploadfileform(forms.ModelForm):
	class Meta:
		model = Upload
		fields = ('upload',)