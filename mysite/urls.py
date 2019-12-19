"""mysite URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/2.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path
from . import views
from . import predictor
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', views.home, name='home'),
    path('tools', views.tools, name='tools'),
    path('alignments',views.alignments, name='alignments'),
    path('analyze',views.analyze, name='analyze'),
    path('dna_to_rna',views.dna_to_rna, name='dna_to_rna'),
    path('mrna_to_protein',views.mrna_to_protein, name='mrna_to_protein'),
    path('complement', views.complement, name='complement'),
    path('gc_content', views.gc_content, name='gc_content'),
    path('count_mut', views.count_mut, name='count_mut'),
    path('count_nucleo', views.count_nucleo, name='count_nucleo'),
    path('globa_align', views.globa_align, name='globa_align'),
    path('local_align', views.local_align, name='local_align'),
    path('pred_gene_family', views.pred_gene_family, name='pred_gene_family'),
    path('result_tools', views.result_tools, name='result_tools'),
    path('result_tools1', views.result_tools1, name='result_tools1'),
    path('result_tools2', views.result_tools2, name='result_tools2'),
    path('result_tools3', views.result_tools3, name='result_tools3'),
    path('result_tools4', views.result_tools4, name='result_tools4'),
    path('result_tools5', views.result_tools5, name='result_tools5'),
    path('result_class', predictor.result_class, name='result_class'),
    path('pred_newgene', views.pred_newgene, name='pred_newgene'),
    path('result_newgene', predictor.result_newgene, name='result_newgene'),
    path('result_global', views.result_global, name='result_global'),
    path('result_local', views.result_local, name='result_local'),
] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)





