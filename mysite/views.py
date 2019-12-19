# I have created
from django.http import HttpResponse
from django.shortcuts import render
import pandas as pd
import Bio
from Bio.Seq import Seq
from Bio import SeqIO
from Bio.SeqRecord import SeqRecord
from Bio.Alphabet import generic_dna
from Bio.SeqUtils import GC
from Bio import pairwise2
from Bio.pairwise2 import format_alignment

def home (request):
    return render(request, 'mysite/home.html')
def tools(request):
    return render(request, 'mysite/tools.html')
def alignments(request):
    return render(request, 'mysite/alignments.html')
def analyze(request):
    return render(request,'mysite/analyze.html')
def dna_to_rna (request):
    return render(request, 'mysite/dna_to_rna.html')
def mrna_to_protein (request):
    return render(request, 'mysite/mrna_to_protein.html')
def complement (request):
    return render(request, 'mysite/complement.html')
def gc_content (request):
    return render(request, 'mysite/gc_content.html')
def count_mut (request):
    return render(request, 'mysite/count_mut.html')
def count_nucleo (request):
    return render(request, 'mysite/count_nucleo.html')
def globa_align (request):
    return render(request, 'mysite/globa_align.html')
def local_align (request):
    return render(request, 'mysite/local_align.html')
def pred_gene_family (request):
    return render(request, 'mysite/pred_gene_family.html')
def pred_newgene (request):
    return render(request,'mysite/pred_newgene.html')
def result_tools (request):
    input_seq=request.POST.get('tool1','default')
    rna2 = Seq(input_seq)
    ans=rna2.transcribe()
    params={'res':ans}
    return render(request, 'mysite/result_tools.html',params)
def result_tools1 (request):
    input_seq=request.POST.get('tool1','default')
    mrna8 = Seq(input_seq)
    ans=mrna8.translate(to_stop=True)
    params={'res':ans}
    return render(request, 'mysite/result_tools.html',params)
def result_tools2 (request):
    input_seq=request.POST.get('tool1','default')
    dna3 = Seq(input_seq)
    ans=dna3.reverse_complement()
    params={'res':ans}
    return render(request, 'mysite/result_tools.html',params)
def result_tools3 (request):
    input_seq=request.POST.get('tool1','default')
    rec1 = Seq(input_seq)
    ans= GC(rec1)
    params={'res':ans}
    return render(request, 'mysite/result_tools.html',params)
def result_tools4 (request):
    input_seq=request.POST.get('tool1','default')
    dna1 = Seq(input_seq)
    ans= str("A == " + str(dna1.count("A"))) + " " + "C == " + str(dna1.count("C")) + " " + "G == " + str(dna1.count("G")) + " " + "T == " + str(dna1.count("T"))
    params={'res':ans}
    return render(request, 'mysite/result_tools.html',params)
def result_tools5 (request):
    count = 0
    hamming = 0
    input_seq = request.POST.get('tool1', 'default')
    input_seq2 = request.POST.get('tool2', 'default')
    dna6a = Seq(input_seq)
    dna6b = Seq(input_seq2)
    if(len(dna6a)<len(dna6b)):
        while (count < len(dna6a)):
            if (dna6a[count] != dna6b[count]):
                hamming = hamming + 1
            count = count + 1
    else:
        while (count < len(dna6b)):
            if (dna6a[count] != dna6b[count]):
                hamming = hamming + 1
            count = count + 1
    ans = hamming
    params = {'res': ans}
    return render(request, 'mysite/result_tools.html', params)

def result_global (request):
    input_seq = request.POST.get('tool1', 'default')
    input_seq2 = request.POST.get('tool2', 'default')
    str1=""
    for x in input_seq:
        if(x!=" "):
            str1=str1+x
    str2=""
    for x in input_seq2:
        if(x!=" "):
            str2=str2+x
    dna6a = Seq(str1)
    dna6b = Seq(str2)
    alignments = pairwise2.align.globalms(dna6a, dna6b, 2, -1, -0.5, -0.1)
    stri=""
    for a in alignments:
        stri=stri+str(format_alignment(*a))
    params={'res':stri}
    return render(request, 'mysite/result_align.html', params)
def result_local(request):
    input_seq = request.POST.get('tool1', 'default')
    input_seq2 = request.POST.get('tool2', 'default')
    str1=""
    for x in input_seq:
        if(x!=" "):
            str1=str1+x
    str2=""
    for x in input_seq2:
        if(x!=" "):
            str2=str2+x
    dna6a = Seq(str1)
    dna6b = Seq(str2)
    stri=""
    for a in alignments:
        stri=stri+str(format_alignment(*a))
    params={'res':stri}
    return render(request, 'mysite/result_align.html', params)






































