import ollama
from math import sqrt

EMBED_MODEL = "bge-m3"

sentences = [
    "本サービスは先進医療の情報を提供します。",       # A
    "このウェブサイトでは最新の医療技術を案内します。", # B - similar meaning
    "明日の東京の天気は雨でしょう。",                 # C - totally different
    "当サイトでは先端医療に関するをデータご提供しています。" #D
]


def embed(text):
    return ollama.embeddings(model=EMBED_MODEL, prompt=text)["embedding"]


def cosine(v1, v2):
    dot = sum(a * b for a, b in zip(v1, v2))
    n1 = sqrt(sum(a * a for a in v1))
    n2 = sqrt(sum(b * b for b in v2))
    return dot / (n1 * n2)

vecs = [embed(s) for s in sentences]

print(f"A vs B (similar topic):     {cosine(vecs[0], vecs[1]):.4f}")
print(f"A vs C (different topic):   {cosine(vecs[0], vecs[2]):.4f}")
print(f"B vs C (different topic):   {cosine(vecs[1], vecs[2]):.4f}")

print(f"A vs D (different topic):   {cosine(vecs[0], vecs[3]):.4f}")
print(f"B vs D (different language): {cosine(vecs[1], vecs[3]):.4f}")
print(f"C vs D (different topic):   {cosine(vecs[2], vecs[3]):.4f}")