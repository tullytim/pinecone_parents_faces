using PyCall
using JSON3
using Pinecone
using YAML

const OUTPUT_DIR = "/output/"
const RAW_DIR = "/raw/"

py"""
# pylint: disable=missing-module-docstring
# pylint: disable=missing-class-docstring
# pylint: disable=missing-function-docstring
import glob
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from deepface import DeepFace
from julia import Main
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from tqdm import tqdm

def do_deep_face(inputfile, model):
    try:
        return DeepFace.represent(img_path=inputfile, model_name=model)
    except:
        print("No face detected on ", inputfile)

def gen_vectors(name, model):
    files = glob.glob(name + Main.RAW_DIR + "/*jpg")
    for img_file in files:
        fname = os.path.basename(img_file)
        outfilename = f"{name}{Main.OUTPUT_DIR}{fname}.vec"
        vec = do_deep_face(img_file, model)
        if vec is not None and len(vec) > 0:
            np.savetxt(outfilename, [vec], delimiter=',')

def gen_tsne_df(person, perplexity):
    vectors =[]
    vectorfiles = sorted(glob.glob(f"{person}/output/*.vec"))
    for v in vectorfiles:
        with open(v, 'r', encoding='UTF-8') as vfile:
            f_list = [float(i) for line in vfile for i in line.rstrip().split(',')]
            vectors.append(f_list)
    pca = PCA(n_components=20)
    tsne = TSNE(2, perplexity=perplexity, random_state = 0, n_iter=10000,
        verbose=0, metric='euclidean', learning_rate=75)
    pca50transform = pca.fit_transform(vectors)
    embeddings2d = tsne.fit_transform(pca50transform)
    return pd.DataFrame({'x':embeddings2d[:,0], 'y':embeddings2d[:,1]})

def plot_tsne(perplexity, model):
    (_, ax) = plt.subplots(figsize=(8,5))
    plt.style.use('seaborn-whitegrid')
    plt.grid(color='#EAEAEB', linewidth=0.5)
    ax.spines['top'].set_color(None)
    ax.spines['right'].set_color(None)
    ax.spines['left'].set_color('#2B2F30')
    ax.spines['bottom'].set_color('#2B2F30')
    colormap = {'jess':'#4c93db', 'tim':'#ee8933', 'jackson':'#4fad5b', 'jameson':'#ea3323'}
    for person in tqdm(colormap):
        embeddingsdf = gen_tsne_df(person, perplexity)
        ax.scatter(embeddingsdf.x, embeddingsdf.y, alpha=.5, label=person, color=colormap[person])
    plt.title(f'Scatter plot of faces using {model}', fontsize=16, fontweight='bold', pad=20)
    plt.suptitle(f't-SNE [perplexity={perplexity}]', y=0.92, fontsize=13)
    plt.legend(loc='best', frameon=True)
    plt.show()
    plt.savefig(f'plots/{model}/plot_perp_{perplexity}.png')

"""
################## Julia Begin ################
const VECTOR_KEY_FIRST_NAME = "first_name"
function readVectorFromFile(datafile::String)
   line = readline(datafile)
   vecs = split(line, ",")
   map(x->parse(Float64, x), vecs)
end

function uploadVectorsPinecone(person::String)
   foreach(readdir(person * OUTPUT_DIR)) do vectorfile
      println("Upsert vectors for: $vectorfile")
      vecoffloats = readVectorFromFile(person * OUTPUT_DIR * vectorfile)
      rv = Pinecone.upsert(context, index, [vectorfile], [vecoffloats], 
         [Dict{String, Any}(VECTOR_KEY_FIRST_NAME=>person)])
      println(rv)
   end
end

function testSimiliarity(person::String, child::String)
   kidsmeta = Dict{String, Any}(VECTOR_KEY_FIRST_NAME=>child)
   res = []
   queryvecs = Vector{PineconeVector}()
   foreach(readdir(person * OUTPUT_DIR)) do vectorfile
      vecoffloats = readVectorFromFile(person * OUTPUT_DIR * vectorfile)
      push!(queryvecs, PineconeVector(vectorfile, vecoffloats, Dict{String,Any}()))
   end
   rv = Pinecone.query(context, index, queryvecs, 20, "", false, false, kidsmeta)
   obj = JSON3.read(rv)
   for i in 1:length(obj.results)
      res = vcat(res, obj.results[i].matches)
   end

   sort!(res, by = x->x.score) #lower (distance) is better
   println("\n\nResults for $person and $child")
   println("----------------------------------------")
   println(res[1:2])
end

function extractFacesAndVectorize(model)
   for person in ["jess", "jackson", "jameson", "tim"]
      py"gen_vectors"(person, model)
      uploadVectorsPinecone(person)
   end
   println("Done generating vectors with ", model)
end

function distanceTest()
   testSimiliarity("jess", "jackson")
   testSimiliarity("tim", "jackson")
   testSimiliarity("jess", "jameson")
   testSimiliarity("tim", "jameson")
   testSimiliarity("tim", "jess")
end

function runPlots()
   for perplexity in 2:50
      py"plot_tsne"(perplexity, MODEL)
   end
end

############# Main ##############
MODEL = "Facenet"

config = YAML.load_file("config.yml")
context = Pinecone.init(config["pinecone_key"], config["region"])
index = Pinecone.Index("kids-facenet");

extractFacesAndVectorize(MODEL)
distanceTest()
runPlots()