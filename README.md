# Pinecone Facial Similarity for Parents
This is the source for the Medium post I wrote that uses Julia with Pinecone to look for facial similarity with my wife and I against our young kids. It is written in Julia but has some Python usage using PyCall within.

# Usage
    python3 ./runfaces.jl

The script is hardcoded w/ the names of the candidate faces being evaluated, but is very easy to change in the code. Within each person (candidate) directory there is a subdir structure expected of {person}/raw and {person}/output which holds the raw image and output vectors, respectively.  The script expects a local file named config.yml and is not taken as a commandline argument.  The format of the config file is trival and show below.

# Example config.yml
```
---
pinecone_key: '123456-****-abcdefg-******'
region: 'us-west1-gcp'
 ```
