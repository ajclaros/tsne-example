# UnarXiv is a dataset that contains 1.7 million scholarly articles from arXiv.org
# This script is used to plot the t-SNE of the arXiv dataset sections and see if there is any implicit structure
# This is a skeleton, other datasets would need documents and sections to be extracted from the source files

import jsonlines
import re
# from llama_index import Document
# from llama_index.node_parser import HierarchicalNodeParser, SimpleNodeParser
# from llama_index.node_parser import get_leaf_nodes, get_root_nodes
# from llama_index.llms import OpenAI
# from langchain.embeddings import LlamaCppEmbeddings
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
import openai
from tenacity import retry, stop_after_attempt, wait_random_exponential
import plotly.express as px
import plotly.graph_objects as go
import textwrap
os.environ["OPENAI_API_KEY"] = ''
openai.api_key = os.environ["OPENAI_API_KEY"]
model =  "text-embedding-ada-002"

def distance_cluster(df, distance):
    '''
    for every point in the dataframe, if the tsne distance to another point with the same label is less than
    the distance, then the point is in the cluster. Remove all points in the cluster from the dataframe
    and replace with the cluster centroid
    '''
    index = 0
    df_copy = df.copy()
    mask = (df['content_type'] == 'paragraph') & (df['text'].str.len() > 100)
    df_copy = df[mask].copy()
    while index < df_copy.shape[0]:
        # if index<=3:
        #     index += 1
        #     continue
        # if index % 100 == 0:
        #     print(index, end=" ")
        row = df_copy.iloc[index]
        # df_doc = df_copy[df_copy['id'] == row['id']]
        df_doc = df_copy[df_copy['id'] == row['id']].copy()
        mask = df_doc[['tsne1', 'tsne2']].apply(lambda x: np.linalg.norm(x - row[['tsne1', 'tsne2']]), axis=1) < distance
        df_doc = df_doc.loc[mask].copy()
        if df_doc.shape[0] == 1:
            index += 1
            continue
        avg = df_doc[['tsne1', 'tsne2']].mean(axis=0)
        df_copy.at[index, 'tsne1'] = avg['tsne1']
        df_copy.at[index, 'tsne2'] = avg['tsne2']
        string = df_doc['text'].str.cat( sep='|')
        # df_copy.at[index, 'text'] = string + "...(continued)"
        df_copy.at[index, 'cluster_size'] = df_doc.shape[1]
        df_copy.drop(df_doc.index[1:].tolist(), inplace=True)
        df_copy.reset_index(drop=True, inplace=True)
        index += 1
    return df_copy
# df_copy = distance_cluster(df, 0.5)

def replace_in_curly_braces(text, replacement):
    '''
    replace text in curly braces with replacement
    '''

    curly = re.findall('{{(.+?)}}', text)
    for i, c in enumerate(curly):
        text = text.replace('{{'+c+'}}', replacement+str(i))
    
    return text

def get_data(num_docs=20, start=0, **kwargs):
    ''''
    function to load documents from jsonl file
    to construct the dataframe:
    run script inside root of zip folder

    and run this script in doc/
    '''
    data = []
    print(start, num_docs)
    with jsonlines.open('arXiv_src_2212_086.jsonl') as reader:
        for i, obj in enumerate(reader):
            if i < start:
                continue
            if i == start + num_docs:
                break
            data.append(obj)
    return data


def create_df(data=None, num_docs=20, **kwargs):
    '''
    function iterates over each document object and adds the text to a dataframe
    '''
    docs = []
    df = pd.DataFrame(columns = ['id', 'section', 'sec_number', 'sec_type', 'content_type', 'text'])
    for j in range(num_docs):
        for i in range(len(data[j]['body_text'])):
            text = data[j]['body_text'][i]['text']
            df2 = pd.DataFrame([[data[j]['paper_id'], data[j]['body_text'][i]['section'], data[j]['body_text'][i]['sec_number'], data[j]['body_text'][i]['sec_type'], data[j]['body_text'][i]['content_type'],
                                 text]], columns = ['id', 'section', 'sec_number', 'sec_type', 'content_type', 'text'])
            df = pd.concat([df, df2])
    df.reset_index(drop=True, inplace=True)
    return df

def clean_text(df):
    df['text'].apply(lambda x: replace_in_curly_braces(x, "formula"))
    return df


@retry(stop=stop_after_attempt(3), wait=wait_random_exponential(min=1, max=10))
def get_embedding(text, model):
    '''
    retrieve the embedding for a given text and model
    '''
    embedding = openai.Embedding.create(input=[text], model=model)['data'][0]['embedding']
    return embedding

def add_embedding_col(df):
    ''''
    adds embedding column to dataframe
    '''
    embedds = []
    for i, row in enumerate(df.iterrows()):
        print("Getting embedding for row: ", i, " out of ", len(df))
        embed = get_embedding(row[1]['text'], model)
        embedds.append(embed)
    df['embedding'] = embedds
    return df

add_docs = {'val':False, 'num_docs': 20, 'start': 40}
reembed = False
plot = True
df = pd.DataFrame()
df_tsne = pd.DataFrame()
if not os.path.exists('arxiv_df.pkl'): # if file not exist construct from dataset
    data = get_data(20)
    df = create_df(data, 20)
    df.reset_index(drop=True, inplace=True)
    df = add_embedding_col(df)
    df.to_pickle('arxiv_df40.pkl')
elif add_docs['val']: # if file exists and add_docs is true, add docs to existing dataframe
    print("Adding docs to existing dataframe")
    data = get_data(**add_docs)
    df = create_df(data=data, **add_docs)
    df.reset_index(drop=True, inplace=True)
    df = add_embedding_col(df)
    df1 = pd.read_pickle('arxiv_df.pkl')
    df = pd.concat([df1, df])
    df.reset_index(drop=True, inplace=True)
    df.to_pickle('arxiv_df.pkl')
    print("Done adding docs to existing dataframe")
else:
    print("Loading pickle file")
    if reembed:
        df = pd.read_pickle('arxiv_df.pkl')
        df = add_embedding_col(df)
        df.to_pickle('arxiv_df.pkl')
    else:
        df = pd.read_pickle('arxiv_df.pkl')
    if plot:
        df = pd.read_pickle('arxiv_df.pkl')
        X = np.array(df['embedding'].tolist(), dtype=np.float32)
        tsne = TSNE(n_components=2, perplexity=40, n_iter=300)
        tsne_results = tsne.fit_transform(X)
        df_tsne = pd.DataFrame(tsne_results, columns=['tsne1', 'tsne2'])
        df_tsne['paper_id'] = df['id']
        df['tsne1'] = df_tsne['tsne1']
        df['tsne2'] = df_tsne['tsne2']

        df['cluster_size'] = 1
        df_copy = distance_cluster(df, 0.2)
        df_copy['text_wrap'] = df_copy['text'].apply(lambda t: "<br>".join(textwrap.wrap(t)))
        fig = px.scatter(df_copy, x='tsne1', y='tsne2', color='id', hover_data=['text_wrap', 'cluster_size'], render_mode='svg')
        fig.update_traces(marker_size=10)
        fig.show()

# if __name__ == '__main__':
#     main()


text = df.iloc[0].text
tt = replace_in_curly_braces(text, "formula")
print(tt)

df['text']=df['text'].apply(lambda x: replace_in_curly_braces(x, "formula"))
