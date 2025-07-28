'''
BERGEN
Copyright (c) 2024-present NAVER Corp.
CC BY-NC-SA 4.0 license
'''
import os

import numpy as np
import torch
import sys

from bergen.models.retrievers.retriever import Retriever
from transformers import AutoModelForMaskedLM, AutoTokenizer
from seismic import SeismicIndex

def get_argv(argv):
    return dict(arg.split("=") for arg in sys.argv[1:] if "=" in arg).get(argv)

class Splade(Retriever):
    def __init__(self, model_name, max_len=512, query_encoder_name=None):
        self.model_name = model_name
        self.max_len = max_len
        self.model = AutoModelForMaskedLM.from_pretrained(self.model_name, low_cpu_mem_usage=True, torch_dtype=torch.float16)
        if query_encoder_name:
            self.query_encoder = AutoModelForMaskedLM.from_pretrained(query_encoder_name, torch_dtype=torch.float16, low_cpu_mem_usage=True)
        else:
            self.query_encoder = self.model  # otherwise symmetric
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, max_length=self.max_len)
        self.reverse_vocab = {v: k for k, v in self.tokenizer.vocab.items()}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.eval()
        if query_encoder_name:
            self.query_encoder = self.query_encoder.to(self.device)
            self.query_encoder.eval()
        if torch.cuda.device_count() > 1 and torch.cuda.is_available():
            self.model = torch.nn.DataParallel(self.model)
            if query_encoder_name:
                self.query_encoder = torch.nn.DataParallel(self.query_encoder)
        
    def __call__(self, query_or_doc, kwargs):
        kwargs = {key: value.to('cuda') for key, value in kwargs.items()}
        outputs = self.model(**kwargs).logits

        # pooling over hidden representations
        emb, _ = torch.max(torch.log(1 + torch.relu(outputs)) * kwargs["attention_mask"].unsqueeze(-1), dim=1)

        return {
                "embedding": emb
            }

    def collate_fn(self, batch, query_or_doc=None):
        key = 'generated_query' if query_or_doc == "query" else "content"
        content = [sample[key] for sample in batch]
        return_dict = self.tokenizer(content, padding=True, truncation=True, max_length= self.max_len, return_tensors='pt')
        return return_dict

    def similarity_fn(self, query_embds, doc_embds):
        return torch.sparse.mm(query_embds.to_sparse(), doc_embds.t()).to_dense()


class SeismicSplade(Splade):
    def __init__(self, model_name,  max_len=512, query_encoder_name=None):
        super().__init__(model_name, max_len=max_len, query_encoder_name=query_encoder_name)
        self.idx = None
        self.dataset = get_argv('dataset')
        self.jsonl = []
        self.id = 0

    def __call__(self, query_or_doc, kwargs):
        json = super().__call__(query_or_doc, kwargs)
        # Ignore if query
        if query_or_doc == 'query':
            return json

        # Check if index is already built
        if os.path.exists(f'indexes/splade-{self.dataset}-seismic.p'):
            return json

        def clear(s):
            import re
            return re.sub(r'[^a-zA-Z0-9\s]', '', s)

        tokens = kwargs['input_ids']
        embeddings_list = json["embedding"]

        sparse_embeddings = []
        # convert embedding into dict format
        for emb in embeddings_list:
            emb = emb.squeeze()
            nonzero_indices = torch.nonzero(emb > 1e-6, as_tuple=False).squeeze()
            doc_dict = {
                clear(self.tokenizer.convert_ids_to_tokens(idx.item())): float(emb[idx].item())
                for idx in nonzero_indices
            }
            sparse_embeddings.append(doc_dict)

        # convert document tokens to document text
        contents = self.tokenizer.batch_decode(tokens, skip_special_tokens=True)

        # Append document batch to existing converted dataset
        for i in range(len(tokens)):
            self.jsonl.append({
                "id": self.id,
                "content": clear(contents[i]),
                "vector": sparse_embeddings[i]
            })
            self.id += 1

        return json

    # Requires to return torch.tensor q*d | t[q,d]=score.
    # Inefficient if using index => Result like: 0...0, 4.3, 0...0. Full scan to find the good docs
    # Change internally?
    def similarity_fn(self, query_embds, doc_embds):
        print(self.jsonl[:3])

        def clear(s):
            import re
            return re.sub(r'[^a-zA-Z0-9\s]', '', s)

        # if index doesn't exists
        if not os.path.exists(f'indexes/splade-{self.dataset}-seismic.index.seismic'):
            # create jsonl with documents
            with open("tmp.jsonl", 'w+') as f:
                for row in self.jsonl:
                    f.write(str(row).replace("'", '"') + "\n")
            # let seismic build the index
            index = SeismicIndex.build("tmp.jsonl")
            # save the index persistently
            index.save(f'indexes/splade-{self.dataset}-seismic')

        # Load the index
        if self.idx is None:
            self.idx = SeismicIndex.load(f'indexes/splade-{self.dataset}-seismic.index.seismic')

        # for all queries convert to seismic format and search for documents
        MAX_TOKEN_LEN = 30
        string_type = f'U{MAX_TOKEN_LEN}'
        id = 0

        all_results = []
        for emb in query_embds:
            emb = emb.squeeze()
            nonzero_indices = torch.nonzero(emb > 1e-6, as_tuple=False).squeeze()
            query = {
                clear(self.tokenizer.convert_ids_to_tokens(idx.item())): float(emb[idx].item())
                for idx in nonzero_indices
            }
            keys = np.array(list(query.keys()), dtype=string_type)
            values = np.array(list(query.values()), dtype=np.float32)

            results = self.idx.__call__(query_id=str(id), query_components=keys,
                                        query_values=values, k=10, query_cut=3, heap_factor=0.8)
            all_results.append(results)
            id += 1

        print(all_results)

        num_queries = query_embds.size(0)
        num_documents = doc_embds.size(0)

        score_matrix = torch.zeros((num_queries, num_documents), device=query_embds.device, dtype=query_embds.dtype)

        for results in all_results:
            for result in results:
                score_matrix[int(result[0]), int(result[2])] = result[1]

        return score_matrix
