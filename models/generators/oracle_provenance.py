'''
BERGEN
Copyright (c) 2024-present NAVER Corp.
CC BY-NC-SA 4.0 license
'''
from bergen.models.generators.generator import Generator


class OracleProvenance(Generator):
    """
    Generator whose generation is the set of documents
    """
    def __init__(self, 
                 model_name=None, 
                 batch_size=1,
                 **kwargs
                 ):
        Generator.__init__(self, model_name=model_name, batch_size=batch_size)

    def tokenizer(self, instr, **kwargs):
        return instr

    def format_instruction(self, sample):
        docs_prompt = ''
        for i, doc in enumerate(sample['doc']):
            docs_prompt += f"{doc} "
        return f"""{docs_prompt}"""
    
    def generate(self, inp):
        return inp
        
    def collate_fn(self, examples, **kwargs):
        q_ids = [e['q_id'] for e in examples]
        instr = [self.format_instruction(e) for e in examples]
        print(instr)
        label = [[e['label']] if isinstance(e['label'], str) else e['label'] for e in examples]
        query = [e['query'] for e in examples]
        ranking_label = [e['ranking_label'] for e in examples] if 'ranking_label' in examples[0] else [None] * len(examples)
        return {
            'model_input': instr,
            'q_id': q_ids, 
            'query': query, 
            'instruction': instr,
            'label': label, 
            'ranking_label': ranking_label,
        }
