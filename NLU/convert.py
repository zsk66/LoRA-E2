import tensorflow as tf
import tensorflow_datasets as tfds

dataset_prefixes = {
    'cola': "classify the grammaticality of the text:",
    'sst2': "sentiment classification:",
    'mrpc': "classify the semantic similarity of the text:",
    'qqp': "classify the semantic similarity of the text:",
    'mnli': "classify the semantic similarity of the text:",
    'qnli': "classify the semantic similarity of the question and the sentence:",
    'rte': "classify the semantic similarity of the text:",
    'wnli': "classify the semantic similarity of the text:",
    'stsb': "predict the semantic similarity score between the texts:"
}

def glue(x, benchmark_name, label_names, feature_names=None, id_key='idx'):
    prefix = dataset_prefixes.get(benchmark_name, benchmark_name)

    if benchmark_name == 'stsb':
        strs_to_join = [
            prefix, 'sentence1:', x['sentence1'], 'sentence2:', x['sentence2']
        ]
        label_string = tf.as_string(tf.round(x['label'] * 5) / 5, precision=1)
        joined = tf.strings.join(strs_to_join, separator=' ')
        return {'inputs': joined.numpy().decode('utf-8'), 'targets': label_string.numpy().decode('utf-8'), 'idx': x['idx']}
    if benchmark_name == 'qqp':
        strs_to_join = [
            prefix, 'question1:', x['question1'], 'question2:', x['question2']
        ]
        label_string = tf.as_string(x['label'])
        if label_string == '1':
            label_string = tf.constant('duplicate')
        elif label_string == '0':
            label_string = tf.constant('not_duplicate')
        joined = tf.strings.join(strs_to_join, separator=' ')
        return {'inputs': joined.numpy().decode('utf-8'), 'targets': label_string.numpy().decode('utf-8'), 'idx': x['idx']}
    else:
        feature_keys = (
            feature_names or sorted(set(x.keys()).difference(['label', 'idx', 'id'])))
        strs_to_join = []
        for key in feature_keys:
            strs_to_join.append(x[key])
        strs_to_join.insert(0, prefix)

        label_name = tf.cond(
            tf.equal(x['label'], -1),
            lambda: tf.constant('<unk>'),
            lambda: tf.gather(label_names, x['label']),
        )
        
        joined = tf.strings.join(strs_to_join, separator=' ')

        ex = {}

        if benchmark_name == 'multirc':
            joined = tf.strings.regex_replace(joined, '<br>', ' ')
            joined = tf.strings.regex_replace(joined, '<(/)?b>', '')
            ex['idx/paragraph'] = x['idx']['paragraph']
            ex['idx/question'] = x['idx']['question']
            ex['idx/answer'] = x['idx']['answer']
        else:
            if id_key:
                ex['idx'] = x[id_key]


        ex['inputs'] = joined.numpy().decode('utf-8')
        ex['targets'] = label_name.numpy().decode('utf-8')

        return ex




datasets_set = ['cola', 'sst2', 'mrpc', 'qqp', 'stsb', 'mnli','mnli_mismatched', 'mnli_matched', 'qnli', 'rte', 'wnli', 'ax']
test_data = {
    'cola':{
            "sentence": "Our friends won't buy this analysis, let alone the next one we propose.",
            "label": 1,
            "idx": 0
    },
    'sst2':{
            "sentence": "hide new secretions from the parental units",
            "label": 0,
            "idx": 0
    },
    'mrpc':{
            "sentence1": "Amrozi accused his brother, whom he called \"the witness\", of deliberately distorting his evidence.",
            "sentence2": "Referring to him as only \"the witness\", Amrozi accused his brother of deliberately distorting his evidence.",
            "label": 1,
            "idx": 0
    },
    'qqp':{
            "question1": "How is the life of a math student? Could you describe your own experiences?",
            "question2": "Which level of prepration is enough for the exam jlpt5?",
            "label": 0,
            "idx": 0
    },
    'stsb':{
            "sentence1": "A plane is taking off.",
            "sentence2": "An air plane is taking off.",
            "label": 5.0,
            "idx": 0
    },
    'mnli':{
            "premise": "Conceptually cream skimming has two basic dimensions - product and geography.",
            "hypothesis": "Product and geography are what make cream skimming work.",
            "label": 1,
            "idx": 0
    },
    'mnli_mismatched':{
            "premise": "Conceptually cream skimming has two basic dimensions - product and geography.",
            "hypothesis": "Product and geography are what make cream skimming work.",
            "label": 1,
            "idx": 0
    },
    'mnli_matched':{
            "premise": "Conceptually cream skimming has two basic dimensions - product and geography.",
            "hypothesis": "Product and geography are what make cream skimming work.",
            "label": 1,
            "idx": 0
    },
    'qnli':{
            "question": "When did the third Digimon series begin?",
            "sentence": "Unlike the two seasons before it and most of the seasons that followed, Digimon Tamers takes a darker and more realistic approach to its story featuring Digimon who do not reincarnate after their deaths and more complex character development in the original Japanese.",
            "label": 1,
            "idx": 0
    },
    'rte':{
            "sentence1": "No Weapons of Mass Destruction Found in Iraq Yet.",
            "sentence2": "Weapons of Mass Destruction Found in Iraq.",
            "label": 1,
            "idx": 0
    },
    'wnli':{
            "sentence1": "I stuck a pin through a carrot. When I pulled the pin out, it had a hole.",
            "sentence2": "The carrot had a hole.",
            "label": 1,
            "idx": 0
    },
    'ax':{
            "premise": "The cat sat on the mat.",
            "hypothesis": "The cat did not sit on the mat.",
            "label": -1,
            "idx": 0
    }
}
def find_ds_config(name):
     if 'mnli' in name or name == 'ax':
         name = 'mnli'
     for b in tfds.text.glue.Glue.builder_configs.values():
        if b.name == name:
            benchmark_config = b
            return benchmark_config
