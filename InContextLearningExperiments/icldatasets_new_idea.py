import random
import torch
from torch.utils.data import Dataset



class KVRetrievalDatasetFixedDict(Dataset):
    """ 
    Dataset for InContext Learning. 
    """

    def __init__(self, split, length=6, vocab_size=10, additional_vocab = 2, dictionary=None):
        assert split in {'train', 'test'}
        self.split = split
        self.length = length
        self.vocab_size = vocab_size + additional_vocab
        self.true_vocab_size = vocab_size
        self.dictionary = dictionary if dictionary is not None else self._generate_dictionary()
          
    def _generate_dictionary(self):
        return torch.randperm(self.true_vocab_size)

    def __len__(self):
        return 10000 # ...
    
    def get_vocab_size(self):
        return self.vocab_size
    
    def get_block_size(self):
        # the length of the sequence that will feed into transformer, 
        # containing concatenated input and the output, but -1 because
        # the transformer starts making predictions at the last input element
        return 2*self.length + 1
    
    def __getitem__(self, idx):
        dictionary = self.dictionary
        context_keys = torch.randint(self.true_vocab_size, size=(self.length,), dtype=torch.long)
        context_labels = torch.index_select(dictionary, 0, context_keys)
        # find a key thats appeared in this context
        question_key = torch.index_select(context_keys, 0, torch.randint(self.length, size=(1,), dtype=torch.long))
        question_label = torch.index_select(dictionary, 0, question_key)
        
        # make final context
        context_keys = torch.concat([context_keys, question_key], dim=0)
        context_labels = torch.concat([context_labels, question_label], dim=0)
        #interleave them
        full_context = torch.flatten(torch.stack([context_keys, context_labels]).t())

        # provide everything but the last label to TX
        x = full_context[:-1].clone()
        # Mask everything but the last label in the loss
        y = full_context[1:].clone()
        y[:2*self.length] = -1
        return x, y

    # Old stuff when no guarantee of choosing something from context 
    # def __getitem__(self, idx):
    #     dictionary = self.dictionary
    #     context_keys = torch.randint(self.true_vocab_size, size=(self.length+1,), dtype=torch.long)
    #     context_labels = torch.index_select(dictionary, 0, context_keys)
    #     #interleave them
    #     full_context = torch.flatten(torch.stack([context_keys, context_labels]).t())
    #     # provide everything but the last label to TX
    #     x = full_context[:-1].clone()
    #     # Mask everything but the last label in the loss
    #     y = full_context[1:].clone()
    #     y[:2*self.length] = -1
    #     return x, y
    

# Generating mixed datasets

class KVRetrievalDatasetMixedDict(Dataset):
    """ 
    Dataset for InContext Learning. 
    """

    def __init__(self, split, length=6, vocab_size=10, additional_vocab = 2, dictionary=None, mixing_fraction=0.0):
        assert split in {'train', 'test'}
        self.split = split
        self.length = length
        self.vocab_size = vocab_size + additional_vocab
        self.true_vocab_size = vocab_size
        self.mixing_fraction = mixing_fraction
        self.dictionary = dictionary if dictionary is not None else self._generate_dictionary()
          
    def _generate_dictionary(self):
        return torch.randperm(self.true_vocab_size)

    def __len__(self):
        return 10000 # ...
    
    def get_vocab_size(self):
        return self.vocab_size
    
    def get_block_size(self):
        # the length of the sequence that will feed into transformer, 
        # containing concatenated input and the output, but -1 because
        # the transformer starts making predictions at the last input element
        return 2*self.length + 1

    def __getitem__(self, idx):
        dictionary = self.dictionary if torch.rand(1) < self.mixing_fraction else self._generate_dictionary()
        context_keys = torch.randint(self.true_vocab_size, size=(self.length,), dtype=torch.long)
        context_labels = torch.index_select(dictionary, 0, context_keys)
        # find a key thats appeared in this context
        question_key = torch.index_select(context_keys, 0, torch.randint(self.length, size=(1,), dtype=torch.long))
        question_label = torch.index_select(dictionary, 0, question_key)
        # make final context
        context_keys = torch.concat([context_keys, question_key], dim=0)
        context_labels = torch.concat([context_labels, question_label], dim=0)
        
        #interleave them
        full_context = torch.flatten(torch.stack([context_keys, context_labels]).t())
        # provide everything but the last label to TX
        x = full_context[:-1].clone()
        # Mask everything but the last label in the loss
        y = full_context[1:].clone()
        y[:2*self.length] = -1
        return x, y
    
class KVRetrievalDatasetMixedDictSeparatedVocab(Dataset):
    """ 
    Dataset for InContext Learning. 
    """

    def __init__(self, split, length=6, vocab_size=10, additional_vocab = 2, dictionary=None, mixing_fraction=0.0):
        assert split in {'train', 'test'}
        self.split = split
        self.length = length
        self.vocab_size = 2*vocab_size + additional_vocab
        self.true_vocab_size = vocab_size
        self.mixing_fraction = mixing_fraction
        self.dictionary = dictionary if dictionary is not None else self._generate_dictionary()
          
    def _generate_dictionary(self):
        return torch.randperm(self.true_vocab_size)

    def _generate_extended_dictionary(self):
        return torch.randperm(self.true_vocab_size) + self.true_vocab_size

    def __len__(self):
        return 10000 # ...
    
    def get_vocab_size(self):
        return self.vocab_size
    
    def get_block_size(self):
        # the length of the sequence that will feed into transformer, 
        # containing concatenated input and the output, but -1 because
        # the transformer starts making predictions at the last input element
        return 2*self.length + 1

    def __getitem__(self, idx):
        decider = torch.rand(1) < 0.5
        dictionary = self.dictionary if decider else self._generate_extended_dictionary()
        context_keys = torch.randint(self.true_vocab_size, size=(self.length,), dtype=torch.long)
        context_labels = torch.index_select(dictionary, 0, context_keys)
        # find a key thats appeared in this context
        question_key = torch.index_select(context_keys, 0, torch.randint(self.length, size=(1,), dtype=torch.long))
        question_label = torch.index_select(dictionary, 0, question_key)
        # make final context
        context_keys = torch.concat([context_keys, question_key], dim=0) if decider else torch.concat([context_keys + self.true_vocab_size, question_key], dim=0)
        context_labels = torch.concat([context_labels, question_label], dim=0)


        
        #interleave them
        full_context = torch.flatten(torch.stack([context_keys, context_labels]).t())
        # provide everything but the last label to TX
        x = full_context[:-1].clone()
        # Mask everything but the last label in the loss
        y = full_context[1:].clone()
        y[:2*self.length] = -1
        return x, y
    

class KVRetrievalDatasetFixedDictNewIdea(Dataset):
    """ 
    Dataset for InContext Learning. 
    """

    def __init__(self, split, vocab_size=10, length=None, dictionary=None, mixing_fraction=0.9, perm_or_random='perm'):
        assert split in {'train', 'test'}
        self.split = split
        self.length = length or vocab_size
        self.vocab_size = vocab_size
        if perm_or_random == 'perm':
            if self.vocab_size < self.length:
                raise ValueError('vocab_size must be greater than length')
        self.perm_or_random = perm_or_random
        self.dictionary = dictionary if dictionary is not None else self._generate_dictionary()
        self.mixing_fraction = mixing_fraction
          
    def _generate_dictionary(self):
        return torch.randperm(self.vocab_size)+self.vocab_size

    def __len__(self):
        return 10000 # ...
    
    def get_vocab_size(self):
        return self.vocab_size
    
    def get_block_size(self):
        # the length of the sequence that will feed into transformer, 
        # containing concatenated input and the output, but -1 because
        # the transformer starts making predictions at the last input element
        return 2*self.length-1

    def __getitem__(self, idx):
        dictionary = self.dictionary
        if self.perm_or_random == 'perm':
            context_keys = torch.randperm(self.vocab_size)
        else:
            context_keys = torch.randint(self.vocab_size, size=(self.length,), dtype=torch.long)

        context_labels = torch.index_select(dictionary, 0, context_keys)
        full_context = torch.flatten(torch.stack([context_keys, context_labels]).t())

        # provide everything but the last label to TX
        x = full_context[:2*self.length-1].clone()
        # Mask everything but the last label in the loss
        y = full_context[1:2*self.length].clone()
        #y[:2*self.length] = -1
        return x, y


def non_neighbour_dict(dd, base_dict):
    return ((dd-base_dict) == 0).any() | ((torch.roll(dd, 1)-base_dict) == 0).any() | ((torch.roll(dd, -1)-base_dict) == 0).any()
    
class KVRetrievalDatasetMixedDictNewIdea(Dataset):
    """ 
    Dataset for InContext Learning. 
    """

    def __init__(self, split, vocab_size=10, length=None, dictionary=None, mixing_fraction=0.9, perm_or_random='perm'):
        assert split in {'train', 'test'}
        self.split = split
        self.length = length or vocab_size
        self.vocab_size = vocab_size
        if perm_or_random == 'perm':
            if self.vocab_size < self.length:
                raise ValueError('vocab_size must be greater than length')
        self.perm_or_random = perm_or_random
        self.dictionary = dictionary if dictionary is not None else self._generate_dictionary()
        self.mixing_fraction = mixing_fraction
          
    def _generate_dictionary(self):
        dd = torch.randperm(self.vocab_size)
        base_dict = torch.arange(0, self.vocab_size, 1)
        while non_neighbour_dict(dd, base_dict):
            dd = torch.randperm(self.vocab_size)
        return torch.randperm(self.vocab_size)+self.vocab_size

    def __len__(self):
        return 10000 # ...
    
    def get_vocab_size(self):
        return self.vocab_size
    
    def get_block_size(self):
        # the length of the sequence that will feed into transformer, 
        # containing concatenated input and the output, but -1 because
        # the transformer starts making predictions at the last input element
        return 2*self.length-1

    def __getitem__(self, idx):
        dictionary = self.dictionary if torch.rand(1) < self.mixing_fraction else self._generate_dictionary()
        if self.perm_or_random == 'perm':
            context_keys = torch.randperm(self.vocab_size)
        else:
            context_keys = torch.randint(self.vocab_size, size=(self.length,), dtype=torch.long)

        context_labels = torch.index_select(dictionary, 0, context_keys)
        full_context = torch.flatten(torch.stack([context_keys, context_labels]).t())

        # provide everything but the last label to TX
        x = full_context[:2*self.length-1].clone()
        # Mask everything but the last label in the loss
        y = full_context[1:2*self.length].clone()
        #y[:2*self.length] = -1
        return x, y
    

class KVRetrievalDatasetChangingDictNewIdea(Dataset):
    """ 
    Dataset for InContext Learning. 
    """

    def __init__(self, split, vocab_size=10, length=None, perm_or_random='perm'):
        assert split in {'train', 'test'}
        self.split = split
        self.length = length or vocab_size
        self.vocab_size = vocab_size
        if perm_or_random == 'perm':
            if self.vocab_size < self.length:
                raise ValueError('vocab_size must be greater than length')
        self.perm_or_random = perm_or_random
          
    def _generate_dictionary(self):
        return torch.randperm(self.vocab_size)+self.vocab_size

    def __len__(self):
        return 10000 # ...
    
    def get_vocab_size(self):
        return self.vocab_size
    
    def get_block_size(self):
        # the length of the sequence that will feed into transformer, 
        # containing concatenated input and the output, but -1 because
        # the transformer starts making predictions at the last input element
        return 2*self.length-1

    def __getitem__(self, idx):
        dictionary = self._generate_dictionary()
        if self.perm_or_random == 'perm':
            context_keys = torch.randperm(self.vocab_size)
        else:
            context_keys = torch.randint(self.vocab_size, size=(self.length,), dtype=torch.long)

        context_labels = torch.index_select(dictionary, 0, context_keys)
        full_context = torch.flatten(torch.stack([context_keys, context_labels]).t())

        # provide everything but the last label to TX
        x = full_context[:2*self.length-1].clone()
        # Mask everything but the last label in the loss
        y = full_context[1:2*self.length].clone()
        #y[:2*self.length] = -1
        return x, y