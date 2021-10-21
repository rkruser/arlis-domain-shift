#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utils
"""

import os
import sys
import ast
import argparse
import copy

import torch


# Deprecated; now favor calling ast.literal_eval once on s where s is formatted like a python object
def parse_string(s):
    s_no_spaces = ''.join(s.split())
    arg_split = s_no_spaces.split(';')
    
    string_dict = {}
    for arg_pairs in arg_split:
        pair = arg_pairs.split(':', maxsplit=1)
        if len(pair) != 2:
            print("{0} invalid argument; ignoring".format(pair))
            continue
        
        try:
            arg_value = ast.literal_eval(pair[1])
        except ValueError:
            print("{0}:{1} is malformed (if {1} is a string, did you put quotes around it?). Interpreting as string".format(pair[0], pair[1]))
            arg_value = pair[1]
        
        string_dict[pair[0]] = arg_value
    
    return string_dict


def dict_to_neat_string(d, level=0, max_prepend_level=4):
    prepend = '  '*min(level, max_prepend_level)
    print_str = prepend+'{\n'
    for key in d:
        item = d[key]
        print_str += prepend + str(key) + ':'
        if isinstance(item, EasyDict) or isinstance(item, dict):
            print_str += '\n' + dict_to_neat_string(item, level=level+1, max_prepend_level=max_prepend_level)
        else:
            print_str += str(item) + '\n'
    print_str += prepend+'}\n'
    return print_str


# init and update could be done better here
class EasyDict:
    def __init__(self, dictionary=None, namespace=None, **kwargs):
        self.update(dictionary)
        self.update(namespace)
        self.update(kwargs)
        
    def update(self, mapping, overlapping_only = False, disjoint_only=False):
        if mapping is not None:
            if isinstance(mapping, argparse.Namespace):
                mapping = vars(mapping)
            for key in mapping:
                if overlapping_only and (key not in self):
                    continue
                if disjoint_only and (key in self):
                    continue
                value = mapping[key]
                if isinstance(value, dict) or isinstance(value, EasyDict): #recreate dicts so as not to cross pointers
                    value = self.__class__(dictionary=value)
                if isinstance(value, argparse.Namespace):
                    value = self.__class__(namespace=value)
                setattr(self, key, value)
        
    def get_dict(self):
        return vars(self)
    
    def keys(self):
        return vars(self).keys()
    
    def __len__(self):
        return len(self.keys())

    def __getitem__(self, key):
        return getattr(self, key)
    
    def get(self, key, default=None):
        if key in self:
            return self.__getitem__(key)
        else:
            return default
    
    def __setitem__(self, key, value):
        if isinstance(value, dict) or isinstance(value, EasyDict): # recreate dicts so as not to cross pointers
            value = self.__class__(dictionary=value)
        setattr(self, key, value)
            
    def __iter__(self):
        return vars(self).__iter__()
    
    def __str__(self):
        return repr(self)
    
    def __repr__(self):
        return dict_to_neat_string(self)
    
    def copy(self):
        return copy.deepcopy(self)

    def walk(self, return_dict=False, ignore_underscored_keys=True):
        walk = []
        values = []
        cur_dict = self.get_dict()
        for key in cur_dict:
            if key.startswith('_'):
                continue
            item = cur_dict[key]
            if isinstance(item, self.__class__):
                sub_walk, sub_values = item.walk(return_dict=False, ignore_underscored_keys=ignore_underscored_keys)
                for s in sub_walk:
                    walk.append(os.path.join(key,s))
                values += sub_values
            else:
                walk.append(key)
                values.append(item)
                
        if return_dict:
            output = {walk[k]:values[k] for k in range(len(walk))}
        else:
            output = (walk, values)
                
        return output
    
    def get_path(self, path):
        path = path.split('/')
        item = self
        for key in path:
            item = item.get(key)
        return item

    def set_path(self, path, value):
        path = path.split('/')
        item = self
        for key in path[:-1]:
            if key in item:
                item = item[key]
            else:
                item[key] = self.__class__()
                item = item[key]
        item[path[-1]] = value


# Is this class even a good idea? Maybe
class DataDict(EasyDict):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        if len(self) == 0:
            self._size = 0
            self._first_leaf_item = None
        else:
            _, vals = self.walk()
            self._first_leaf_item = vals[0]
            if '_size' not in self:
                self._size = len(self._first_leaf_item)

    def data_size(self):
        return self._size

    def data_type(self):
        return type(self._first_leaf_item)

        
        
"""
Method for making an easydict from a dict of paths
"""
def dict_from_paths(paths, dict_type=EasyDict):
    cur_dict = {}
    for key in paths:
        path = key.split('/', maxsplit=1)
        if len(path) > 1:          
            if path[0] not in cur_dict:
                cur_dict[path[0]] = {}
            cur_dict[path[0]][path[1]] = paths[key]
        else:
            cur_dict[path[0]] = paths[key]
    
    for key in cur_dict:
        item = cur_dict[key]
        if isinstance(item, dict):
            cur_dict[key] = dict_from_paths(item, dict_type=dict_type)
            
    return dict_type(**cur_dict)

"""
Method for concatenating multiple dicts with identical keys
"""
def collate_dicts(dict_list, dict_type = EasyDict, cast_func = None, ignore_underscored_keys=True):
    if len(dict_list) == 0:
        return dict_type()
    keys, _ = dict_list[0].walk(ignore_underscored_keys=ignore_underscored_keys)
    
    collated = {}
    for key in keys:
        collated[key] = []
    for d in dict_list:
        walk_dict = d.walk(return_dict=True, ignore_underscored_keys=ignore_underscored_keys)
        for key in keys:
            collated[key].append(walk_dict[key])
    
    if cast_func is not None:
        for key in keys:
            collated[key] = cast_func(collated[key])
            
    return dict_from_paths(collated, dict_type=dict_type)






def make_simple_dict(num):
    return EasyDict(x=torch.tensor(num), y=torch.tensor(num+10), z=EasyDict(w=torch.tensor(num+20), h=torch.tensor(num+30)))

def test_easy_dict():
    d = EasyDict()
    d.a = 1
    d.b = EasyDict()
    d.b.a = 2
    d.b.b = 3
    d.b.c = 4
    d.b.d = EasyDict()
    d.b.d.a = 5
    d.b.d.b = 6
    d.b.d.c = 7
    d.c = EasyDict()
    d.c.a = 8
    d.c.b = 9
    d.c.c = EasyDict()
    d.c.c.a = 10

    print(d)

    walk, vals = d.walk()
    print("Separate", walk, vals)
    dictwalk = d.walk(return_dict=True)
    print("Together", dictwalk)
    
    reconstruction = dict_from_paths(dictwalk)
    print(reconstruction)

    dict_list = [make_simple_dict(i) for i in range(10)]
    collated_dict = collate_dicts(dict_list, cast_func=torch.stack)
    print("list", dict_list)
    print("collated", collated_dict)







if __name__ == '__main__':
    test_easy_dict()














            
