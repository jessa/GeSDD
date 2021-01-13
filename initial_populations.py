import re
import numpy as np
import sys
import argparse

from pysdd.sdd import Vtree, SddManager, SddNode

from logic.conj import conj
from logic.disj import disj
from logic.equiv import equiv
from logic.lit import lit
from logic.neg import neg

from model.indicator_manager import indicator_manager
from model.model import Model
from model.count_manager import count_manager

from optimizer.multiprog_model_fitter import fit_population_multitprog

from model_IO import model_io

from datio import IO_manager


class LearnSddFeatureParser:
    
    def __init__(self):
        self._p_literal = re.compile(r'^(-?\d+)')
        self._p_head = re.compile(r'^([^\[]*)')

        self._heads = {
            "conj": conj,
            "disj": disj,
        #     "mutex": 
        #     "exact-1":
        }
               
   
    def parse(self, model_dir):
        features_dict = self.readFeatures(f'{model_dir}/features.txt')
        weights = np.loadtxt(f'{model_dir}/weights.csv')
        factors = self.formulasAndWeights2factors(features_dict, weights)
        return list(factors), features_dict

    def formulasAndWeights2factors(self, features_dict, weights):
        for indicator, weight in enumerate(weights[1:],1):
            factor = features_dict[indicator]
            encoding = equiv([factor, lit(indicator)])
            yield factor, weight, encoding, indicator
            
    def readFeatures(self, path):
        formulas = {}
        for line in open(path):
            literal, formula =  self.parseLearnSddToLogic(line, formulas)
            formulas[literal] = formula

        return formulas

    def parseLearnSddToLogic(self, learnsdd_notation, flatten=None):
        if flatten is None:
            flatten = {}
        literal, formula = learnsdd_notation.split(':')
        literal = int(literal)
        formula = formula.strip()

        if formula.startswith('logic.Literal'):
            formula_logic = lit(literal)

        else:
            formula_logic = self._parseFormula(formula, flatten)

        return literal, formula_logic   
        
    def _parseFormula(self, formula, flatten=None):
        if flatten is None:
            flatten = {}
        logic,rest = self._forwardParseFormula(formula, flatten)
        assert not rest, f"rest: {rest}"
        return logic

    def _forwardParseFormula(self, formula, flatten):
        # try to parse it as an integer:
        found, rest = self._match_and_rest(self._p_literal, formula)
        if found is not None:
            integer = int(found)
            if abs(integer) in flatten:
                lit_form = flatten[abs(integer)]
                if integer<0:
                    lit_form = neg(lit_form)
                return lit_form, rest
            else:
                return lit(integer)

        else:
            head, rest = self._match_and_rest(self._p_head, formula)
            assert head in self._heads, head
            rest = rest[1:] #remove leading bracket
            els = []
            while rest[0]!=']':
                el, rest = self._forwardParseFormula(rest, flatten)
                els.append(el)
                rest = rest.lstrip(',')
            rest = rest[1:]# remove closing bracket

            return self._heads[head](els), rest
      
    def _match_and_rest(self, pattern, string):
        m = re.search(pattern, string)
        if m is None:
            return None, string
        else:
            found = m.group()
            rest = string[len(found):]
            return found, rest

            
def claim_indicator_for_factor(indicator_manager, factor, indicator):
    assert factor is not None
    
    # either the factor is already cached by the indicator manager
    if indicator_manager.factor_cached(factor):
        # in this case, it should be linked to the correct indicator
        assert indicator_manager.indicator_of_factor(factor)==indicator
        # and we only need to increment the count
        indicator_manager.increment_variable(indicator)
        
    # or it is not cached yet
    else:
        # in that case, the count of the indicator of choice should 0
        assert indicator_manager.availability[indicator]==0
        # We should claim that indicator (which increments its count)
        indicator_manager.claim_variable(indicator)
        # and link this factor in the cache,
        indicator_manager.add_factor_cache(factor, indicator)
        
    return indicator





def sample_initial_population_from_learnSdd_model(sample_prob, population_size, max_nb_f, in_dir, train_path, valid_path, test_path, out_dir):
    
    all_factors, features = LearnSddFeatureParser().parse(in_dir)
    
    n_vars = max(k for k,v in features.items() if isinstance(v,lit))
    

    # get indicator variables of non-trivial factors
    factor_indicators = np.array([k for k,v in features.items() if not isinstance(v, lit)])

    
    assert max_nb_f>=len(factor_indicators), f'the number of features ({len(factor_indicators)}) exceeds the maximum number of factors specified ({max_nb_f})'
    
    # make count manager
    cmgr = count_manager()
    
    datasets = {}
    for data_path, data_name in ( (train_path,"train"), (valid_path, "valid")) + (() if test_path is None else ((test_path, "test"),)):
        data, _, _ = IO_manager.read_from_csv(data_path, ',')        
        data = cmgr.compress_data_set(data, data_name)
        cmgr.count_factors(features.values(), data, data_name)
        datasets[data_name]=data

    # initialize indicator manager
    imgr = indicator_manager(range(n_vars+1, n_vars+max_nb_f+2))
    
    # sample which factor indicators to keep for each of the models 
    sample_vecs = []

    for pop_i in range(population_size):

        # sample
        sample_vec = np.array([False])
        while sample_vec.sum()<1:
            sample_vec = np.random.rand(len(factor_indicators))<sample_prob
            sample_vecs.append(sample_vec)
        
        # update the indicator manager
        for indicator in factor_indicators[sample_vec]:
            claim_indicator_for_factor(imgr, features[indicator], indicator)

    
    # build the models based on the sampled factors
    models = []
    for sample_i, sample_vec in enumerate(sample_vecs,1):
        print('make sample', sample_i, 'out of', population_size)

        # make sdd of sample
        vtr = Vtree.from_file(f'{in_dir}/vtree.vtree')
        mgr = SddManager.from_vtree(vtr)
        mgr.auto_gc_and_minimize_on()
        sdd = mgr.read_sdd_file(f'{in_dir}/model.sdd'.encode())
        exists_map = np.zeros(mgr.var_count()+1, np.int32)
        exists_map[factor_indicators[~sample_vec]]=1
        sdd = mgr.exists_multiple_static(exists_map, sdd)
        sdd.ref()
        mgr.minimize()
        sdd.deref()

        # make model of sample
        model = Model(n_vars)
        model.factors = [(f,w,e,i) for f,w,e,i in all_factors if i<=n_vars or i in factor_indicators[sample_vec]]
        model.domain_size = n_vars
        model.nb_factors = sample_vec.sum()
        model.max_nb_factors = max_nb_f
        model.indicator_manager=imgr
        model.count_manager=cmgr
        model.mgr=mgr
        model.sdd=sdd
        model.dirty=True

        models.append(model)
    
    models = fit_population_multitprog(models, datasets["train"], "train", 'popinit')
    
    mio = model_io(out_dir,'')    
    mio.save_models(models)


    


    
def parse(args):
    parser = argparse.ArgumentParser()
    parser.add_argument('-init_type', required=True, choices=['sample'],)
    parser.add_argument('-in_dir', required = True, help='The LearnSDD model directory')
    parser.add_argument('-out_dir', required = True, help='The directory in which the population is to be safed.')
    parser.add_argument('-train', required = True, help='The train data set')
    parser.add_argument('-valid', required = True, help='The validation data set')
    parser.add_argument('-test', required = False, help='The test data set', default=None)
    parser.add_argument('-max_nb_f',
                    help='The maximum number of features a MLN can have.',
                    action='store',
                    type = int,
                    default=100)    
    parser.add_argument('-sample_prob',
                    help='The maximum number of features a MLN can have.',
                    type = float,
                    default=0.5)    
    parser.add_argument('-population_size',
                    help='''The size of the population.''',
                    action='store',
                    type = int,
                    default=52)
    
    return parser.parse_args(args)
    
def main(args):
    args = parse(args)
    if args.init_type=='sample':
        sample_initial_population_from_learnSdd_model(args.sample_prob, args.population_size, args.max_nb_f, args.in_dir, args.train, args.valid, args.test, args.out_dir)

    
if __name__ == "__main__":
    main(sys.argv[1:])
    
    