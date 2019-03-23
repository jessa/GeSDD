from logic.factor import factor
from functools import reduce
import pysdd

class disj(factor):

    # List should be integers
    def __init__(self, list_of_factors):
        self.list_of_factors = list_of_factors
        self.cached_sdd = None
        super().__init__()

    def to_string(self):
        return "(" + " | ".join(map(lambda x: x.to_string(), self.list_of_factors)) + ")"

    def ref(self):

        if self.cached_sdd == None:
            return

        if self.cached_sdd.garbage_collected():
            self.cached_sdd = None
            return

        self.cached_sdd.ref()

    def deref(self):
        if self.cached_sdd == None:
            return

        if self.cached_sdd.garbage_collected(): #Already derefd
            self.cached_sdd = None
            return

        self.cached_sdd.deref()

    def to_sdd(self, mgr):

        if self.cached_sdd != None and not self.cached_sdd.garbage_collected():
            return self.cached_sdd

        sdd_of_factors = map(lambda x: x.to_sdd(mgr), self.list_of_factors)
        disjunction_of_sdd = reduce( lambda x,y: x | y, sdd_of_factors )

        self.cached_sdd = disjunction_of_sdd

        return disjunction_of_sdd

    def evaluate(self, world):
        for factor in self.list_of_factors:
            if factor.evaluate(world) == True:
                return True
        return False

    def __eq__(self, other):
        if not isinstance(other, disj):
            return False

        return self.list_of_factors == other.list_of_factors

    pass
