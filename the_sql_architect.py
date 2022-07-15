###########################################################################
## Code for Database Applications Design and Tuning - NUS CS5421
## Creator: Kenneth Goh
## Message: You're to clone, re-produce. Will appreciate if you would help improve and provide a merge request.
## Coverage:
##   - Functional Dependencies
###########################################################################

###########################################################################
## Libraries:
from itertools import combinations, permutations
import copy
import random
import numpy as np
###########################################################################

# a = np.array([
#         [ 0,  1,  2],
#         [ 3,  4,  5],
#         [ 6,  7,  8],
#         [ 0,  1,  2],
#         [12, 13, 14]
#     ])
# print(a)
# print()
# answer = a[np.where((a[:,0] == 0) & (a[:,1] == 1))]
# print(answer)

class Dependency:

    def __init__(self, lhs, rhs):
        lhs_copy = copy.deepcopy(lhs)
        lhs_copy.sort()
        rhs_copy = copy.deepcopy(rhs)
        rhs_copy.sort()
        self.lhs = lhs_copy
        self.rhs = rhs_copy
        pass

    def __lt__(self, other):
        """
        The sort() method (and the sorted() function) will then be able to compare the objects, and thereby sort them. This works best when you will only ever sort them on this attribute, however.
        """
        return [self.lhs, self.rhs] < [other.lhs, other.rhs]

    def is_same(self, dependency):
        """
        is_same takes in another dependency and compares it with itself if they are the same
        """
        lhs_equal = self.lhs == dependency.lhs
        rhs_equal = self.rhs == dependency.rhs
        return lhs_equal and rhs_equal

    def convert_to_list(self):
        return [self.lhs, self.rhs]

    def set_lhs_index(self, lhs_index):
        self.lhs_index = lhs_index
        pass

    def set_rhs_index(self, rhs_index):
        self.rhs_index = rhs_index
        pass

class FunctionalDependency(Dependency):
    """
    This is the FunctionalDependency class which holds all the functions for a functional dependency
    """
    def is_mvd(self):
        return False

    def is_fd(self):
        return True

    def get_copy(self):
        return FunctionalDependency(self.lhs, self.rhs)

    def get_trivial_type(self):
        """
        get_trivial_type returns you a string, from enumeration ["completely_not_trivial", "completely_trivial", "not_completely_not_trivial"]
        """
        dependent_Y = set(self.rhs)
        dependee_X = set(self.lhs)
        # If dependent_Y is empty set its trivial
        if len(dependent_Y) == 0 or dependent_Y.issubset(dependee_X):
            return "trivial"
        if len(dependent_Y) > 0 and len(dependent_Y.intersection(dependee_X)) == 0:
            return "completely_not_trivial"
        if not dependent_Y.issubset(dependee_X):
            return "not_completely_not_trivial"

    def is_trivial(self):
        """
        Accepts one functional_dependency, a nested list, [["X"],["Y"]], X -> Y.
        Returns True if it's trivial.
        """
        dependent_Y = set(self.rhs)
        dependee_X = set(self.lhs)

        if len(dependent_Y) == 0:
            return True

        return dependent_Y.issubset(dependee_X)


    def lhs_is_equal_to(self, target_set):
        """
        lhs_is_equal_to takes in a list of attribute and compares with it's own lhs
        returns a boolean if they are equal
        """
        lhs_copy = copy.deepcopy(self.lhs)
        lhs_copy.sort()
        target_set = copy.deepcopy(target_set)
        target_set.sort()
        return lhs_copy == target_set

    def rhs_is_subset_of(self, target_set):
        """
        rhs_is_subset_of returns boolean if the rhs is subset of the targetset.
        """
        return set(self.rhs).issubset(set(target_set))

    def lhs_is_subset_of(self, target_set):
        """
        lhs_is_subset_of returns boolean if the lhs is subset of the targetset.
        """
        return set(self.lhs).issubset(set(target_set))

class MultiValueDependency(Dependency):
    """
    This is the MultiValueDependency class which holds all the functions for a multi value dependency
    """
    def is_mvd(self):
        return True

    def is_fd(self):
        return False

    def get_copy(self):
        return MultiValueDependency(self.lhs, self.rhs)

class SQLDatabaseArchitect:
    """
    This is a DatabaseDesigner class, engineered to fit the context of NUS, CS5421.
    As the name suggest this is used to design, plan and tune a SQL database architecture.
    """

    def __init__(self, relation_schema=None, functional_dependencies=None, multivalued_dependencies=None):
        """
        Accepts 3 parameters, "relation_schema",  "functional_dependencies" and "multivalued_dependencies" where 
        "relation_schema" is a list of attributes. For example, ["A", "B", "C", "D", "E"].
        And "functional_dependencies" & "multivalued_dependencies" is a nested list of functional dependencies. For example,
        [ [["A"], ["C"]], [["A", "B"], ["C", "D"]] ] is read as A -> C, A implies C and 
        AB -> CD, AB implies CD. Or, CD depends on AB and C depends on A
        """
        if relation_schema:
            relation_schema_copy = list(set(relation_schema))
            relation_schema_copy.sort()
            self.relation_schema = relation_schema_copy
        if functional_dependencies:
            temp_functional_dependencies = []
            for dependency in functional_dependencies:
                temp_dep = None
                if isinstance(dependency, FunctionalDependency):
                    temp_dep = dependency.get_copy()
                else:
                    temp_dep = FunctionalDependency(dependency[0], dependency[1])
                temp_functional_dependencies.append(temp_dep)
            self.functional_dependencies = temp_functional_dependencies
        if multivalued_dependencies:
            temp_multivalued_dependencies = []
            for dependency in multivalued_dependencies:
                temp_dep = MultiValueDependency(dependency[0], dependency[1])
                temp_multivalued_dependencies.append(temp_dep)
            self.multivalued_dependencies = temp_multivalued_dependencies
        pass
    
    ################
    # DEPENDENCIES #
    ################

    def list_to_functional_dependencies(self, functional_dependencies):
        """
        list_to_functional_dependencies takes in a list of functional dependencies and converts each
        functional dependency into an instance of the class FunctionalDependency
        """
        temp_functional_dependencies = []
        for dependency in functional_dependencies:
            temp_dep = FunctionalDependency(dependency[0], dependency[1])
            temp_functional_dependencies.append(temp_dep)
        return temp_functional_dependencies

    def list_to_multivalued_dependencies(self, multivalued_dependencies):
        """
        list_to_multivalued_dependencies takes in a list of multivalued dependencies and converts each
        multivalued dependency into an instance of the class MultiValueDependency
        """
        temp_multivalued_dependencies = []
        for dependency in multivalued_dependencies:
            temp_dep = MultiValueDependency(dependency[0], dependency[1])
            temp_multivalued_dependencies.append(temp_dep)
        return temp_multivalued_dependencies

    def fd_closure(self, functional_dependencies, S):
        """
        closure accepts 2 parameters. functional_dependencies and target set S.
        returns the closure, S+, a list of attributes.
        NOTE: the functional dependencies needs to be in the FunctionalDependency class form.
        """
        # 1. Add elements of attribute set to the result set.
        results = set(copy.deepcopy(S))
        # 2. Recursively add elements to the result set, until NO NEW implication/dependent is found
        continue_loop = True
        while continue_loop:
            # Break loop by default
            continue_loop = False
            # Get combination of results of various length
            # Loop through current results
            for i in range(len(results) + 1):
                # Get combination based on current desired length
                combo = list(combinations(results, i))
                for combi in combo:
                    combiList = list(combi)
                    combiList.sort()
                    # Check if the combi is in the functional dependency
                    # Loop through functional dependency
                    for dependency in functional_dependencies:
                        # If the combination is equal to the lhs of the dependency AND dependency's dependent, rhs is not a subset of the results,
                        # we have found a new dependent and should restart our search, set back loop to true
                        if dependency.lhs_is_equal_to(combiList) and not dependency.rhs_is_subset_of(results):
                            results = set.union(set(dependency.rhs), results)
                            continue_loop = True
            # if not new depedent found, break loop
            if not continue_loop:
                break
        results = list(results)
        results.sort()
        return results

    def generate_all_fd_closures(self, functional_dependencies):
        """
        generate_all_closures accepts one functional depedency as a parameter and returns all closures, F+ from self.relation_schema.
        NOTE: functional_dependencies needs to be a list of the class FunctionalDependency.
        NOTE: It DOES overide the self.candidate_keys
        NOTE: It DOES overide the self.prime_attributes
        NOTE: It DOES overide the self.all_closures
        NOTE: Returns the closure of all subsets of attributes excluding super keys that are not candidate keys.
        """
        # Initialize candidate keys list
        candidate_keys = []
        results = []
        # For each combination of the relation_schema LENGTH starting from 1. Thus, this excludes the whole relation_schema itself because we are NOT running len + 1
        for each_combi_len in range(1, len(self.relation_schema)):
            # Generate combo of relation_schema, based on the length
            combo = list(combinations(self.relation_schema, each_combi_len))
            # For each combo, check that it's NOT a superkey that is NOT a candidate key
            # A superkey closure S+, will be the same as relation_schema. 
            # But since we ONLY WANT candidate keys, we just have to compare it with the candidate keys list
            for combi in combo:
                # Check that the current combination is not a superkey of a candidate_key
                combiList = list(combi)
                is_subset = False
                for candidate_key in candidate_keys:
                    if(set(candidate_key).issubset(set(combiList))):
                        # If it's a subset then we just break this candidate_key loop and skip the combi loop
                        is_subset = True
                        break
                if is_subset:
                    # skips the current combi loop, this will prevent adding superkeys that are not candidate keys
                    continue
                combi_closure = self.fd_closure(functional_dependencies, copy.deepcopy(combiList))
                combi_closure.sort()
                combiList.sort()
                # If the closure has the same length as relation_schema
                if len(combi_closure) == len(self.relation_schema):
                    candidate_keys.append(combiList)
                results.append([combiList, combi_closure])
        # set candidate keys
        candidate_keys.sort()
        self.candidate_keys = candidate_keys
        prime_attributes = []
        # set primary attributes
        for c_keys in candidate_keys:
            for key in c_keys:
                if key not in prime_attributes:
                    prime_attributes.append(key)
        prime_attributes.sort()
        self.prime_attributes = prime_attributes
        self.all_closures = results
        return results

    def generate_fd_min_covers(self, functional_dependencies):
        """
        generate_min_covers takes in one parameter, functional_dependencies and returns ALL possible minimum covers.
        A minimal cover, Σ', of a set of functional dependencies, Σ, is the set of functional dependencies, Σ', that is both minimal and equivalent to Σ.
        Basically it's another minimum way to represent Σ.
        NOTE: If functional_dependencies is needs to be a list of class FunctionalDependency
        NOTE: it will overide self.minimum_covers
        """
        # Step 1: Make RHS of each functional dependency singletons
        # Intuition: This is to break down the dependent and later minimalise each set of dependency
        sigma_1 = []
        # For each functional dependency
        for dependency in functional_dependencies:
            # loop through RHS and construct decomposition
            if len(dependency.rhs) > 1:
                for attribute in dependency.rhs:
                    new_dep = FunctionalDependency(copy.deepcopy(dependency.lhs), [attribute])
                    sigma_1.append(new_dep)
            else:
                dependency_copy = dependency.get_copy()
                sigma_1.append(dependency_copy)
        # remove duplicates
        final_sigma1 = self.__remove_duplicates_in_dependencies(sigma_1)
        final_sigma1.sort()
        # Step 2: Minimalise the LHS, of each dependency
        # Intuition: This is to understand the minimal LHS to represent the singleton
        sigma_2 = []
        # this is to keep the order of final_sigma1 and sigma_2 the SAME, as we will be mutating sigma2
        for dependency in final_sigma1:
            dependency_copy = dependency.get_copy()
            sigma_2.append(dependency_copy)
        for dependency_index, dependency in enumerate(final_sigma1):
            temp_dep = FunctionalDependency(dependency.lhs, dependency.rhs)
            dep_index = self.__get_dependency_index_(sigma_2, temp_dep)
            old_lhs_dependency = copy.deepcopy(temp_dep.lhs)
            # If the LHS is already a singleton there's nothing to change
            if len(temp_dep.lhs) <= 1:
                continue
            # A copy is required because as we loop through we cannot change the original list
            new_lhs_dependency = copy.deepcopy(temp_dep.lhs)
            new_rhs_dependency = copy.deepcopy(temp_dep.rhs)
            for attribute_X in old_lhs_dependency:
                x_index = new_lhs_dependency.index(attribute_X)
                x_popped = new_lhs_dependency.pop(x_index) # temporarily pop X out
                # Get the closure of the remaining attributes
                remainder_closure = self.fd_closure(sigma_2, new_lhs_dependency)
                # If the remainder_closure also implies the RHS of the original dependency, then attribute X is redundant
                # BUT if it does not implies we have to add it back
                if not temp_dep.rhs_is_subset_of(remainder_closure):
                    new_lhs_dependency.append(attribute_X)
            # Update the sigma_2
            sigma_2[dep_index] = FunctionalDependency(new_lhs_dependency, new_rhs_dependency)
        
        # remove duplicates
        sigma_2_less_duplicates = self.__remove_duplicates_in_dependencies(sigma_2)
        # remove trivials
        final_sigma2 = []
        for functional_dependency in sigma_2_less_duplicates:
            if not functional_dependency.is_trivial():
                final_sigma2.append(functional_dependency)
        final_sigma2.sort()
        final_sigma2_permutate = [list(fds) for fds in list(permutations(final_sigma2, len(final_sigma2)))]
        sigma_3 = []
        # For each permutation
        for fd_permutation in final_sigma2_permutate:
            # Step 3: Minimalise this set of functional dependency, no redundant dependencies
            # Intuition: Based on various permutation we can test redundancy of each dependency.
            # Because the order of testing each dependency matters therefore we permutated the orders earlier
            new_fd_permutation = []
            # This for loop is to keep the order
            for dep in fd_permutation:
                dep_copy = dep.get_copy()
                new_fd_permutation.append(dep_copy)
            # For each dependency
            for dependency in fd_permutation:
                # temporarily remove the dependency from the permutation
                dep_index = self.__get_dependency_index_(new_fd_permutation, dependency)
                popped_dep = new_fd_permutation.pop(dep_index)
                # Get the closure of the new_fd_permutation based on the attribute set, lhs_dependency
                closure_without_dependency = self.fd_closure(new_fd_permutation, popped_dep.lhs)
                # If less_dependency_closure CAN STILL represent the RHS of the popped dependency,
                # the dependency is redundant. If NOT redundant, put it back in
                if not popped_dep.rhs_is_subset_of(closure_without_dependency):
                    new_fd_permutation.append(popped_dep)
            # sort this for easy removal of duplicates
            new_fd_permutation.sort()
            sigma_3.append(new_fd_permutation)
        final_sigma_3 = []
        for functional_dependencies in sigma_3:
            functional_dependencies_expanded = self.__return_expanded_dependencies(functional_dependencies)
            functional_dependencies_expanded.sort()
            final_sigma_3.append(functional_dependencies_expanded)
        final_sigma_3 = self.__remove_duplicated_dependencies_set_in_list_of_dependencies(final_sigma_3)
        self.mininum_covers = final_sigma_3
        return final_sigma_3

    def general_fd_all_min_covers(self, functional_dependencies):
        all_closures = self.generate_all_fd_closures(functional_dependencies)
        all_closures_in_functional_dependency_class = [self.__convert_list_to_functional_dependency(dep) for dep in all_closures]
        # print('all_closures : {}'.format(all_closures))
        all_min_covers = self.generate_fd_min_covers(all_closures_in_functional_dependency_class)
        return all_min_covers
    
    def generate_compact_minimal_cover(self, minimal_cover):
        """
        generate_compact_minimal_cover accepts 1 parameter, minimal_cover.
        This returns the compact minimal cover list and sets self.compact_minimal_cover.
        NOTE: minimal_cover has to be a list of class FunctionalDependency.
        """
        # Compile all LHS by first targeting the first indexed dependency and loop through the rest to check if they have the same
        results = []
        # For each dependency
        for target_index in range(len(minimal_cover)):
            target_lhs = copy.deepcopy(minimal_cover[target_index].lhs)
            target_rhs = set(copy.deepcopy(minimal_cover[target_index].rhs))
            target_lhs.sort()
            # Search for another dependency
            for dependency in minimal_cover:
                search_dependency_lhs = dependency.lhs
                search_dependency_rhs = set(dependency.rhs)
                search_dependency_lhs.sort()
                # That have the same lhs, union their rhs
                if search_dependency_lhs == target_lhs:
                    target_rhs = target_rhs.union(search_dependency_rhs)
            target_rhs = list(target_rhs)
            target_rhs.sort()
            results.append(FunctionalDependency(target_lhs, target_rhs))
        results = self.__remove_duplicates_in_dependencies(results)
        results = [dep.convert_to_list() for dep in results]
        results.sort()
        self.compact_minimal_cover = results
        return results

    def get_min_cover(self, randomness=False):
        if len(self.mininum_covers) < 1:
            return []
        elif not randomness:
            return self.mininum_covers[0]
        else:
            return random.choice(self.mininum_covers)

    #################
    # NORMALISATION #
    #################
    def is_2nf(self, functional_dependencies):
        """
        is_2nf takes 1 parameter, functional_dependencies.
        NOTE: You may pass in minimal cover as functional_dependencies.)
        NOTE: functional_dependencies is a list of FunctionalDependency class.
        This returns the dictionary { result: boolean, vilolation_dependency: [[X],[Y]] }, where [[X],[Y]] equals X implies Y, if the set of functional_dependencies is in 2NF.
        The vilolation_dependency parameter in the response will be an empty list if there's no violation found.
        A relation R with set of functional dependencies Σ is in 2NF if and only if for every functional dependency X → {A} ∈ Σ+, (singleton rhs, {A})
        NOTE: for each and every X we have to check:
            1. X → {A} is trivial OR
            2. A is a prime attribute (A ∈ candidate key) OR (NOTE: thus we have to generate_all_closures FIRST)
            3. X is not a proper subset of candidate key. (X is not subset of a candidate key AND can imply the same thing as the candidate key)
        """
        if not self.prime_attributes:
            return "This function requires prime_attributes."
        if not self.candidate_keys:
            return "This function requires candidate_keys."

        # Find functional dependencies that is a singleton on the rhs
        singleton_rhs = []
        for dependency in functional_dependencies:
            if len(dependency.rhs) == 1:
                dependency_copy = dependency.get_copy()
                singleton_rhs.append(dependency_copy)

        # For each dependency in singleton_rhs
        for dependency in singleton_rhs:
            # if it's trivial, it is in 2NF, skip and continue to loop
            if dependency.is_trivial():
                continue
            # if rhs of dependency is a prime attribute, it is in 2NF, skip
            if dependency.rhs not in self.prime_attributes:
                continue
            # if it is a candidate key continue
            if dependency.lhs in self.candidate_keys:
                continue
            for key in self.candidate_keys:
                if dependency.lhs_is_subset_of(key):
                    return { "result": False, "vilolation_dependency": dependency.convert_to_list() }

        return { "result": True , "vilolation_dependency": [] }

    def is_3nf(self, functional_dependencies=None):
        """
        is_3nf takes 1 parameter, functional_dependencies.
        NOTE: functional_dependencies is a list of class FunctionalDependency.
        NOTE: You may pass in minimal cover as functional_dependencies.
        This returns the dictionary { result: boolean, vilolation_dependency: [[X],[Y]] }, where [[X],[Y]] equals X implies Y, if the set of functional_dependencies is in 3NF.
        the vilolation_dependency parameter in the response will be an empty list if there's no violation found.
        A relation R with set of functional dependencies Σ is in 3NF if and only if for every functional dependency X → {A} ∈ Σ+:, {A} is a single attribute aka singleton.
        NOTE: for each and every X we have to check:
            1. X → {A} is trivial OR
            2. A is a prime attribute OR NOTE: thus we have to generate_all_closures FIRST
            3. X is a superkey, A superkey is a set of attributes of a relation whose knowledge determines the value of the entire t-uple. Including the candidate keys.
        """
        if not self.prime_attributes:
            return "This function requires prime_attributes."
        
        # This is in case you will want to use Σ+ instead of Σ OR even better we can use minimal cover.

        singleton_rhs = []
        # Find functional dependencies that is a singleton on the rhs
        for dependency in functional_dependencies:
            if len(dependency.rhs) == 1:
                dependency_copy = dependency.get_copy()
                singleton_rhs.append(dependency_copy)
        
        # For each dependency in singleton_rhs
        for dependency in singleton_rhs:
            # if it's trivial, it is in 2NF, skip and continue to loop
            if dependency.is_trivial():
                continue
            # if rhs of dependency is a prime attribute, it is in 2NF, skip
            if dependency.rhs in self.prime_attributes:
                continue
            dependency_closure = self.fd_closure(functional_dependencies, dependency.lhs)
            # if it's not a superkey, then it violates the 3NF rule and the function can terminate.
            if len(dependency_closure) != len(self.relation_schema):
                return { "result": False, "vilolation_dependency": dependency.convert_to_list() }

        return { "result": True, "vilolation_dependency": [] }

    def is_bcnf(self, functional_dependencies=None):
        """
        is_bcnf takes 1 parameter, functional_dependencies.
        NOTE: functional_dependencies is a list of class FunctionalDependency.
        NOTE: You may pass in minimal cover as functional_dependencies.
        This returns the dictionary { result: boolean, vilolation_dependency: [[X],[Y]] }, where [[X],[Y]] equals X implies Y, if the set of functional_dependencies is in BCNF.
        the vilolation_dependency parameter in the response will be an empty list if there's no violation found.
        A relation R with set of functional dependencies Σ is in BCNF if and only if for every functional dependency X → {A} ∈ Σ+:
        NOTE: for each and every X we have to check:
            1. X → {A} is trivial OR
            2. X is a superkey, A superkey is a set of attributes of a relation whose knowledge determines the value of the entire t-uple. Including the candidate keys.
        """
        if not self.prime_attributes:
            return "This function requires prime_attributes."

        singleton_rhs = []
        # Find functional dependencies that is a singleton on the rhs
        for dependency in functional_dependencies:
            if len(dependency.rhs) == 1:
                dependency_copy = dependency.get_copy()
                singleton_rhs.append(dependency_copy)
        
        # For each dependency in singleton_rhs
        for dependency in singleton_rhs:
            # if it's trivial, it is in 2NF, skip and continue to loop
            if dependency.is_trivial():
                continue
            # if rhs of dependency is a prime attribute, it is in 2NF, skip
            if dependency.rhs in self.prime_attributes:
                continue
            dependency_closure = self.fd_closure(functional_dependencies, dependency.lhs)
            # if it's not a superkey, then it violates the 3NF rule and the function can terminate.
            if len(dependency_closure) != len(self.relation_schema):
                return { "result": False, "vilolation_dependency": dependency.convert_to_list() }

        return { "result": True, "vilolation_dependency": [] }

    # bcnf_decomposition Train of thoughts:
    # Check if the functional_dependencies with the relation schema violates the BCNF,
    # if it does not this is in BCNF, return relation_schema and functional_dependencies
    # else we need to let R1 be {X}+, where X -> Y violates the BCNF,
    # let Σ1 be the set of functional dependencies derived from the minimal cover that correlates to R1
    # let R2 be (R- {X}+ ) U X,
    # let Σ2 be the set of functional dependencies derived from the minimal cover that correlates to R2
    def bcnf_decomposition(self, relation_schema, functional_dependencies):
        """
        bcnf_decomposition takes in the relation_schema and functional_dependencies.
        This relation is then decomposed into smaller relations, fragments, in order to remove redundant data.
        NOTE: functional_dependencies is a list of class FunctionalDependency.
        NOTE: The decomposition method is based on the assumption that a database can be represented by a universal relation 
        which contains all the attributes of the database (this is called the universal relation assumption). 
        NOTE: Synthesis method assumes universal relation assumption also, However the decomposition and synthesis method can be applied to parts of the design
        NOTE: bcnf_decomposition is guaranteed lossless decompositions BUT NOT guaranteed dependency preserving.
        """
        # NOTE: If the recursion has no more functional_dependencies, we return an empty list and mark as BCNF Complete.
        if functional_dependencies == []:
            return [[relation_schema, []]]
        temp_functional_dependencies = []
        # This makes a copy and keeps the function flexible so that we can return lists type
        for dependency in functional_dependencies:
            temp_dep = None
            if isinstance(dependency, FunctionalDependency):
                temp_dep = dependency.get_copy()
            else:
                temp_dep = FunctionalDependency(dependency[0], dependency[1])
            temp_functional_dependencies.append(temp_dep)
        functional_dependencies = temp_functional_dependencies

        # iniitalize class for the recusion to work as we make use heavily on self.candidate_keys etc. so it has to be a new agent
        bcnf_worker = SQLDatabaseArchitect(relation_schema, functional_dependencies)
        # NOTE: To work on this algorithmn we need the Attribute Closures, Candidate Keys, Prime Attributes, Minimal Cover and Compact Minimal Cover
        # Step 1. generate_all_closures to get Attribute Closures, Candidate Keys, Prime Attributes
        bcnf_worker.generate_all_fd_closures(functional_dependencies) # sets self.candidate_keys, self.prime_attributes, self.all_closures
        bcnf_worker.generate_fd_min_covers(functional_dependencies) # sets self.minimum_covers
        bcnf_worker_first_min_cover = bcnf_worker.get_min_cover() # gets the first min cover
        bcnf_worker_first_min_cover_class = [FunctionalDependency(dep[0], dep[1]) for dep in bcnf_worker_first_min_cover]
        # Step 2. generate_compact_minimal_cover to get Compact Minimal Cover
        bcnf_worker_compact_min_covers = bcnf_worker.generate_compact_minimal_cover(bcnf_worker_first_min_cover_class)
        bcnf_worker_compact_min_covers_class = [FunctionalDependency(dep[0], dep[1]) for dep in bcnf_worker_compact_min_covers]

        # Check if bcnf_worker_compact_min_covers is in BCNF
        is_bcnf_results = bcnf_worker.is_bcnf(bcnf_worker_compact_min_covers_class)
        # If is_bcnf_results["result"] is True
        if is_bcnf_results['result']:
            return [[copy.deepcopy(bcnf_worker.relation_schema), copy.deepcopy(bcnf_worker.functional_dependencies) ]] # TODO check that if it does not violates it is always an empty set for the bcnf
        else:
            vilolation_dependency = is_bcnf_results["vilolation_dependency"]
            vilolation_dependency_lhs = copy.deepcopy(vilolation_dependency[0])
            
            r1 = bcnf_worker.fd_closure(bcnf_worker_compact_min_covers_class, vilolation_dependency_lhs)
            # R2 = X_lhs U (R − R1)
            r2 = []
            for relation in bcnf_worker.relation_schema:
                if relation not in r1:
                    r2.append(relation)
            r2 = list(set.union(set(r2), set(vilolation_dependency_lhs)))
            
            # Calculate Sigma 1
            sigma1 = []
            # For each dependency in the compact minimal covers
            for dependency in bcnf_worker_compact_min_covers_class:
                # get the set of attributes and check if they are all a subset of the relation.
                attributes = set(dependency.lhs + dependency.rhs)
                if attributes.issubset(set(r1)):
                    sigma1.append(dependency.get_copy())
            # Calculate Sigma 2
            sigma2  =[]
            # For each dependency in the compact minimal covers
            for dependency in bcnf_worker_compact_min_covers_class:
                # get the set of attributes and check if they are all a subset of the relation.
                attributes = set(dependency.lhs + dependency.rhs)
                if attributes.issubset(set(r2)):
                    sigma2.append(dependency.get_copy())
            return bcnf_worker.bcnf_decomposition(r1, sigma1) + bcnf_worker.bcnf_decomposition(r2, sigma2)

    def sythesis_3nf_normalisation(self, relation_schema, functional_dependencies):
        """
        sythesis_3nf_normalisation takes in the relation_schema and functional_dependencies.
        This returns the sets of fragments.
        NOTE: functional_dependencies is a list of class FunctionalDependency.
        NOTE: Synthesis method assumes universal relation assumption. However the decomposition and synthesis method can be applied to parts of the design
        NOTE: sythesis_3nf_normalisation is guaranteed results are in 3nf but not in BCNF, it will be a lossless decompositions, AND guaranteed to be dependency preserving.
        If there are several minimal covers there are several synthesis
        """

        # NOTE: To work on this algorithmn we need the Attribute Closures, Candidate Keys, Prime Attributes, Minimal Cover and Compact Minimal Cover
        # Step 1. generate_all_fd_closures to get Attribute Closures, Candidate Keys, Prime Attributes
        self.generate_all_fd_closures(functional_dependencies) # sets self.candidate_keys, self.prime_attributes, self.all_closures
        self.generate_fd_min_covers(functional_dependencies) # sets self.minimum_covers
        synthesis_worker_first_min_cover = self.get_min_cover() # gets the first min cover
        synthesis_worker_first_min_cover_class = [FunctionalDependency(dep[0], dep[1]) for dep in synthesis_worker_first_min_cover]
        # Step 2. generate_compact_minimal_cover to get Compact Minimal Cover
        synthesis_worker_compact_min_covers = self.generate_compact_minimal_cover(synthesis_worker_first_min_cover_class)
        synthesis_worker_compact_min_covers_class = [FunctionalDependency(dep[0], dep[1]) for dep in synthesis_worker_compact_min_covers]
        results = []
        fragment_only_results = [] # for us to compare fragments easily later

        # Step 3. Create fragments For each dependency in the compact minimal covers
        for dependency in synthesis_worker_compact_min_covers_class:
            # create a fragment/relation with X U Y, where X -> Y
            fragment = list(set.union(set(dependency.lhs), set(dependency.rhs)))
            projected_sigma_for_fragment = [] # projected functional dependencies

            # For each dependency2 in the compact minimal covers
            for dependency2 in synthesis_worker_compact_min_covers_class:
                # get the set of attributes and check if they are all a subset of the relation.
                attributes = set(dependency2.lhs + dependency2.rhs)
                if attributes.issubset(set(fragment)):
                    projected_sigma_for_fragment.append(dependency2)
            fragment_only_results.append(fragment)
            results.append([fragment, projected_sigma_for_fragment])

        # Step 4. Removed subsumed
        # If some fragment, f, is a subset of another fragment, remove it (subsumed)
        fragment_only_results_copy = copy.deepcopy(fragment_only_results)
        for fragment in fragment_only_results_copy:
            # Remove fragment from fragment list to be mutated
            fragment_index = fragment_only_results.index(fragment)
            target_fragment = fragment_only_results.pop(fragment_index)
            is_subset = False
            # Check the mutated list if there's any subset
            for remaining_fragment in fragment_only_results:
                # if it is a subset of any other fragment
                if set(target_fragment).issubset(remaining_fragment):
                    # remember to pop from the results list too
                    results.pop(fragment_index)
                    is_subset = True
            # if it's not a subset. put it back in
            if not is_subset:
                fragment_only_results.append(target_fragment)

        candidate_key_exists = False
        # Step 5. Check if candidate key exists
        for key in self.candidate_keys:
            if key in fragment_only_results:
                candidate_key_exists = True
        
        # Step 6. If it does not exist get one candidate key and add it to result
        if not candidate_key_exists:
            # pick one at random
            candidate_key = random.choice(self.candidate_keys)
            candidate_key_attributes = set(candidate_key[0] + candidate_key[1])
            # calculate projected functiona dependency
            # For each dependency2 in the compact minimal covers
            candidate_key_sigma = []
            for dependency in synthesis_worker_compact_min_covers_class:
                # get the set of attributes and check if they are all a subset of the relation.
                attributes = set(dependency.lhs + dependency.rhs)
                if attributes.issubset(candidate_key_attributes):
                    candidate_key_sigma.append(dependency)
            fragment_only_results.append(candidate_key)
            results.append([candidate_key, candidate_key_sigma])

        return { "results": results, "fragment_only_results": fragment_only_results}

    def mvd_is_satisfied(self, relation_schema, mvd, table):
        """
        An instance r of a relation schema R satisfies the multi-valued dependency small_sigma:
        X → Y , X multi-determines Y or Y is multi-dependent on X, with X ⊂ R, Y ⊂ R and X∩Y = empty set if and only if,forZ=R - (X U Y),
        two tuples of r agree on their X-value, then there exists a t-uple of r that agrees with the first 
        tuple on the X- and Y -value and with the second on the Z-value.
        """
        lhs_X = mvd.lhs
        rhs_Y = mvd.rhs
        R_set = set(relation_schema)
        X_U_Y = set.union(set(lhs_X), set(rhs_Y))
        Z = list(R_set.difference(X_U_Y))
        X_attribute_index = []
        Y_attribute_index = []
        Z_attribute_index = []
        np_table = np.array(table)
        # SET up attribute index for each X,Y and Z
        for attribute in lhs_X:
            table_col_index = relation_schema.index(attribute)
            X_attribute_index.append(table_col_index)
        for attribute in rhs_Y:
            table_col_index = relation_schema.index(attribute)
            Y_attribute_index.append(table_col_index)
        for attribute in Z:
            table_col_index = relation_schema.index(attribute)
            Z_attribute_index.append(table_col_index)
        # We need to find 3 different tuples t1 t2 t3, so we need 3 for loops
        # There exists
        for t1_index, t1_row in enumerate(np_table):
            # A T1 that agree with the X and Y value of T3
            for t2_index, t2_row in enumerate(np_table):
                if t2_index == t1_index:
                    continue
                for t3_index, t3_row in enumerate(np_table):
                    if t3_index == t1_index or t3_index == t2_index:
                        continue
                    X_Y_index = X_attribute_index + Y_attribute_index
                    t3_X_Y = np_table[t3_index, X_Y_index]
                    t1_X_Y = np_table[t1_index, X_Y_index]
                    t3_Z = np_table[t3_index, Z_attribute_index]
                    t2_Z = np_table[t2_index, Z_attribute_index]
                    if np.array_equal(t3_X_Y, t1_X_Y) and np.array_equal(t3_Z, t2_Z):
                        return True
        return False

    def fd_is_satisfied(self, relation_schema, fd, table):
        """
        fd_is_satisfied
        """
        np_table = np.array(table)
        fd_lhs_index = fd.lhs_index
        fd_rhs_index = fd.rhs_index
        for row1_index, row1 in enumerate(np_table):
            for row2_index, row2 in enumerate(np_table):
                if row2_index == row1_index:
                    continue
                if row1[fd_lhs_index] == row2[fd_lhs_index] and row1[fd_rhs_index] != row2[fd_rhs_index]:
                    return False
        return True

    def chase_algorithm(self, relation_schema, dependencies, chase):
        """
        chase_algorithm accepts the 4 parameters, a relation_schema, dependencies, list of class FunctionalDependency or MultiValuedDependency, 
        the chase, a class FunctionalDependency or MultiValuedDependency.
        NOTE: This algorithm is the one taught in the lecture and can only apply to 2 fragments.
        The Chase is an algorithm that solves the decision problem of whether a functional or multi-valued dependency (or join dependency) small_sigma is satisfied by R with a set of functional and mutli-valued (and join) dependencies Σ.
        """
        # Step 0. set index of each dependency
        for dependency in dependencies:
            lhs_index = []
            for attr in dependency.lhs:
                att_index = relation_schema.index(attr)
                lhs_index.append(att_index)
            dependency.set_lhs_index(lhs_index)
            rhs_index = []
            for attr in dependency.rhs:
                att_index = relation_schema.index(attr)
                rhs_index.append(att_index)
            dependency.set_rhs_index(rhs_index)
        # Step 1. Create a table with two t-uples with all different values.
        # initial_table is to be mutated
        row_1 = []
        for relation_idx, relation in enumerate(relation_schema):
            row_1.append('{}{}'.format(relation,1))
        row_2 = []
        for relation_idx, relation in enumerate(relation_schema):
            row_2.append('{}{}'.format(relation,2))
        initial_table = [row_1, row_2]
        # Step 2. For each column A ∈ X (lhs), make the A-values the same
        for attribute in chase.lhs:
            table_col_index = relation_schema.index(attribute)
            for each_row in initial_table:
                each_row[table_col_index] = '{}{}'.format(attribute,1)
        
        print('initial_table')
        self.__python_2d_list_pretty_print(initial_table) 
        # Repeat until there are no changes
        changes = True
        while changes:
            changes = False

            # For each dependency
            for dependency in dependencies:
                # Step 3.1 if it is a multi-valued dependency
                # σ = X → Y : t-uples in the table with the same X-value, add two new t-uples with Y -values swapped.
                if dependency.is_mvd():
                    print('we wabnt to chase mvd {} {} is_mvds {}'.format(dependency.lhs,dependency.rhs, dependency.is_mvd()))
                    # Check if the multivalued dependency holds
                    if self.mvd_is_satisfied(relation_schema, dependency, initial_table):
                        continue
                    print('not satisfied : {}'.format(dependency.lhs))
                    # If NOT satisfied, there will be a change
                    changes = True
                    # Get indexes of X and Y values
                    all_various_table_X_values = []
                    X_values_indexes = []
                    for attribute in dependency.lhs:
                        att_index = relation_schema.index(attribute)
                        X_values_indexes.append(att_index)
                    Y_values_indexes = []
                    for attribute in dependency.rhs:
                        att_index = relation_schema.index(attribute)
                        Y_values_indexes.append(att_index)
                    # Get distinct X_values
                    for row in initial_table:
                        np_arr = np.array(row)
                        target_X_values = np_arr[X_values_indexes]
                        target_X_values = target_X_values.tolist()
                        if target_X_values not in all_various_table_X_values:
                            all_various_table_X_values.append(target_X_values)
                    # Get 2 rows with the same X values copy and swap
                    if len(all_various_table_X_values) == 2:
                        np_table = np.array(initial_table)
                        row_1 = np_table[0]
                        row_1_copy = np.copy(np_table[0])
                        row_1_Y = row_1_copy[Y_values_indexes]
                        row_2 = np_table[1]
                        row_2_Y = np_table[1, Y_values_indexes]
                        row_1[Y_values_indexes] = row_2_Y
                        row_2[Y_values_indexes] = row_1_Y
                        row_1 = row_1.tolist()
                        row_2 = row_2.tolist()
                        initial_table.append(row_1)
                        initial_table.append(row_2)
                    else:
                        for each_X_value in all_various_table_X_values:
                            rows = [] # this row will be mutated to be appended to the initial_table
                            for table_row in initial_table:
                                in_table_row = True
                                # check all attributes in table
                                for attri in each_X_value:
                                    if attri not in table_row:
                                        in_table_row = False
                                # add row to rows if all attribute in table, which means they have the same X values
                                if in_table_row:
                                    # print('each_X_value to change : ', each_X_value)
                                    rows.append(table_row)
                            rows_copy = np.array(copy.deepcopy(rows))
                            # print('old rows_copy : {}'.format(rows_copy))
                            rows_Y_copy = rows_copy[:,Y_values_indexes]
                            new_Y_rows = np.flip(rows_Y_copy, axis=0)
                            rows_copy[:,Y_values_indexes] = new_Y_rows
                            # print('new rows_copy : {}'.format(rows_copy))
                            # print()
                            rows_copy_list = rows_copy.tolist()
                            for each_new_row_copy in rows_copy_list:
                                initial_table.append(each_new_row_copy)
                elif dependency.is_fd():
                # Step 3.2 For Each functional_dependency in dependencies
                # σ = X → Y : t-uples in the tables with the same X-value, make the Y -values the same.
                    # Get all distinct X Values
                    all_various_table_X_values = []
                    np_table = np.array(initial_table)
                    X_index = dependency.lhs_index
                    Y_index = dependency.rhs_index
                    for index, row in enumerate(np_table):
                        row_value = row[X_index]
                        row_value = row_value.tolist()
                        if row_value not in all_various_table_X_values:
                            all_various_table_X_values.append(row_value)
                    # For each X value
                    for X_value in all_various_table_X_values:
                        # Get row indexes to change
                        indexes = []
                        all_Y_values = []
                        lowest_Y = None
                        for index, row in enumerate(np_table):
                            if row[X_index].tolist() == X_value:
                                indexes.append(index)
                                if lowest_Y == None:
                                    lowest_Y = row[Y_index].tolist()
                                elif row[Y_index].tolist() < lowest_Y:
                                    lowest_Y = row[Y_index].tolist()
                        np_table[indexes, Y_index] = lowest_Y
                    initial_table = np_table.tolist()
            print('initial_table :')
            self.__python_2d_list_pretty_print(initial_table)
            return initial_table

    def distinct_chase_algorithm(self, relation_schema, relation_schemas, dependencies):
        """
        distinct_chase_algorithm accepts the 4 parameters, a relation_schema list of attributes, relation_schemas, a list of relation_schema,
        the dependencies, list of class FunctionalDependency or MultiValuedDependency.
        NOTE: IF you found a row with distinguished attributes at every column, then the decomposition is lossless otherwise it is lossy.
        NOTE: This algorithm is for MORE THAN 2 FRAGMENTS
        """
        # Step 0. set index of each dependency
        for dependency in dependencies:
            lhs_index = []
            for attr in dependency.lhs:
                att_index = relation_schema.index(attr)
                lhs_index.append(att_index)
            dependency.set_lhs_index(lhs_index)
            rhs_index = []
            for attr in dependency.rhs:
                att_index = relation_schema.index(attr)
                rhs_index.append(att_index)
            dependency.set_rhs_index(rhs_index)
        # Step 1. Fill in the cells with distinguished attribute
        initial_table = []
        for each_relation in relation_schemas:
            relation_tuple = [0 for i in range(len(relation_schema)) ]
            for i in range(len(relation_schema)):
                if relation_schema[i] in each_relation:
                    relation_tuple[i] = 'A' # A for distinguished
            initial_table.append(relation_tuple)
        print('start')
        self.__python_2d_list_pretty_print(initial_table)
        print()
        # Step 2. For each σ : X → Y ∈ Σ: IF there are two rows (Ra and Rb) that have distinguished attribute in X, 
        # where Ra have distinguished attribute in Y but Rb do not have, add distinguished attribute to Y in Rb.
        continue_loop = True
        while continue_loop:
            continue_loop = False
            # For each dependency
            for dependency in dependencies:
                lhs_index = dependency.lhs_index
                rhs_index = dependency.rhs_index
                # Check If there are rows with same lhs values BUT DIFFERENT Y values
                for row1_index, row1 in enumerate(initial_table):
                    for row2_index, row2 in enumerate(initial_table):
                        if row2_index == row1_index:
                            continue
                        row1_np = np.array(row1)
                        row2_np = np.array(row2)
                        X_all_distinguished = True
                        for value in row1_np[lhs_index]:
                            if value != 'A':
                                X_all_distinguished = False
                        for value in row2_np[lhs_index]:
                            if value != 'A':
                                X_all_distinguished = False
                        # if all X values of row1 and row2 are distinguished compare their Y values
                        if X_all_distinguished:
                            row1_Y_distinguished = True
                            for value in row1_np[rhs_index]:
                                if value != 'A':
                                    row1_Y_distinguished = False
                            row2_Y_distinguished = True
                            for value in row2_np[rhs_index]:
                                if value != 'A':
                                    row2_Y_distinguished = False
                            if row1_Y_distinguished and not row2_Y_distinguished:
                                continue_loop = True
                                # BROADCASTING
                                row2_np[rhs_index] = 'A'
                        row1_list = row1_np.tolist()
                        row2_list = row2_np.tolist()
                        initial_table[row1_index] = row1_list
                        initial_table[row2_index] = row2_list
                        self.__python_2d_list_pretty_print(initial_table)
                        print()
        print('end')
        self.__python_2d_list_pretty_print(initial_table)
        print()
        return initial_table

    ####################
    # HELPER FUNCTIONS #
    ####################

    def __return_expanded_dependencies(self, dependencies):
        """
        __return_expanded_dependencies returns the list form of the list of dependencies
        """
        results = []
        for dependency in dependencies:
            results.append([dependency.lhs, dependency.rhs])
        return results

    def __remove_duplicated_dependencies_set_in_list_of_dependencies(self, dependencies_list):
        """
        __remove_duplicated_dependencies_set_in_list_of_dependencies removes duplicated sets of dependencies in
        a list of dependencies.
        NOTE: this assumes that each set of dependencies IS ALREADY SORTED AND EXPANDED
        """
        new_dependencies_list = []
        # for each old_dependency in the list of dependencies
        for old_dependencies_set in dependencies_list:
            # check if there exist a dependencies set that is the same
            if old_dependencies_set in new_dependencies_list:
                continue
            # If there's no same copy append the old_dependency_set into the new list
            new_dependencies_list.append(old_dependencies_set)
        return new_dependencies_list

    def __remove_duplicates_in_dependencies(self, dependencies):
        """
        __remove_duplicates_in_dependencies removes duplicated dependencies in
        """
        new_dependencies = []
        # for each old_dependency in the list of dependencies
        for old_dependency in dependencies:
            # for each new_dependency in the list of new_dependencies
            exist_same = False
            for new_dependency in new_dependencies:
                # check if there exist a dependency that is the same
                if new_dependency.is_same(old_dependency):
                    exist_same = True
            # If there's no same copy append the old_dependency back into the new list
            if not exist_same:
                new_dependencies.append(old_dependency)
        return new_dependencies
    
    def __get_dependency_index_(self, list_of_dependencies, dependency):
        """
        __get_dependency_index_ returns you the index of the dependency in the dependency list
        NOTE: this assumes that there are no duplicated dependencies
        """
        for index , temp_dep in enumerate(list_of_dependencies):
            if dependency.lhs == temp_dep.lhs and dependency.rhs == temp_dep.rhs:
                return index

    def __convert_list_to_functional_dependency(self, dependency_list):
        return FunctionalDependency(dependency_list[0], dependency_list[1])

    def __python_2d_list_pretty_print(self, python_list):
        np_list = np.array(python_list)
        print(np_list)
        pass

def main():
    # CLOSURE TEST
    # R = ['A', 'B', 'C', 'D']
    # FD = [[['A', 'B'], ['C']], [['C'], ['D']]]
    # closure_case1 = SQLDatabaseArchitect(R)
    # clsoure_case1_fds = closure_case1.list_to_functional_dependencies(FD)
    # closure_1 = closure_case1.fd_closure(clsoure_case1_fds, ['A', 'B'])
    # print('closure_1 : {}'.format(closure_1))

    # R = ["E", "F", "G", "H", "I", "J", "K", "L", "M", "N"]
    # FD = [[['E', 'F'], ['G']], [['F'], ['I', 'J']], [['E', 'H'], ['K', 'L']], [['K'], ['M']], [['L'], ['N']]]
    # closure_case2 = SQLDatabaseArchitect(R)
    # clsoure_case2_fds = closure_case2.list_to_functional_dependencies(FD)
    # closure_2_1 = closure_case2.fd_closure(clsoure_case2_fds, ['E', 'F'])
    # print('closure_2_1 : {}'.format(closure_2_1))
    # closure_2_2 = closure_case2.fd_closure(clsoure_case2_fds, ['E', 'F', 'H'])
    # print('closure_2_2 : {}'.format(closure_2_2))
    # closure_2_3 = closure_case2.fd_closure(clsoure_case2_fds, ['E', 'F', 'H', 'K', 'L'])
    # print('closure_2_3 : {}'.format(closure_2_3))
    # closure_2_4 = closure_case2.fd_closure(clsoure_case2_fds, ['E'])
    # print('closure_2_4 : {}'.format(closure_2_4))

    # ALL CLOSURE TEST
    # R = ['A', 'B', 'C', 'D']
    # FD = [[['A', 'B'], ['C']], [['C'], ['D']]]
    # all_closure_case1 = SQLDatabaseArchitect(R)
    # all_clsoure_case1_fds = all_closure_case1.list_to_functional_dependencies(FD)
    # all_closure_1 = all_closure_case1.generate_all_fd_closures(all_clsoure_case1_fds)
    # print('all_closure_1 : {}'.format(all_closure_1))

    # MIN COVER TEST
    # R = ['A', 'B', 'C', 'D', 'E', 'F']
    # FD = [[['A'], ['B', 'C']], [['B'], ['C','D']], [['D'], ['B']], [['A','B','E'], ['F']]]
    # min_cover_case1 = SQLDatabaseArchitect(R)
    # min_cover_case1_fds = min_cover_case1.list_to_functional_dependencies(FD)
    # min_cover_case1.generate_fd_min_covers(min_cover_case1_fds)
    # min_cover = min_cover_case1.get_min_cover()
    # print(min_cover)

    # MIN COVERS TEST
    # R = ['A', 'B', 'C', 'D', 'E']
    # FD = [[['A','B'] , ['C']], [['D'] , ['D','B']], [['B'], ['E']], [['E'] , ['D']], [['A','B','D'] , ['A','B','C','D']]]
    # min_covers_case1 = SQLDatabaseArchitect(R)
    # min_covers_case1_fds = min_covers_case1.list_to_functional_dependencies(FD)
    # min_covers_1 = min_covers_case1.generate_fd_min_covers(min_covers_case1_fds)
    # print('min_covers_1 : {}'.format(min_covers_1))

    # TODO TEST THIS ONE LAST TIME
    # ALL MIN COVERS TEST
    # R = ['A', 'B', 'C', 'D', 'E']
    # FD = [[['A','B'] , ['C']], [['D'] , ['D','B']], [['B'], ['E']], [['E'] , ['D']], [['A','B','D'] , ['A','B','C','D']]]
    # all_min_covers_case1 = SQLDatabaseArchitect(R)
    # all_min_covers_case1_fds = all_min_covers_case1.list_to_functional_dependencies(FD)
    # all_min_covers_1 = all_min_covers_case1.general_fd_all_min_covers(all_min_covers_case1_fds)
    # print('all_min_covers_1 : {}'.format(all_min_covers_1))

    # COMPACT MIN COVER TEST
    # R = ['A', 'B', 'C', 'D', 'E']
    # FD = [[['A'],['A','B','C']],[['A','B'],['A']],[['B','C'],['A','D']],[['B'], ['A', 'B']], [['C'] , ['D']]]
    # compact_min_cover_case1 = SQLDatabaseArchitect(R)
    # test_min_cover = [[['A'], ['B']],[['A'], ['C']],[['B'], ['A']],[['C'], ['D']]]
    # test_min_cover_class = [FunctionalDependency(dep[0], dep[1]) for dep in test_min_cover]
    # test_results = compact_min_cover_case1.generate_compact_minimal_cover(test_min_cover_class)
    # print(test_results)

    # BCNF DECOMPOSITION TEST
    # R = ['A', 'B', 'C', 'D', 'E']
    # FD = [[['A'],['A','B','C']],[['A','B'],['A']],[['B','C'],['A','D']], [['B'], ['A', 'B']], [['C'] , ['D']]]
    # bcnf_case1 =  SQLDatabaseArchitect(R, FD)
    # bcnf_case1_fds = bcnf_case1.list_to_functional_dependencies(FD)
    # results = bcnf_case1.bcnf_decomposition(R, bcnf_case1_fds)
    # print('bcnf')
    # results = [[decom[0], [dep.convert_to_list() for dep in decom[1]]] for decom in results]
    # print(results)

    # SYNTHESIS TEST
    # R = ['A', 'B', 'C', 'D', 'E']
    # FD = [[['A'],['A','B','C']],[['A','B'],['A']],[['B','C'],['A','D']], [['B'], ['A', 'B']], [['C'] , ['D']]]
    # synthesis_case1 =  SQLDatabaseArchitect(R, FD)
    # synthesis_case1_fds = synthesis_case1.list_to_functional_dependencies(FD)
    # response = synthesis_case1.sythesis_3nf_normalisation(R, synthesis_case1_fds)
    # print('synthesis')
    # print('results')
    # results = [[result[0], [dep.convert_to_list() for dep in result[1]]] for result in response['results']]
    # print(results)
    # print('fragments')
    # print(response['fragment_only_results'])

    # CHASE ALGORITHM TEST 1
    # R = ['A', 'B', 'C', 'D', 'E', 'G']
    # R1 = ['A', 'B', 'C', 'D', 'G'] # X
    # R2 = ['C', 'D', 'E'] # Y
    # FD = [[['A', 'B'], ['C']]]
    # MVD = [[['A', 'B'], ['E']], [['C', 'D'], ['A', 'B']]]
    # MVDS = [(dep[0], dep[1]) for dep in MVD]
    # # Check (X∩Y)->>(Y −(X∩Y)) or (X∩Y)->>(X−(X∩Y))
    # # (X∩Y)->>(Y −(X∩Y)) ==> CD ->> CDE - CD = E
    # # (X∩Y)->>(X−(X∩Y)) ==> CD ->> ABG
    # dependencies = [FunctionalDependency(['A', 'B'], ['C']), MultiValueDependency(['A', 'B'], ['E']), MultiValueDependency(['C', 'D'], ['A', 'B'])]
    # chase1 = MultiValueDependency(['C', 'D'], ['E'])
    # chase2 = MultiValueDependency(['C', 'D'], ['A', 'B', 'G'])
    # chase_case1 = SQLDatabaseArchitect(R)
    # results = chase_case1.chase_algorithm(R, dependencies, chase1)
    # mvd_is_satisfied = chase_case1.mvd_is_satisfied(R, MultiValueDependency(['C','D'], ['E']) ,results)
    # print('mvd_is_satisfied : {}'.format(mvd_is_satisfied))
    # # chase_case1.chase_algorithm(R, dependencies, chase2)

    # CHASE ALGORITHM TEST 2
    # R = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
    # FD = [[['A', 'B'], ['C']], [['C'], ['D', 'E']], [['E'], ['D']], [['F'], ['G']] ]
    # R1 = ['A', 'B', 'C', 'D', 'E'] # X
    # R2 = ['A', 'B', 'F', 'G'] # Y
    # dependencies = [
    #     FunctionalDependency(['A', 'B'], ['C']),
    #     FunctionalDependency(['C'], ['D', 'E']),
    #     FunctionalDependency(['E'], ['D']),
    #     FunctionalDependency(['F'], ['G']),
    # ]
    # # Check (X∩Y)->>(Y −(X∩Y)) or (X∩Y)->>(X−(X∩Y))
    # # (X∩Y)->>(Y −(X∩Y)) ==> AB ->> ABFG - AB = FG
    # # (X∩Y)->>(X−(X∩Y)) ==> AB ->> ABCDE - AB = CDE
    # chase1 = MultiValueDependency(['A', 'B'], ['F', 'G'])
    # chase2 = MultiValueDependency(['A', 'B'], ['C', 'D', 'E'])
    # chase_case2 = SQLDatabaseArchitect(R)
    # results = chase_case2.chase_algorithm(R, dependencies, chase2)
    # mvd_is_satisfied = chase_case2.mvd_is_satisfied(R, chase2 ,results)
    # print('mvd_is_satisfied : {}'.format(mvd_is_satisfied))

    # DISTINGUISHED CHASE ALGORITHM TEST 1
    # R = ['A', 'B', 'C', 'D', 'E']
    # RS = [
    #     ['A', 'E'],
    #     ['C', 'D'],
    #     ['A', 'B', 'C'],
    # ]
    # FD = [ [['A'], ['B','C']], [['B'], ['A']], [['C'], ['D']] ]
    # dependencies = [
    #     FunctionalDependency(['A'], ['B','C']),
    #     FunctionalDependency(['B'], ['A']),
    #     FunctionalDependency(['C'], ['D']),
    # ]
    # distinct_chase_case1 = SQLDatabaseArchitect(R)
    # distinct_chase_case1.distinct_chase_algorithm(R, RS, dependencies)

    pass

if __name__ == '__main__':
    main()