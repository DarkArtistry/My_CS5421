from the_sql_architect import Dependency, FunctionalDependency, MultiValueDependency, SQLDatabaseArchitect


# ALL MIN COVERS TEST
# R = ['A', 'B', 'C', 'D', 'E', 'F']
# FD = [[['B','C'] , ['A']], [['F'] , ['D','E']], [['B', 'C', 'F'], ['D', 'E' ,'F',]]]
# all_min_covers_case1 = SQLDatabaseArchitect(R)
# all_min_covers_case1_fds = all_min_covers_case1.list_to_functional_dependencies(FD)
# all_min_covers_1 = all_min_covers_case1.general_fd_all_min_covers(all_min_covers_case1_fds)
# print('all_min_covers_1 : {}'.format(all_min_covers_1))

# BCNF DECOMPOSITION TEST
R = ['A', 'B', 'C', 'D']
FD = [[['A','B'] , ['C']], [['C'] , ['D']], [['D'], ['B']]]
bcnf_case1 =  SQLDatabaseArchitect(R, FD)
# bcnf_case1_fds = bcnf_case1.list_to_functional_dependencies(FD)
# results = bcnf_case1.bcnf_decomposition(R, bcnf_case1_fds)
# print('bcnf')
# results = [[decom[0], [dep.convert_to_list() for dep in decom[1]]] for decom in results]
# print(results)
# min_covers_case1_fds = bcnf_case1.list_to_functional_dependencies(FD) # <<<<<
# bcnf_case1.generate_all_fd_closures(min_covers_case1_fds) # <<<<<
# print(bcnf_case1.is_bcnf(min_covers_case1_fds)) # <<<<<
# print(bcnf_case1.candidate_keys)

# SYNTHESIS
# R = ['A', 'B', 'C', 'D', 'E']
# FD = [[['B'], ['A']], [['C', 'D'], ['E']]]
synthesis_case1 =  SQLDatabaseArchitect(R, FD)
synthesis_case1_fds = synthesis_case1.list_to_functional_dependencies(FD)
response = synthesis_case1.sythesis_3nf_normalisation(R, synthesis_case1_fds)
print('synthesis')
print('results')
results = [[result[0], [dep.convert_to_list() for dep in result[1]]] for result in response['results']]
# print(results)
results = [[['C', 'A', 'B'], [[['A', 'B'], ['C']]]], [['C', 'D'], [[['C'], ['D']]]], [['D', 'B'], [[['D'], ['B']]]]]
synthesis_case1_fds = synthesis_case1.list_to_functional_dependencies(results)
print(synthesis_case1.is_bcnf(synthesis_case1_fds))

# min_covers_case1_fds = synthesis_case1.list_to_functional_dependencies(FD) # <<<<<
# synthesis_case1.generate_all_fd_closures(min_covers_case1_fds) # <<<<<
# print(synthesis_case1.is_3nf(min_covers_case1_fds)) # <<<<<<

# print('fragments')
# print(response['fragment_only_results'])
# FDS = [[['B'], ['A']], [['C', 'D'], ['E']]]
# new_fds_class = synthesis_case1.list_to_functional_dependencies(FDS)
# is_bcnf = synthesis_case1.is_bcnf(new_fds_class)
# print('is_bcnf {}'.format(is_bcnf))


# # COMPACT MIN COVER TEST
# R = ['A', 'C', 'E']
# FD = [[['A', 'C'], ['B', 'D', 'E']], [['B'], ['A']]]
# compact_min_cover_case1 = SQLDatabaseArchitect(R)
# min_covers_case1 = SQLDatabaseArchitect(R)
# min_covers_case1_fds = min_covers_case1.list_to_functional_dependencies(FD)
# min_covers_1 = min_covers_case1.generate_fd_min_covers(min_covers_case1_fds)
# print('min_covers_1 : {}'.format(min_covers_1))
# min_covers = [[['B'], ['A']], [['C', 'D'], ['E']]]
# covers = min_covers_case1.list_to_functional_dependencies(min_covers)
# test_results = compact_min_cover_case1.generate_compact_minimal_cover(covers)
# print(test_results)


# DISTINGUISHED CHASE ALGORITHM TEST 1
# R = ['A', 'B', 'C', 'D', 'E']
# RS = [
#     ['A', 'B', 'C'],
#     ['B', 'C' ,'D'],
#     ['A', 'C', 'E'],
# ]
# FD = [[['A', 'C'], ['B', 'D', 'E']], [['B'], ['A']]]

# dependencies = [
#     FunctionalDependency(['B'], ['A']),
#     FunctionalDependency(['C', 'D'], ['E']),
# ]

# distinct_chase_case1 = SQLDatabaseArchitect(R)
# distinct_chase_case1.distinct_chase_algorithm(R, RS, dependencies)


# CHASE ALGORITHM TEST 2
# R = ['A', 'B', 'C', 'D', 'E']
# FD = [[['A', 'B'], ['C']], [['C'], ['D', 'E']], [['E'], ['D']], [['F'], ['G']] ]
# R1 = ['A', 'B', 'C', 'D', 'E'] # X
# R2 = ['A', 'B', 'F', 'G'] # Y
# dependencies = [
#     MultiValueDependency(['A', 'B'], ['C']),
#     FunctionalDependency(['A', 'B'], ['D']),
# ]

# chase1 = MultiValueDependency(['A', 'B'], ['C', 'D'])
# chase_case2 = SQLDatabaseArchitect(R)
# results = chase_case2.chase_algorithm(R, dependencies, chase1)
# print('results: {}'.format(results))
# mvd_is_satisfied = chase_case2.mvd_is_satisfied(R, chase1 ,results)
# print('mvd_is_satisfied : {}'.format(mvd_is_satisfied))