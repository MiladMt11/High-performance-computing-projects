#!/bin/python3

num_threads = [1,2,4,8,16,24]
prob_sizes = [10, 30, 50, 70, 100, 150, 200]
# 'poisson_j_omp_v0', TODO: running
versions = ['poisson_j_omp_v0','poisson_j_omp_v1']

psv = 'PROBSIZE' 
tnv = 'NTHREADS'
exv = 'EXEVAR'

# # serial
# with open('omp_jacobi_simple.sub', 'r') as file :
#   filedata = file.read()
#   exe_lines = ''
#   for ps in prob_sizes:
#     # Replace the target string
#     filedata = filedata.replace(psv, str(ps))
#     filedata = filedata.replace(tnv, str(1))
#     filedata = filedata.replace(exv, 'jacobi_serial')

#     exe_lines += f'./$EXECUTABLE {ps} $EXECOPTS\n'

#   file_name_root = f'jacobi_serial'
#   filedata = filedata.replace('OUTFILENAME', file_name_root + '.out')
#   filedata = filedata.replace('ERRFILENAME', file_name_root + '.err')
#   filedata = filedata.replace('./$EXECUTABLE $EXECOPTS', exe_lines)
#   # Write the file out again
#   with open(f'experiments/jacobi_serial/{file_name_root}.sub', 'w') as file:
#     file.write(filedata)

# middle versions
for v in versions:
  for nth in num_threads:
    with open('omp_jacobi_simple.sub', 'r') as file:
      filedata = file.read()
      exe_lines = ''
      for ps in prob_sizes:
        # Replace the target string
        filedata = filedata.replace(psv, str(ps))
        filedata = filedata.replace(tnv, str(nth))
        filedata = filedata.replace(exv, v)

        exe_lines += f'./$EXECUTABLE {ps} $EXECOPTS\n'

      file_name_root = f'{v}_{nth}'
      filedata = filedata.replace('OUTFILENAME', file_name_root + '.out')
      filedata = filedata.replace('ERRFILENAME', file_name_root + '.err')
      filedata = filedata.replace('./$EXECUTABLE $EXECOPTS', exe_lines)
      # Write the file out again
      with open(f'experiments/{v}/{file_name_root}.sub', 'w') as file:
        file.write(filedata)

# runtime scheduling
schedules = ['static,1','dynamic,1','dynamic,5','dynamic,10','dynamic,20']
v = 'poisson_j_omp_v2'
for sch in schedules:
  for nth in num_threads:
    with open('omp_jacobi_simple.sub', 'r') as file:
      filedata = file.read()
      exe_lines = ''
      for ps in prob_sizes:
        # Replace the target string
        filedata = filedata.replace(psv, str(ps))
        filedata = filedata.replace(tnv, str(nth))
        filedata = filedata.replace(exv, v)
        filedata = filedata.replace('export OMP_SCHEDULE=dynamic,50', f'export OMP_SCHEDULE={sch}')

        exe_lines += f'./$EXECUTABLE {ps} $EXECOPTS\n'
      
      file_name_root = f'{v}_{nth}_{sch}'
      filedata = filedata.replace('OUTFILENAME', file_name_root + '.out')
      filedata = filedata.replace('ERRFILENAME', file_name_root + '.err')
      filedata = filedata.replace('./$EXECUTABLE $EXECOPTS', exe_lines)
      # Write the file out again
      with open(f'experiments/{v}/{file_name_root}.sub', 'w') as file:
        file.write(filedata)