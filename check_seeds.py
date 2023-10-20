import os
import json
import sys
import shutil

seeds = [2119213981, 1608860012, 1021032354, 280853612, 1415121920, 503407898, 995043888, 333388907, 1971069637, 1335198443, 285161167, 894408494, 952170761, 704127742, 168220153, 48936849, 1822305184, 1550130155, 812730049, 833357148, 1043290698, 369867697, 1119789429, 495194068, 806185573, 980810461, 1323666201, 1112576223, 33383858, 735190115, 2114747825, 153301904, 1417633242, 572670284, 71283607, 545220238, 1708331336, 31319830, 795335164, 698059710, 1298677938, 1248108292, 129243081, 869963795, 1378116027, 73798405, 1729011228, 1539271366, 999822958, 1251819451]

seeds_found = {}
duplicate_seeds = []

move_things = False
directory = "runs/optimal_svhn_Adam_fb"
if len(sys.argv) > 1:
   directory = sys.argv[1]

def find_seed_indices(to_find):
   indices = []
   for seed in to_find:
      indices.append(seeds.index(seed))
   indices.sort()
   return indices


counter = 0
for (root, dirs, files) in os.walk(directory):
   if counter == 0:
      partial_dir = os.path.join(root, "partial")
      if not os.path.isdir(partial_dir) and move_things:
         os.mkdir(partial_dir)
   if "partial" in root and "_partial" not in root:
      continue
   for file in files:
      if "config" in file:
         f = open(os.path.join(root, file))
         config = json.load(f)
         seed = config['seed']
         if seed in seeds_found.keys():
            duplicate_seeds.append(seed)
            # Move to partial directory
            # Check there are only 2 files in the directory, as we expect
            if len(files) > 2:
               print("WARNING: {} has multiple runs in a single timestamp".format(root))
               quit()
            if move_things:
               root_name = root.split('/')[-1]
               shutil.move(root, os.path.join(partial_dir, root_name))
            
         else:
            seeds_found[seed] = root
   counter += 1

seeds_set = set(seeds)
seeds_found_set = set(seeds_found.keys())
missing_seeds = list(seeds_set.difference(seeds_found_set))
missing_indices = find_seed_indices(missing_seeds)
duplicate_indices = find_seed_indices(duplicate_seeds)
found_indices = find_seed_indices(seeds_found.keys())
print("Indices found {}: \t({})".format(found_indices, len(found_indices)))
print("Indices missing {}: \t({})".format(missing_indices, len(missing_indices)))
print("Indices duplicated {}".format(duplicate_indices))
print("Num duplicates: {}".format(len(duplicate_indices)))
