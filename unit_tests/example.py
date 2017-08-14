import os
import shutil
import subprocess

runinfo_fname = os.path.abspath('./example.runinfo')

output_directory = os.path.abspath('./runfiles/output/')
if os.path.isdir(output_directory):
    shutil.rmtree(output_directory, ignore_errors=True)
print('making {}'.format(output_directory))
os.mkdir(output_directory)
print('Directory exists: {}'.format(os.path.isdir(output_directory)))

path_to_trainpredict = os.path.abspath('../build/src/cpp/score_shuffle')
verbose_fname = os.path.join(output_directory, 'example.verbose')
done_file = os.path.join(output_directory, 'example.done')
cmd = "rm -f %s; %s %s >> %s; echo Done! >> %s" % (done_file, path_to_trainpredict, runinfo_fname, verbose_fname, done_file)
process_id = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
out, err = process_id.communicate()
print('Directory exists: {}'.format(os.path.isdir(output_directory)))
if err or err2 or not os.path.isfile(done_file):
    print('Error: while executing command:\n{}'.format(cmd))
    print(err)
else:
    print('Done!')
print('Directory exists: {}'.format(os.path.isdir(output_directory)))
