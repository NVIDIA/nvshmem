#!/usr/bin/env python3
import sys
import os
import re
import time
import signal
import codecs

from subprocess import Popen, PIPE
from threading import Thread

test_process = 0
failed_binary_cmdlines_list = []
NVSHMEM_LAUNCHER = 0
MPI_LAUNCHER = 1
SHMEM_LAUNCHER = 2

# Set by perftestRunner for --sleep; when unset, PERF_TEST_SLEEP_SEC in the environment is used.
class _SleepAfterTestUnset:
  pass


_SLEEP_AFTER_TEST_UNSET = _SleepAfterTestUnset()
_sleep_after_test_sec_override = _SLEEP_AFTER_TEST_UNSET


def configure_sleep_after_each_test(seconds):
  """
  Runner-only: delay in seconds between test cases. Non-negative float.
  Replaces setting PERF_TEST_SLEEP_SEC in the environment for this process.
  """
  global _sleep_after_test_sec_override
  _sleep_after_test_sec_override = max(0.0, float(seconds))


def _parse_perf_test_sleep_sec_from_env():
  """Return non-negative float from PERF_TEST_SLEEP_SEC, or 0 if unset/invalid."""
  sec = os.environ.get("PERF_TEST_SLEEP_SEC")
  if sec is None:
    return 0.0
  try:
    t = float(sec)
    if t < 0:
      print(
        "perftestCommon: PERF_TEST_SLEEP_SEC %r is negative; using 0." % (sec,),
        file=sys.stderr,
      )
      return 0.0
    return t
  except (TypeError, ValueError):
    print(
      "perftestCommon: invalid PERF_TEST_SLEEP_SEC value %r; using 0." % (sec,),
      file=sys.stderr,
    )
    return 0.0


def _resolved_sleep_after_test_sec():
  if _sleep_after_test_sec_override is not _SLEEP_AFTER_TEST_UNSET:
    return _sleep_after_test_sec_override
  return _parse_perf_test_sleep_sec_from_env()


def to_bytes(s):
  if type(s) is bytes:
    return s
  elif type(s) is str or (sys.version_info[0] < 3 and type(s) is unicode):
    return codecs.encode(s, 'utf-8')
  else:
    raise TypeError("Expected bytes or string, but got %s." % type(s))

def display_time(func):
  def wrapper(*args):
    t1 = time.time()
    req = func(*args)
    t2 = time.time()
    print('Total time {:.4}s'.format(t2 - t1))
    return req
  return wrapper

def report_failure(cmd_line, test_path, ftesto, fteste):
  global failed_binary_cmdlines_list
  Popen(['echo', ' '.join([str(elem) for elem in cmd_line]) + ' failed\r\n'], stdout=fteste)
  failed_binary_cmdlines_list.append((test_path, str(cmd_line)))
  return

def get_all_tests(ftestlist):
  tests_set = []
  skipped_tests_set = []
  with open(ftestlist, 'r') as f:
    for line in f:
      if line.startswith("#"):
        skipped_tests_set.append(line[1:-1].strip())
        tests_set.append(line[1:-1].strip())
      else:
        tests_set.append(line.strip())
  return (tests_set, skipped_tests_set)

def get_args_combinations_pe_range(full_test_path, npe_start_end_step, max_pes):
  args_combs = []
  npe_range_ = list(npe_start_end_step)
  npe_range = [npe for npe in npe_range_ if npe <= max_pes]

  if len(npe_range) == 0:
    return (args_combs, npe_range)

  full_args_path = full_test_path + '.args'
  if 'pt-to-pt' in full_test_path:
    npe_range[0] = 2
    if 1 < len(npe_range):
      elemsDelCnt = len(npe_range) - 1
      for cnt in range(0, elemsDelCnt):
        del npe_range[-1]
    #TODO : delete the test of this def
    if not os.path.isfile(full_args_path):
      return (args_combs, npe_range)
    else:
      print(full_args_path)
    with open(full_args_path) as f:
      lines = f.readlines()
      for i in range(0, len(lines)):
        if lines[i]:
          print("Add parameters: %s" % lines[i])
          args_combs.append(lines[i])
    return (args_combs, npe_range)

  if not os.path.isfile(full_args_path):
    return (args_combs, npe_range)
  else:
    print(full_args_path)

  with open(full_args_path) as f:
    lines = f.readlines()
    # f.seek(0)
    for i in range(0,len(lines)):
      print("Add parameters: %s" % lines[i])
      if lines[i]:
        args_combs.append(lines[i])

  return (args_combs, npe_range)

def get_env_combinations(full_test_path):
  envs = []
  env_combs = []
  full_env_path = full_test_path+'.env'
  if not os.path.isfile(full_env_path):
    return env_combs 
  with open(full_env_path) as f:
    for line in f:
      envs.append(line)
  for e in envs:
    env_combs.append(e.split())
  return env_combs

def show_table_partial_data_only(data):
  """
  Prints the first data row and the last data row of each table and table's header lines in the provided data(output).

  Parameters:
  data (str): A string containing one or more text-based tables.
  """
  lines = data.split("\n")

  env_value = os.getenv('NVSHMEM_MACHINE_READABLE_OUTPUT')
  no_new_program = True
  
  i = 0
  total_lines = len(lines)

  if env_value == '1':
    # NVSHMEM_MACHINE_READABLE_OUTPUT is 1
    table_sep_pattern = r'^&&&&'
    separator = re.compile(table_sep_pattern)
    result_pattern = r'^&&&& PERF\s(\w+?)__+(.+?)_size\D+(\d+)_+(\w+)\s+(\S+)\s+(\S+)$'
    result_line = re.compile(result_pattern)

    while i < total_lines:
      no_new_program = False
      found_first_report = False

      while i < total_lines and not separator.match(lines[i]):
        if found_first_report and not no_new_program and len(lines[i]) > 0 and lines[i] != '\n':
          no_new_program = True
          break
        i += 1
        continue

      if i >= total_lines or no_new_program:
        continue

      perf_line = result_line.match(lines[i])
      if perf_line:
        # Only catch the first and the last for each paragraph.
        if "&&&&" not in lines[i - 1] or "&&&&" not in lines[i + 1]:
          print(lines[i])

      if i < total_lines - 1:
          i += 1
      else:
          break
  else:
    # Tables
    table_header1_pattern = r'\|\s*(.+)\s*\|\s*(\w[\w -]*?)\s*\|'
    table_header2_pattern = r'\|\s*([\w-]+)[\s\w\(\)-]*\|\s*([\w-]+)\s*([\w/]+)\s*\|'
    table_content_pattern = r'\|\s*([\d]*)\s*\|\s*([\d.]+)\s*\|'
    table_sep_pattern = r'^\+\-+\+\-+\+$'
    separator = re.compile(table_sep_pattern)
    theader1 = re.compile(table_header1_pattern)
    theader2 = re.compile(table_header2_pattern)
    tcont = re.compile(table_content_pattern)

    while i < total_lines:

      # Maybe include multi tables in one output.
      no_new_program = False
      found_first_report = False

      while i < total_lines and not separator.match(lines[i]):

        if found_first_report and not no_new_program and len(lines[i]) > 0 and lines[i] != '\n':
          no_new_program = True
          break
        i += 1
        continue

      if i >= total_lines or no_new_program:
        continue

      found_first_report = True
      i += 1
      th1 = theader1.match(lines[i])
      print(lines[i - 1])
      print(lines[i])
      i += 2
      th2 = theader2.match(lines[i])
      print(lines[i - 1])
      print(lines[i])
      i += 2

      if not th1 or not th2:
        continue

      data = []
      content = tcont.match(lines[i])
      while content is not None:
        if float(content.group(2)) > 0.0:
          data.append((content.group(1), content.group(2)))
          if len(data) == 1:
            print(lines[i])
        i += 2
        content = tcont.match(lines[i])
      
      if len(data) != 1:
        print(lines[i-2])
        print(lines[i-1])
      else:
        print(lines[i-1])

      print("")
      i += 1

def thread_func(cmd_line, ftesto, fteste):
  global test_process
  cmd_line_str = ' '.join([str(elem) for elem in cmd_line])
  print(cmd_line_str)
  # test_process = Popen(['echo', 'Running ' + ' '.join([str(elem) for elem in cmd_line]) + '\r\n'], stdout=ftesto)
  fteste.write('Running ' + ' '.join([str(elem) for elem in cmd_line]) + '\r\n')
  fteste.flush()
  ftesto.write('Running ' + ' '.join([str(elem) for elem in cmd_line]) + '\r\n')
  ftesto.flush()
  try:
    # Run the command and capture stdout and stderr
    command_line_list = []
    command_line_list.append(cmd_line_str)
    test_process = Popen(command_line_list, stdout=PIPE, stderr=PIPE, shell=True, preexec_fn=os.setsid)
    stdout_data, stderr_data = test_process.communicate()

    # Write the stderr and stdout data to the respective files
    fteste.write(stderr_data.decode('utf-8'))
    fteste.flush()
    ftesto.write(stdout_data.decode('utf-8'))
    ftesto.flush()

    # Optionally print stdout data if SHOW_PERF_DATA is set to "Yes"
    show_perf_data = os.environ.get('SHOW_PERF_DATA', 'No')
    if show_perf_data == "Yes":
        try:
          show_table_partial_data_only(stdout_data.decode('utf-8'))
        except Exception as parse_err:
          print("Warning: failed to parse perf output: %s" % str(parse_err))

    test_process.stderr_data = stderr_data
    test_process.stdout_data = stdout_data
    return test_process.returncode
  except Exception as err:
    print(str(err))
    return 254

def _sleep_between_tests_if_configured():
  """Sleep after each run_cmd (between consecutive 'Running ...' lines) per runner or PERF_TEST_SLEEP_SEC."""
  t = _resolved_sleep_after_test_sec()
  if t > 0:
    time.sleep(t)

_pause_file_notice_printed = False

def _wait_if_pause_file():
  """
  Block before starting the next test case while a sentinel file exists.
  Does not run during the benchmark binary; only between runner invocations,
  so per-case performance numbers are unaffected.
  Path: NVSHMEM_PAUSE_FILE (default /tmp/NVSHMEM-PAUSE).
  Poll interval: QA_NVSHMEM_PAUSE_POLL_SEC (default 0.5 seconds).
  """
  global _pause_file_notice_printed
  pause_path = os.environ.get("NVSHMEM_PAUSE_FILE", "/tmp/NVSHMEM-PAUSE")
  try:
    poll = float(os.environ.get("QA_NVSHMEM_PAUSE_POLL_SEC", "0.5"))
  except ValueError:
    poll = 0.5
  if poll <= 0:
    poll = 0.5

  while os.path.exists(pause_path):
    if not _pause_file_notice_printed:
      print("NVSHMEM perftest paused: remove '%s' to continue (runner idle, benchmark not started)." % pause_path, flush=True)
      _pause_file_notice_printed = True
    time.sleep(poll)
  _pause_file_notice_printed = False

@display_time
def run_cmd(cmd_line, test_path, timeout, ftesto, fteste):
  try:
    _wait_if_pause_file()

    if 'PERF_TEST_CMD_AHEAD' in os.environ:
      cmd_line = os.environ['PERF_TEST_CMD_AHEAD'].split() + cmd_line

    if 'PERF_TEST_CMD_LAST' in os.environ:
      cmd_line = cmd_line + os.environ['PERF_TEST_CMD_LAST'].split()

    th = Thread(target=thread_func, args=(cmd_line, ftesto, fteste))
    th.start()
    th.join(timeout)

    if th.is_alive():
      # Popen(['echo', 'Timed out ' + ' '.join([str(elem) for elem in cmd_line]) + '\r\n'], stdout=fteste)
      fteste.write('Timed out ' + ' '.join([str(elem) for elem in cmd_line]) + '\r\n')
      fteste.flush()
      print("Timed out " + ' '.join([str(elem) for elem in cmd_line]))
      os.killpg(os.getpgid(test_process.pid), signal.SIGTERM)
      # test_process.terminate()
      th.join()
      report_failure(cmd_line, test_path, ftesto, fteste)
      return

    if test_process.returncode:
      if hasattr(test_process, 'stderr_data'):
        print(test_process.stderr_data.decode('utf-8'))
      p = Popen(['echo', 'EXPECTING PASSED, GOT FAILURE'], stdout=PIPE)
      print(to_bytes(p.communicate()[0]).decode('utf-8'))
      report_failure(cmd_line, test_path, ftesto, fteste)
    else:
      p = Popen(['echo', 'PASSED'], stdout=PIPE)
      print(to_bytes(p.communicate()[0]).decode('utf-8'))

    cmd = 'rm'
    args = "%s*" % '/dev/shm/nvshmem-shm'
    Popen("%s %s" % (cmd, args), shell=True, stdin=PIPE, stdout=PIPE, stderr=PIPE, close_fds=True)
  finally:
    _sleep_between_tests_if_configured()

def run_cmd_given_pes(cmd_line_prefix, cmd_line_suffix, test_install_path, full_test_path, npe_all, ppn, timeout, launcher_choice, ftesto, fteste):
  cmd_line = cmd_line_prefix[:]
  cmd_line.append(str(npe_all))
  if launcher_choice == NVSHMEM_LAUNCHER:
    cmd_line.append('-ppn')
  elif launcher_choice == MPI_LAUNCHER or launcher_choice == SHMEM_LAUNCHER:
    cmd_line.append('-npernode')
  cmd_line.append(str(ppn))

  bind_scr = os.environ.get("GPUBIND_SCRIPT")
  if bind_scr != "" and bind_scr is not None:
    cmd_line.append("%s" % bind_scr)

  cmd_line.append(full_test_path)
  if cmd_line_suffix:
    cmd_line.append(cmd_line_suffix)
  run_cmd(cmd_line, full_test_path.replace(test_install_path, ''), timeout, ftesto, fteste)

def run_cmd_vary_pes(cmd_line_prefix, cmd_line_suffix, test_install_path, full_test_path, npe_range, nhosts, timeout, launcher_choice, ftesto, fteste):
  cmd_line = cmd_line_prefix[:]
  if 'pt-to-pt' in full_test_path:
    if nhosts == 1:
      ppn = 2
      npe_all = 2
    else:
      ppn = 1
      npe_all = 2
    run_cmd_given_pes(cmd_line_prefix, cmd_line_suffix, test_install_path, full_test_path, npe_all, ppn, timeout, launcher_choice, ftesto, fteste)
  else:
    if nhosts == 1:
      npe_range_ = npe_range[1:]
    else:
      npe_range_ = npe_range
    for npe in npe_range_:
      ppn = npe
      npe_all = nhosts*ppn
      run_cmd_given_pes(cmd_line_prefix, cmd_line_suffix, test_install_path, full_test_path, npe_all, ppn, timeout, launcher_choice, ftesto, fteste)
  return

def enumerate_env_lines(env_combs, cmd_line_suffix, nvshmem_install_path, test_install_path, full_test_path, npe_range, hosts, timeout, launcher_choice, mpi_install_path, extra_parameters_string, ftesto, fteste):
  nhosts = hosts.count(",")+1
  if 'CUDA_HOME' in os.environ:
    cuda_install_path = os.environ['CUDA_HOME']
  else:
    print('CUDA_HOME not set. Try to use the default value: /usr/local/cuda')
    cuda_install_path = '/usr/local/cuda'

  if 'GDRCOPY_HOME' in os.environ:
    gdrcopy_install_path = "%s/lib:%s/lib64" % (os.environ['GDRCOPY_HOME'], os.environ['GDRCOPY_HOME'])
  else:
    gdrcopy_install_path = ""

  if 'NCCL_HOME' in os.environ:
    nccl_install_lib = ":%s/lib64:%s/lib" % (os.environ['NCCL_HOME'], os.environ['NCCL_HOME'])
  else:
    nccl_install_lib = ""

  if 'PMIX_HOME' in os.environ:
    pmix_install_lib = ":%s/lib" % os.environ['PMIX_HOME']
  else:
    pmix_install_lib = ""
  mpi_install_lib = ":%s/lib:%s/lib64" % (mpi_install_path, mpi_install_path)

  if 'QA_BOOTSTRAP' in os.environ:
    QA_BOOTSTRAP = os.environ['QA_BOOTSTRAP']
  else:
    QA_BOOTSTRAP = "pmi"

  if 'QA_BIND_TO' in os.environ:
    QA_BIND_TO = os.environ['QA_BIND_TO']
  else:
    QA_BIND_TO = "socket"

  if QA_BOOTSTRAP == "uid":
    bootstrap_str = "NVSHMEMTEST_USE_UID_BOOTSTRAP=1"
  elif QA_BOOTSTRAP == "mpi":
    bootstrap_str = "NVSHMEM_BOOTSTRAP=MPI"
  else:
    bootstrap_str = "NVSHMEMTEST_USE_MPI_LAUNCHER=1"

  if 'PERF_TEST_MPIRUN_EXTRA' in os.environ:
    MPIRUN_EXTRA = os.environ['PERF_TEST_MPIRUN_EXTRA']
    MPIRUN_EXTRA_LIST = MPIRUN_EXTRA.split()
  else:
    MPIRUN_EXTRA = ""
    MPIRUN_EXTRA_LIST = []

  if env_combs:
    for combidx in range(0, len(env_combs)):
      if launcher_choice == NVSHMEM_LAUNCHER:
        cmd_line_prefix = [nvshmem_install_path+'/bin/nvshmrun.hydra', '--bind-to', QA_BIND_TO, '--launcher', 'ssh', '--hosts', hosts]
        extra_parameters = extra_parameters_string.split()
        first_e = cmd_line_prefix.index("--launcher")
        for item in extra_parameters[::-1]:
          cmd_line_prefix.insert(first_e, "-genv=%s" % item)

        for envidx in range(0, len(env_combs[combidx]), 2):
          var = env_combs[combidx][envidx]
          val = env_combs[combidx][envidx + 1]
          cmd_line_prefix.append('-genv')
          cmd_line_prefix.append(var)
          cmd_line_prefix.append(val)
        cmd_line_prefix.append('-n')
        run_cmd_vary_pes(cmd_line_prefix, cmd_line_suffix, test_install_path, full_test_path, npe_range, nhosts, timeout, launcher_choice, ftesto, fteste)
      if launcher_choice == MPI_LAUNCHER:
        cmd_line_prefix = [mpi_install_path+'/bin/mpirun', '--mca', 'btl', '^uct', '--allow-run-as-root', '-oversubscribe', '--bind-to', QA_BIND_TO, '-x', 'LD_LIBRARY_PATH='+cuda_install_path+'/lib64:'+gdrcopy_install_path+nccl_install_lib+pmix_install_lib+':'+nvshmem_install_path+'/lib'+mpi_install_lib+':$LD_LIBRARY_PATH', '-x', bootstrap_str , '--host', hosts]
        cmd_line_prefix[1:1] = MPIRUN_EXTRA_LIST
        extra_parameters = extra_parameters_string.split()
        first_x = cmd_line_prefix.index("-x")
        for item in extra_parameters[::-1]:
          cmd_line_prefix.insert(first_x, item)
          cmd_line_prefix.insert(first_x, "-x")

        for envidx in range(0, len(env_combs[combidx]), 2):
          var = env_combs[combidx][envidx]
          val = env_combs[combidx][envidx + 1]
          cmd_line_prefix.append('-x')
          cmd_line_prefix.append(var+'='+val)
        cmd_line_prefix.append('-n')
        run_cmd_vary_pes(cmd_line_prefix, cmd_line_suffix, test_install_path, full_test_path, npe_range, nhosts, timeout, launcher_choice, ftesto, fteste)
      if launcher_choice == SHMEM_LAUNCHER:
        cmd_line_prefix = [mpi_install_path+'/bin/oshrun', '--mca', 'btl', '^uct',  '--allow-run-as-root', '-oversubscribe', '--bind-to', QA_BIND_TO, '-x', 'LD_LIBRARY_PATH='+cuda_install_path+'/lib64:'+gdrcopy_install_path+nccl_install_lib+pmix_install_lib+':'+nvshmem_install_path+'/lib'+mpi_install_lib+':$LD_LIBRARY_PATH', '-x', 'NVSHMEMTEST_USE_SHMEM_LAUNCHER=1' , '--host', hosts]
        extra_parameters = extra_parameters_string.split()
        first_x = cmd_line_prefix.index("-x")
        for item in extra_parameters[::-1]:
          cmd_line_prefix.insert(first_x, item)
          cmd_line_prefix.insert(first_x, "-x")        
        for envidx in range(0, len(env_combs[combidx]), 2):
          var = env_combs[combidx][envidx]
          val = env_combs[combidx][envidx + 1]
          cmd_line_prefix.append('-x')
          cmd_line_prefix.append(var+'='+val)
        cmd_line_prefix.append('-n')
        run_cmd_vary_pes(cmd_line_prefix, cmd_line_suffix, test_install_path, full_test_path, npe_range, nhosts, timeout, launcher_choice, ftesto, fteste)
  else:
    if launcher_choice == NVSHMEM_LAUNCHER:
      cmd_line_prefix = [nvshmem_install_path+'/bin/nvshmrun.hydra', '--bind-to', QA_BIND_TO, '--launcher', 'ssh', '--hosts', hosts, '-n']
      extra_parameters = extra_parameters_string.split()
      first_e = cmd_line_prefix.index("--launcher")
      for item in extra_parameters[::-1]:
        cmd_line_prefix.insert(first_e, "-genv=%s" % item)
      run_cmd_vary_pes(cmd_line_prefix, cmd_line_suffix, test_install_path, full_test_path, npe_range, nhosts, timeout, launcher_choice, ftesto, fteste)
    if launcher_choice == MPI_LAUNCHER:
      cmd_line_prefix = [mpi_install_path+'/bin/mpirun', '--mca', 'btl', '^uct', '--allow-run-as-root', '-oversubscribe', '--bind-to', QA_BIND_TO, '-x', 'LD_LIBRARY_PATH='+cuda_install_path+'/lib64:'+gdrcopy_install_path+nccl_install_lib+pmix_install_lib+':'+nvshmem_install_path+'/lib'+mpi_install_lib+':$LD_LIBRARY_PATH', '-x', bootstrap_str, '--host', hosts, '-n']
      cmd_line_prefix[1:1] = MPIRUN_EXTRA_LIST
      extra_parameters = extra_parameters_string.split()
      first_x = cmd_line_prefix.index("-x")
      for item in extra_parameters[::-1]:
        cmd_line_prefix.insert(first_x, item)
        cmd_line_prefix.insert(first_x, "-x")
      run_cmd_vary_pes(cmd_line_prefix, cmd_line_suffix, test_install_path, full_test_path, npe_range, nhosts, timeout, launcher_choice, ftesto, fteste)
    if launcher_choice == SHMEM_LAUNCHER:
      cmd_line_prefix = [mpi_install_path+'/bin/oshrun', '--mca', 'btl', '^uct', '--allow-run-as-root', '-oversubscribe', '--bind-to', QA_BIND_TO, '-x', 'LD_LIBRARY_PATH='+cuda_install_path+'/lib64:'+gdrcopy_install_path+nccl_install_lib+pmix_install_lib+':'+nvshmem_install_path+'/lib'+mpi_install_lib+':$LD_LIBRARY_PATH', '-x', 'NVSHMEMTEST_USE_SHMEM_LAUNCHER=1', '--host', hosts, '-n']
      extra_parameters = extra_parameters_string.split()
      first_x = cmd_line_prefix.index("-x")
      for item in extra_parameters[::-1]:
        cmd_line_prefix.insert(first_x, item)
        cmd_line_prefix.insert(first_x, "-x")
      run_cmd_vary_pes(cmd_line_prefix, cmd_line_suffix, test_install_path, full_test_path, npe_range, nhosts, timeout, launcher_choice, ftesto, fteste)
  return

def enumerate_args_lines(args_combs, env_combs, nvshmem_install_path, test_install_path, full_test_path, npe_range, hosts, timeout, launcher_choice, mpi_install_path, extra_parameters_string, ftesto, fteste):
  if args_combs:
    for args in args_combs:
      cmd_line_suffix = args.rstrip()
      enumerate_env_lines(env_combs, cmd_line_suffix, nvshmem_install_path, test_install_path, full_test_path, npe_range, hosts, timeout, launcher_choice, mpi_install_path, extra_parameters_string, ftesto, fteste)
  else:
    enumerate_env_lines(env_combs, '', nvshmem_install_path, test_install_path, full_test_path, npe_range, hosts, timeout, launcher_choice, mpi_install_path, extra_parameters_string, ftesto, fteste)
  return

def walk_dir_on_set(nvshmem_install_path, test_install_path, npe_start_end_step, max_pes, hosts, timeout, enable_skip, tests_set, skipped_tests_set, launcher_choice, mpi_install_path, extra_parameters_string, ftesto, fteste):
  for test_path in tests_set:
    full_test_path = os.path.join(test_install_path, test_path.lstrip(os.path.sep))
    if enable_skip and (test_path in skipped_tests_set):
      Popen(['echo', (full_test_path)+' found in list and skipped\r\n'], stdout=ftesto)
      continue
    if not os.access(full_test_path, os.X_OK):
      Popen(['echo', (full_test_path)+' found in list and binary missing\r\n'], stdout=fteste)
      continue
    env_combs = get_env_combinations(full_test_path)
    tup = get_args_combinations_pe_range(full_test_path, npe_start_end_step, max_pes)
    enumerate_args_lines(tup[0], env_combs, nvshmem_install_path, test_install_path, full_test_path, tup[1], hosts, timeout, launcher_choice, mpi_install_path, extra_parameters_string, ftesto, fteste)
  return

def walk_dir(nvshmem_install_path, mpi_install_path, test_install_path, launcher_choice, npe_start_end_step, max_pes, hosts, timeout, enable_skip, ftestlist_any_launcher, extra_parameters_string, ftesto, fteste):
  stup = get_all_tests(ftestlist_any_launcher)
  if len(stup[0]) != 0:
    if launcher_choice == 1:
      walk_dir_on_set(nvshmem_install_path, test_install_path, npe_start_end_step, max_pes, hosts, timeout, enable_skip, stup[0], stup[1], NVSHMEM_LAUNCHER, mpi_install_path, extra_parameters_string, ftesto, fteste)
      walk_dir_on_set(nvshmem_install_path, test_install_path, npe_start_end_step, max_pes, hosts, timeout, enable_skip, stup[0], stup[1], MPI_LAUNCHER, mpi_install_path, extra_parameters_string, ftesto, fteste)
      walk_dir_on_set(nvshmem_install_path, test_install_path, npe_start_end_step, max_pes, hosts, timeout, enable_skip, stup[0], stup[1], SHMEM_LAUNCHER, mpi_install_path, extra_parameters_string, ftesto, fteste)
    elif launcher_choice == 2:
      walk_dir_on_set(nvshmem_install_path, test_install_path, npe_start_end_step, max_pes, hosts, timeout, enable_skip, stup[0], stup[1], SHMEM_LAUNCHER, mpi_install_path, extra_parameters_string, ftesto, fteste)
    elif launcher_choice == 3:
      walk_dir_on_set(nvshmem_install_path, test_install_path, npe_start_end_step, max_pes, hosts, timeout, enable_skip, stup[0], stup[1], NVSHMEM_LAUNCHER, mpi_install_path, extra_parameters_string, ftesto, fteste)
    elif launcher_choice == 0:
      walk_dir_on_set(nvshmem_install_path, test_install_path, npe_start_end_step, max_pes, hosts, timeout, enable_skip, stup[0], stup[1], MPI_LAUNCHER, mpi_install_path, extra_parameters_string, ftesto, fteste)
    else:
      print("Please select launcher use 0/1/2/3. [1: Three launchers, 0: mpirun, 2: openshmem, 3: nvshmem]")
  return
