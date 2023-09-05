import os
import sys
import zipfile
import shutil
import importlib.util

red_text = '\033[91m'
green_text = '\033[92m'
yellow_text = '\033[93m'
blue_text = '\033[94m'

def failwithmessage(message):
    remove_temp_dir()
    print(red_text + message)
    sys.exit(1)

def successwithmessage(message):
    print(green_text + message)
    sys.exit(0)

def warningwithmessage(message):
    print(yellow_text + message)

def infowithmessage(message):
    print(blue_text + message)

def remove_temp_dir():
    cwd = os.getcwd()
    if os.path.exists(os.path.join(cwd, '__temp')):
        shutil.rmtree(os.path.join(cwd, '__temp'))

def get_name_id(file, fileName):
  """
    Input: List[String] of lines in a python file

    Output: Bool indicating if found names and netids and same amount of both
  """
  # get the net ids and names of the students

  # e.g.
  # Names(s): Gavin Fogel, Vivian Nguyen
  # Netid(s): gdf38, vn72

  line0 = file[0].strip()
  line1 = file[1].strip()

  # Check if the first line is a comment
  if line0[0] != "#" or line1[0] != "#":
    failwithmessage("The first two lines of your python files (" + fileName + ") should be single line comments with your names and netids.")
  
  names = line0.split(":")[1].strip() # Gavin Fogel, Vivian Nguyen
  netids = line1.split(":")[1].strip() # gdf38, vn72

  if len(names) == 0:
    failwithmessage("No name(s) found in the first line of your python file " + fileName + ". \nPlease make sure that the first line is a single line comment with your name(s).")
  if len(netids) == 0:
    failwithmessage("No netid(s) found in the second line of your python file " + fileName + ". \nPlease make sure that the second line is a single line comment with your netid(s).")

  # Check that both lines have commas, or neither

  if "," in names and not "," in netids:
     # Two names, one net ID
     failwithmessage("Looks like there are two names in the first line of your python file (" + fileName + "), but only one net ID in the second line. \nPlease make sure that both lines have the same number of names and net IDs.")
    
  if "," not in names and "," in netids:
     failwithmessage("Looks like there is one name in the first line of your python file (" + fileName + "), but two net IDs in the second line. \nPlease make sure that both lines have the same number of names and net IDs.")

  if "," in names:
     # split on comma, ensure neither side is empty
      names = names.split(",")
      if len(names[0].strip()) == 0 or len(names[1].strip()) == 0:
        failwithmessage("Looks like there is a comma in the first line of your python file (" + fileName + "), but at least one of the names is empty. \nPlease make sure that both lines have the same number of names and net IDs.")

  if "," in netids:
     # split on comma, ensure neither side is empty
      netids = netids.split(",")
      if len(netids[0].strip()) == 0 or len(netids[1].strip()) == 0:
        failwithmessage("Looks like there is a comma in the second line of your python file (" + fileName + "), but at least one of the net IDs is empty. \nPlease make sure that both lines have the same number of names and net IDs.")

  return True
      

def load_student_module(module_path):
  """
  Loads the student's module from the given path.
  Input: String, representing the path to the student's module
  Output: Module object, representing the student's module
  """
  spec = importlib.util.spec_from_file_location("student_module", module_path)
  student_module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(student_module)

  return student_module

def check_module_for_functions(module_object, function_names, file_name):
  """
   Checks that the given module object has the given functions. Will stop script
   execution if function name is not found.
   Input:
     module_object: Module object, representing the student's module
     function_names: List[String], representing the names of the functions to
     check for
     file_name: String, representing the name of the file the functions should be in
   Output:
     None
  """
  for function_name in function_names:
    if not hasattr(module_object, function_name):
      failwithmessage("Could not find function '" + function_name + "' in " + file_name + " file. Are you sure it exists?")
    else:
      infowithmessage("Found function '" + function_name + "' in " + file_name + ".")
   

def check_submission():
    cwd = os.getcwd()
    if not os.path.exists(os.path.join(cwd, 'final_submission.zip')):
        failwithmessage("'final_submission.zip' not found in current directory  Are you sure it exists?")
    else:
        infowithmessage("'final_submission.zip' found in current directory.")

    # Make a temp file to extract the zip file to
    if not os.path.exists(os.path.join(cwd, '__temp')):
        os.makedirs(os.path.join(cwd, '__temp'))
    else: 
      # Remove the temp directory if it already exists and all its contents
      remove_temp_dir()
      os.makedirs(os.path.join(cwd, '__temp'))

    zip_ref = zipfile.ZipFile("final_submission.zip", 'r')
    zip_ref.extractall(os.path.join(cwd, '__temp'))
    zip_ref.close()

    python_files = ['models.py', 'helpers.py', 'data_exploration.py', 'viterbi.py', 'validation.py']

    ## CHECK FILES EXIST
    # Look in the temp folder for the python files
    for file in python_files:
        if not os.path.exists(os.path.join(cwd, '__temp', file)):
            remove_temp_dir()
            failwithmessage("Could not find '" + file + "' in your submission. Are you sure it exists?")

    infowithmessage("All python files found in submission.")

    ## VALIDATE PYTHON FILES
    for file in python_files:
      ## CHECK NAMES AND NET IDS ================
      file_lines = open(os.path.join(cwd, '__temp', file), 'r').readlines()
      if get_name_id(file_lines, file):
        infowithmessage("Found names and netids in " + file + ".")
      ## CHECK NAMES AND NET IDS ================

      # TODO: Check the file for the correct imports:

      if (file == 'data_exploration.py'):
        data_exploration = load_student_module(os.path.join(cwd, '__temp', file))
        check_module_for_functions(data_exploration, ['stringify_labeled_doc', 'validate_ner_sequence'], file)

      if (file == 'models.py'):
        models = load_student_module(os.path.join(cwd, '__temp', file))
        # check for classes HMM and MEMM
        if not hasattr(models, 'HMM'):
          failwithmessage("Could not find class 'HMM' in " + file + ". Are you sure it exists?")
        if not hasattr(models, 'MEMM'):
          failwithmessage("Could not find class 'MEMM' in " + file + ". Are you sure it exists?")
        
        # TODO: check for functions in HMM and MEMM

      if (file == 'helpers.py'):
        helpers = load_student_module(os.path.join(cwd, '__temp', file))
        check_module_for_functions(helpers, ['handle_unknown_words', 'apply_smoothing'], file)
      
      if (file == 'viterbi.py'):
        viterbi = load_student_module(os.path.join(cwd, '__temp', file))
        check_module_for_functions(viterbi, ['viterbi'], file)

      if (file == 'validation.py'):
        validation = load_student_module(os.path.join(cwd, '__temp', file))
        check_module_for_functions(validation, ['evaluate_model'], file)

    remove_temp_dir()
    successwithmessage("Submission check complete-- it looks like all your files are in order! This is no guarantee.")

if __name__ == "__main__":
    check_submission()