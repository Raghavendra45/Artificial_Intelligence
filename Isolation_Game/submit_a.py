import time
import os
import sys
import argparse
import json
import datetime
from nelson.gtomscs import submit
# from bonnie.submission import Submission

LATE_POLICY = \
"""Late Policy:

  \"I have read the late policy for CS6601. I understand that only my last
  commit before the late submission deadline will be accepted.\"
"""

HONOR_PLEDGE = \
"""Honor Pledge:
  \"I have read the Collaboration and Academic Honesty policy for CS6601.
  I certify that I have used outside references only in accordance with
  this policy, that I have cited any such references via code comments,
  and that I have not copied any portion of my submission from another
  past or current student.\"\n"
"""

def require_pledges():
  print(LATE_POLICY)
  ans = raw_input("Please type 'yes' to agree and continue>")

  if ans != "yes":
    raise RuntimeError("Late policy not accepted.")

  print
  print(HONOR_PLEDGE)
  ans = raw_input("Please type 'yes' to agree and continue>")
  if ans != "yes":
    raise RuntimeError("Honor pledge not accepted")
  print

def display_assignment_1_output(submission):
  timestamp = "{:%Y-%m-%d-%H-%M-%S}".format(datetime.datetime.now())

  while not submission.poll():
    time.sleep(30.0)

  if submission.feedback():

    if submission.console():
        sys.stdout.write(submission.console())

    filename = "%s-result-%s.json" % (submission.quiz_name, timestamp)

    with open(filename, "w") as fd:
      json.dump(submission.feedback(), fd, indent=4, separators=(',', ': '))

    print("\n(Details available in %s.)" % filename)

  elif submission.error_report():
    error_report = submission.error_report()
    print(json.dumps(error_report, indent=4))
  else:
    print("Unknown error.")

def display_game(submission):
  while not submission.poll():
    time.sleep(3.0)

  if submission.feedback():
    sys.stdout.write(submission.feedback())
  elif submission.error_report():
      error_report = submission.error_report()
      print(json.dumps(error_report, indent=4))
  else:
    print("Unknown error.")

def main():
  parser = argparse.ArgumentParser(description='Submits code to the Udacity site.')
  # parser.add_argument('part', choices = ['assignment_1', 'play_isolation'])
  parser.add_argument('--enable-face-off', action='store_true', help='Include this flag to sign up for the playoffs. AI.txt must be present')

  args = parser.parse_args()

  filenames = ["player_submission.py"]
  if not args.enable_face_off:
    require_pledges()
  else:
    filenames.append('AI.txt')

  submit('cs6601', 'assignment_1a', filenames)

if __name__ == '__main__':
  main()

