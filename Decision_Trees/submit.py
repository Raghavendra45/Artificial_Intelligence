#!/usr/bin/env python
# coding=utf-8
from nelson.gtomscs import submit

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
    past or current student.\"
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


def main():
    filenames = ["decision_trees_submission.py"]
    require_pledges()

    submit('cs6601', 'assignment_4', filenames)


if __name__ == '__main__':
    main()
