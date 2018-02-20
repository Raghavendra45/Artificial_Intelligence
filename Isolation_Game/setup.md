# Setup
Clone this repository:
`git clone https://github.gatech.edu/omscs6601/assignment_1.git`

If you run across certificate authentication issues during the clone, set the git SSL Verify option to false: `git config --global http.sslVerify false`.

## Python Dependencies

The submission scripts depend on the presence of 3 python packages - `requests`, `future`, and `nelson`. If you are missing any of these packages, install them from the online Python registries. The easiest way to do this is through pip:

`pip install -r requirements.txt`

# Keeping your code upto date
After the clone, we recommend creating a branch and developing your agents on that branch:

`git checkout -b develop`

(assuming develop is the name of your branch)

Should the TAs need to push out an update to the assignment, commit (or stash if you are more comfortable with git) the changes that are unsaved in your repository:

`git commit -am "<some funny message>"`

Then update the master branch from remote:

`git pull origin master`

This updates your local copy of the master branch. Now try to merge the master branch into your development branch:

`git merge master`

(assuming that you are on your development branch)

There are likely to be merge conflicts during this step. If so, first check what files are in conflict:

`git status`

The files in conflict are the ones that are "Not staged for commit". Open these files using your favourite editor and look for lines containing `<<<<` and `>>>>`. Resolve conflicts as seems best (ask a TA if you are confused!) and then save the file. Once you have resolved all conflicts, stage the files that were in conflict:

`git add -A .`

Finally, commit the new updates to your branch and continue developing:

`git commit -am "<funny message vilifying TAs for the update>"`

