# Setup
Clone this repository recursively:
`git clone --recursive https://github.gatech.edu/omscs6601/assignment_3.git`

(If your version of git does not support recurse clone, then clone without the option and run `git submodule init` and `git submodule update`).

If you run across certificate authentication issues during the clone, set the git SSL Verify option to false: `git config --global http.sslVerify false`.

## Python Dependencies

The submission scripts depend on the presence of 2 python packages - `requests` and `future`. If you are missing either of these packages, install them from the online Python registries. The easiest way to do this is through pip:

`pip install requests future`

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

# Submit your code
A friendly reminder: please ensure that your submission is in `probability_solution.py`. 
Instructions to submit the code will be added here very soon. 

# Vagrant

You have the option of using vagrant to make sure that your local code runs in the same environment as the servers on Bonnie (make sure you have [Vagrant](https://www.vagrantup.com/) and [Virtualbox](https://www.virtualbox.org/wiki/Downloads) installed).  To use this option run the following commands in the root directory of your assignment:

```
vagrant up --provider virtualbox
vagrant ssh
```

Your code lives in the `/vagrant` folder within this virtual machine. Changes made to files in your assignment folder will automatically be reflected within the machine.
