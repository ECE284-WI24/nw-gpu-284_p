[![Open in Visual Studio Code](https://classroom.github.com/assets/open-in-vscode-718a45dd9cf7e7f842a935f5ebbe5719a5e09af4491e668f4dbf3b35d5cca122.svg)](https://classroom.github.com/online_ide?assignment_repo_id=14019513&assignment_repo_type=AssignmentRepo)
# GPU implementation of Needleman-Wunsch algorithm 

## TASKS
* Parallelize the kernel `alignSeqToSeq` on the GPU
* Multiple threads of a thread block map one pair of reads (reference and query) at a time
* Leverage wavefront parallelism (the current implementation already computes scores and stores traceback pointers of the matrix one antidiagonal after another, but not parallelized) 
* Coalescing memory accesses for reading reference and query sequences
* Exploit the reuse of sequences (both reference and query) using shared memory
* Any additional optimization you can think of that improves performance
* (Bonus) Implement the X-Drop banding technique on top of the NW algorithm

## Setting up

Like before, we will be using UC San Diego's Data Science/Machine Learning Platform ([DSMLP](https://blink.ucsd.edu/faculty/instruction/tech-guide/dsmlp/index.html)) for these assignments.

Please follow the steps below to set up the project:

* SSH into the DSMLP server (dsmlp-login.ucsd.edu) using the AD account. I recommend using PUTTY SSH client (putty.org) or Windows Subsystem for Linux (WSL) for Windows (https://docs.microsoft.com/en-us/windows/wsl/install-manual). MacOS and Linux users can SSH into the server using the following command (replace `yturakhia` with your username)

```
ssh yturakhia@dsmlp-login.ucsd.edu
```

* Next, clone the assignment repository in your HOME directory using the following example command (replace repository name `nw-gpu-yatisht` with the correct name based on step 1) and decompress the data files:
```
cd ~
git clone https://github.com/ECE284-WI24/nw-gpu-yatisht
cd nw-gpu-yatisht/data
xz --decompress reference.fa.xz
xz --decompress query.fa.xz
cd ~
```

* Download a copy of the TBB version 2019_U9 into your HOME directory:

```
wget https://github.com/oneapi-src/oneTBB/archive/2019_U9.tar.gz
tar -xvzf 2019_U9.tar.gz
```



## Code development and testing

Once your environment is set up on the DSMLP server, you can begin code development and testing using either VS code (that many of you must be familiar with) or if you prefer, using the shell terminal itself (with text editors, such as Vim or Emacs). If you prefer the latter, you can skip the step 1 below.

1. Launch a VS code server from the DSMLP login server using the following command:
   ```
   /opt/launch-sh/bin/launch-codeserver
   ```
   If successful, the log of the command will include a message such as:
   ```
   You may access your Code-Server (VS Code) at: http://dsmlp-login.ucsd.edu:14672 using password XXXXXX
   ```
   If the launch command is *unsuccessful*, make sure that there are no aleady running pods:
   ```
   # View running pods
   kubectl get pods
   # Delete all pods
   kubectl delete pod --all
   ```
   As conveyed in the message of the successful launch command, you can access the VS code server by going to the URL above (http://dsmlp-login.ucsd.edu:14672 in the above example) and entering the password displayed. Note that you may need to use UCSD's VPN service (https://blink.ucsd.edu/technology/network/connections/off-campus/VPN/) if you are performing this step from outside the campus network. Once you gain access to the VS code server from your browser, you can view the directories and files in your DSMLP filesystem and develop code. You can also open a terminal (https://code.visualstudio.com/docs/editor/integrated-terminal) from the VS code interface and run commands on the login server.

2. We will be using a Docker container, namely `yatisht/ece284-wi24:latest`, for submitting a job on the cluster containing the right virtual environment to build and test the code. This container already contains the correct Cmake version, CUDA and Boost libraries preinstalled within Ubuntu-18.04 OS. Note that these Docker containers use the same filesystem as the DSMLP login server, and hence the files written to or modified by the conainer is visiable to the login server and vice versa. To submit a job that executes `run-commands.sh` script located inside the `nw-gpu-yatisht` direcotry on a VM instance with 8 CPU cores, 16 GB RAM and 1 GPU device (this is the maxmimum allowed request on the DSMLP platform), the following command can be executed from the VS Code or DSMLP Shell Terminal (replace the username and directory names below appropriately):

```
ssh yturakhia@dsmlp-login.ucsd.edu /opt/launch-sh/bin/launch.sh -v 2080ti -c 8 -g 1 -m 8 -i yatisht/ece284-wi24:latest -f ./nw-gpu-yatisht/run-commands.sh
```
Note that the above command will require you to enter your AD account password again. This command should work and provide a sensible output for the assignment already provided. If you have reached this, you are in good shape to develop and test the code (make sure to modify `run-commands.sh` appropriately before testing). Happy code development! 
