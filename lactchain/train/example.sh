### behave has a login shell of type bin/bash
#!/bin/bash -l
#PBS -l select=10
#PBS -A FoundEpidem
#PBS -l walltime=3:00:00
#PBS -l filesystems=home:eagle
#PBS -q prod

### Name of your session
#PBS -N Critic-FineTuning

### Controlling output of application 
#PBS -k doe
#PBS -o ./output_logs_critic_finetune
#PBS -e ./error_logs_critic_finetune

### email notification
#PBS -m be
#PBS -M brianhsu636@gmail.com