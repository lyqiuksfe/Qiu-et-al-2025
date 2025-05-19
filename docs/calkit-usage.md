# Calkit usage in Qiu et al. 2025

## How this project was set up

1. Created the project on calkit.io.
2. Cloned locally with `calkit clone petebachant/qiu-2025-energy-modeling`.
3. Merged in README.md content from email attachment and ran
   `calkit save -aM` to commit and push with an auto-generated message.
4. Started this document and ran `calkit save -am "Start docs"`.
5. Added working copy of paper as `paper.docx` and ran
   `calkit save -am "Add working copy of paper"`.
6. Unzipped the ZIP file from OneDrive and copied all contents in here
   with `cp ~/Downloads/Qiu_etal2025/* .`.
7. Ran `calkit add .` to stage files for commit without committing, to see
   if they'd belong in Git or DVC.
   Some notebooks ended up in DVC because they were large, which is probably
   not ideal.
   One was 2.9 MB and the other was 4.9 MB.
   It might be nice to have a `--git-size-thresh` option or some other way to
   force certain paths to be added to Git.
   As a workaround, ran `dvc remove input_data.ipynb.dvc` and
   `dvc remove analysis/plot_final.ipynb.dvc`, then manually added both of
   those files to git with the `git add` command.
   Note that this `dvc remove` command actually caused the notebooks to be
   deleted on `dvc pull`. `git checkout .` brought them back, however,
   since they were already committed.
8. Ran `git add .` once everything looked right, and `calkit status` to see
   which would end up in Git and which in DVC.
9. Ran `calkit push` to get everything backed up to GitHub/calkit.io.
