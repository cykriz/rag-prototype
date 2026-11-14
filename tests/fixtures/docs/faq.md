# Frequently Asked Questions

## Is widget a real tool?

No. Widget exists only inside this repository as synthetic test data for the ingest pipeline.

## Why does the build skip files?

The build compares a hash of each input with the hash from the previous run. Inputs with an unchanged hash are skipped, which keeps repeated builds fast.

## Where are logs written?

Logs go to standard error. Pass `--verbose` to any command to include timing information for every step.

## How do I reset everything?

Run `widget clean`, delete the configuration file and run `widget init` again. This restores the defaults.

## Can two builds run at the same time?

No. Widget takes a lock file in the output folder and a second build waits until the first one has finished.

## Does widget send telemetry?

No. Widget never opens a network connection after installation and stores nothing outside the project folder.
