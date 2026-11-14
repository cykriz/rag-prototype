---
title: Command Overview
---

Widget groups its functionality into a small set of commands. Each command prints help text when called with `--help`.

## init

`widget init` creates a configuration file in the current folder. It refuses to overwrite an existing file unless you pass `--force`.

## build

`widget build` reads the configuration, collects all input files and writes the output folder. The build is incremental: unchanged inputs are skipped, so a second build without changes finishes almost instantly.

```sh
widget build --out dist
```

## watch

`widget watch` runs a build and then waits for file changes. Every change triggers a new incremental build. Press Ctrl+C to stop watching.

## clean

`widget clean` removes the output folder and the build cache. Use it when the cache seems out of date or after upgrading widget to a new major version.

## doctor

`widget doctor` checks the environment for common problems, such as a missing configuration file, an unreadable input folder or a stale lock file, and prints one line per finding with a suggested fix.

## upgrade

`widget upgrade` replaces the installed binary with the latest release. It keeps the configuration file and the build cache, but a major upgrade asks you to run `widget clean` afterwards.

## publish

`widget publish` copies the output folder to the target configured under `publish.target`. It refuses to run when the last build failed or when the output folder is older than the newest input file.

## status

`widget status` prints the configured name, the number of input files and the time of the last successful build. It never changes any files.
