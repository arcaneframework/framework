# Contributing to Arcane Framework

Thank you for contributing to Arcane Framework. This guide describes the
workflow and repository conventions used for contributions.

## Before you start

Read the [project README](README.md) and the relevant component documentation
before making a change. Arcane Framework is a monorepo; keep a contribution
focused on the component it changes (`arccore`, `arcane`, `alien`, `arccon`, or
`axlstar`).

Clone the repository with its submodules:

```bash
git clone --recurse-submodules https://github.com/arcaneframework/framework.git
```

If the repository was cloned without submodules, initialise them before
configuring CMake:

```bash
git submodule update --init --recursive
```

For prerequisites and platform-specific installation instructions, see the
[build and installation guide](https://arcaneframework.github.io/arcane/userdoc/html/d7/d94/arcanedoc_build_install.html).

## Build and test your change

Use an out-of-source build directory. A regular Arcane build can be configured
with the supplied preset:

```bash
cmake --preset Arcane
cmake --build out/build/Arcane
```

Alternatively, configure a dedicated build directory explicitly. Select only
the components needed for the change when that makes the feedback loop faster:

```bash
cmake -S . -B build -DARCANEFRAMEWORK_BUILD_COMPONENTS=Arcane
cmake --build build
```

Run the relevant tests before opening or updating a pull request:

The commands below use the explicit `build` directory shown above. If you use
the `Arcane` preset, replace `build` with `out/build/Arcane`.

```bash
ctest --test-dir build --output-on-failure
```

Use a targeted invocation when appropriate:

```bash
ctest --test-dir build --output-on-failure -R <test-name>
```

Some tests require MPI. When running as root in a container or CI environment,
Open MPI may require the following variables:

```bash
export OMPI_ALLOW_RUN_AS_ROOT=1
export OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1
```

## Code and file conventions

Follow [.editorconfig](.editorconfig). In particular, C++, C#, CMake, and
`CMakeLists.txt` files use two spaces for indentation and no trailing
whitespace.

New or edited `.cc` and `.h` files are checked in CI for the project header,
copyright notice, and UTF-8 with BOM encoding. Start from a nearby maintained
source file in the same component so that these conventions are preserved.

Keep public API changes deliberate. In Arcane, the stable public API is limited
to `core/`, `materials/`, `utils/`, `launcher/`, `accelerator/`,
`cartesianmesh/`, and `hdf5/`; other `arcane/` directories are internal.

## Branch names

Create a branch using this pattern:

```text
dev/<initials>-<short-description>
```

For example:

```text
dev/ah-alien-doxygen
```

## Commit messages

Keep each commit focused and as small as practical. A commit must affect only
one framework part (for example, Arcane or Arccore) and leave the framework
buildable.

Use the following subject format:

```text
[<part>:<subdirectory>] Short imperative summary
```

For example:

```text
[arcane:cartesianmesh] Add InPatch/Overlap flags on nodes and faces
```

When a commit affects several subdirectories of the same part, list them in
the scope:

```text
[<part>:<subdirectory-1>,<subdirectory-2>] Short imperative summary
```

When work is delivered as a small, ordered commit series, number the commits
clearly:

```text
[<part>:<subdirectory>] [1/2] First change
[<part>:<subdirectory>] [2/2] Follow-up change
```

Temporary work-in-progress commits are acceptable locally. Mark them as
`[Draft]`, then amend or squash them before the pull request is merged.

Commit signing will become mandatory at the end of 2026. See GitHub's
[documentation on commit signature verification](https://docs.github.com/en/authentication/managing-commit-signature-verification/about-commit-signature-verification)
to prepare your environment.

## Pull requests

Before requesting review:

- Assign the pull request to yourself.
- Choose every label that describes the change.
- Ensure the title clearly describes the result; add a short description when
  the title alone is insufficient.
- Build the affected configuration and run relevant tests locally.
- Keep the pull request focused. Explain compatibility implications, generated
  files, or intentionally untested paths in its description.

Keep the branch up to date with `main` by rebasing; do not merge `main` into
the contribution branch:

```bash
git fetch origin
git rebase origin/main
```

Resolve conflicts and rerun the appropriate build and tests after the rebase.
Once approved and checks pass, merge the pull request into `main` with a merge
commit; do not use squash or rebase merging.
