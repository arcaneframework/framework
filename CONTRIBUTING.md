# Development Guidelines

## Commit messages

The commit message must follow this format :


`[` `part` `:` `subdir` `]` `Short message`

`Long message`



A commit must only affect one part of the framework (for example, only Arcane or only Arccore).

You can edit multiple subdirectories in a single commit :

`[part:subdir` `, subdir2` `, subdirN` `] Short message`

Example :

`[arcane:cartesianmesh] Add InPatch/Overlap flags on nodes and faces`

If one of your commits requires a second commit to be compiled, you must indicate this as follows :

`[part1:subdir1]` `[1/2]` `Short message of the first commit`

`[part2:subdir2]` `[2/2]` `Short message of the second commit`

For all your commits, the framework must be able to be compiled. You can create a temporary draft to save your work, but
you will need to amend it (or squash it with the next commit) before merging your PR :

`[Draft] Short message`

Your commits should be as short as possible.

## Branch name

The branch name must follow this form :

`dev/` `initials` `-` `short-message`

Example :

`dev/ah-alien-doxygen`


## Pull request (PR)

You must assign yourself every PR you have created.

You must select all labels that describe your development.

You can designate reviewers if necessary.

Your PR must be up to date in order to be merged. To update it, you must rebase it onto the main branch. You must not
merge the main branch into your branch.

To merge your pull request into the main branch, you need to merge it into that branch. Do not use squash or rebase.

If the title of your pull request does not sufficiently describe your changes, you should write a message explaining your work.

## Signed commits

End 2026, your commits must be signed.

https://docs.github.com/en/authentication/managing-commit-signature-verification/about-commit-signature-verification

