# Changelog

All notable changes to this project will be documented in this file.

## Unreleased

**Fixed bugs:**
- [hypre]: Correctly clear hypre solver errors
- [core]: Fix name clashing between `Move` and `Ref` APIs ([issue#12](https://github.com/arcaneframework/alien/issues/12))

**Changes:**
- switch to mono-repo for all Alien related projects
- `move` api is now in `Alien::Move` namespace