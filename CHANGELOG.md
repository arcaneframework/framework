# Changelog

All notable changes to this project will be documented in this file.

## 1.1.0

**Fixed bugs:**

- [hypre]: Correctly clear hypre solver errors
- [core]: Fix name clashing between `Move` and `Ref`
  APIs ([issue#12](https://github.com/arcaneframework/alien/issues/12))

**Changes:**

- switch to mono-repo for all Alien related projects
- `move` api is now in `Alien::Move` namespace
- Setting CMake parameter `ALIEN_DEFAULT_OPTIONS` to `OFF` disable all optional external dependencies
