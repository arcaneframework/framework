# Changelog

All notable changes to this project will be documented in this file.

## Unreleased

**News:**

- \[move\]: add a `clone()` method to `MatrixData` and `VectorData`
- \[core\]: dramatic improvement of `DoK` performance

## 1.1.3

**News:**

- MatrixMarket file importer in \[move\] API
- \[ginkgo\]: new wrapper for the Ginkgo library
- \[SYCL\]: experimental kernel

**Fixed bugs:**

- \[core\]: fixed out of range access in DoKMatrix communication.

## 1.1.2

**Changes:**

- plugins are now part of Alien distribution. That means that only `find_package(Alien)` is necessary.

## 1.1.1

**Fixed bugs:**

- \[core, hypre, petsc\]: fix CMake export files
- \[petsc\]: internal call to PetscInitialize (fixes [issue#14](https://github.com/arcaneframework/alien/issues/14))

**Changes:**

- install tutorials and examples sources

## 1.1.0

**Fixed bugs:**

- \[hypre\]: Correctly clear hypre solver errors
- \[core\]: Fix name clashing between `Move` and `Ref`
  APIs ([issue#12](https://github.com/arcaneframework/alien/issues/12))

**Changes:**

- switch to mono-repo for all Alien related projects
- `move` api is now in `Alien::Move` namespace
- Setting CMake parameter `ALIEN_DEFAULT_OPTIONS` to `OFF` disable all optional external dependencies
