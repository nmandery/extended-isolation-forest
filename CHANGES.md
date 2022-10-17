# Changelog

All notable changes to this project will be documented in this file.

The format is loosely based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres
to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).


## extended-isolation-forest Unreleased

### Changed

* use boxed slice for the trees instead of vec as no growing is necessary
* Reduce image size of example plot
* Derive `Clone`, `Eq` and `PartialEq` on `ForestOptions`.

## extended-isolation-forest 0.2.1 - 2022-08-02

### Changed
* Preventing panic during sampling random values when upper and lower bound are equal.
### Added
* Acceleration example

## extended-isolation-forest 0.2.0 - 2021-05-12

First version released on crates.io
