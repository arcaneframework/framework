#ifndef TESTS_REFSEMANTICMVHANDLERS_OPTIONCONFIGURATION_H
#define TESTS_REFSEMANTICMVHANDLERS_OPTIONCONFIGURATION_H

#include <boost/program_options/value_semantic.hpp>
#include <boost/program_options/options_description.hpp>
#include <boost/program_options/variables_map.hpp>
#include <boost/program_options/parsers.hpp>

#include <Tests/Environment.h>

namespace Environment {

extern boost::program_options::variables_map
options(int argc, char** argv, boost::program_options::options_description& options)
{
  boost::program_options::options_description desc;

  desc.add_options()
      ("solver-package",
        boost::program_options::value<std::string>()->default_value("petsc"),
        "solver package name")
      ("solver",
        boost::program_options::value<std::string>()->default_value("bicgs"),
        "solver algo name")
      ("precond",
        boost::program_options::value<std::string>()->default_value("none"),
        "preconditioner id diag ilu ddml poly")
      ("max-iter",
        boost::program_options::value<int>()->default_value(1000), "max iterations")
      ("tol",
        boost::program_options::value<double>()->default_value(1.e-10),
        "solver tolerance")
      ("niter",
        boost::program_options::value<int>()->default_value(1),
        "nb of tests for perf measure")
      ("kernel",
        boost::program_options::value<std::string>()->default_value("CPUCBLAS"),
        "mcgsolver kernel name")
      ("output-level",
        boost::program_options::value<int>()->default_value(0),
        "output level")
      ("redist-strategy",
        boost::program_options::value<std::string>()->default_value("pair"),
        "redistribution strategy: valid values are : pair or unique")
      ("redist-method",
        boost::program_options::value<std::string>()->default_value("dok"),
        "redistribution method: valid values are : dok or csr");

  // Ajout des options de configuration des solveurs linéaires
  options.add(desc);

  boost::program_options::variables_map vm;
  auto parsed = boost::program_options::parse_command_line(argc, argv, options);
  boost::program_options::store(parsed, vm);
  boost::program_options::notify(vm);

  return vm;
}
}

#endif
