// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*
 * This file is based on the work on AMGCL library (version march 2026)
 * which can be found at https://github.com/ddemidov/amgcl.
 *
 * Copyright (c) 2012-2022 Denis Demidov <dennis.demidov@gmail.com>
 * SPDX-License-Identifier: MIT
 */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include <iostream>
#include <string>

#include <boost/program_options.hpp>
#include "arccore/alina/AlinaUtils.h"
#include "arccore/alina/Adapters.h"
#include "arccore/alina/IO.h"

int main(int argc, char* argv[])
{
  namespace po = boost::program_options;
  namespace io = Arcane::Alina::IO;

  using Arcane::Alina::precondition;

  po::options_description desc("Options");

  desc.add_options()("help,h", "Show this help.")("dense,d", po::bool_switch()->default_value(false),
                                                  "Matrix is dense.")("input,i", po::value<std::string>()->required(),
                                                                      "Input binary file.")("output,o", po::value<std::string>()->required(),
                                                                                            "Ouput matrix in the MatrixMarket format.");

  po::variables_map vm;
  po::store(po::parse_command_line(argc, argv, desc), vm);

  if (vm.count("help")) {
    std::cout << desc << std::endl;
    return 0;
  }

  po::notify(vm);

  if (vm["dense"].as<bool>()) {
    size_t n, m;
    std::vector<double> v;

    io::read_dense(vm["input"].as<std::string>(), n, m, v);
    io::mm_write(vm["output"].as<std::string>(), v.data(), n, m);

    std::cout
    << "Wrote " << n << " by " << m << " dense matrix"
    << std::endl;
  }
  else {
    size_t n;
    std::vector<ptrdiff_t> ptr, col;
    std::vector<double> val;

    io::read_crs(vm["input"].as<std::string>(), n, ptr, col, val);
    io::mm_write(vm["output"].as<std::string>(), std::tie(n, ptr, col, val));

    std::cout
    << "Wrote " << n << " by " << n << " sparse matrix, "
    << ptr.back() << " nonzeros" << std::endl;
  }
}
