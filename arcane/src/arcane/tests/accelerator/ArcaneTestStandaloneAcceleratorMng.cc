﻿// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------

#include "arcane/launcher/ArcaneLauncher.h"
#include "arcane/utils/NumArray.h"
#include "arcane/accelerator/core/IAcceleratorMng.h"

#include "arcane/accelerator/NumArrayViews.h"
#include "arcane/accelerator/RunQueue.h"
#include "arcane/accelerator/RunCommandLoop.h"

using namespace Arcane;

namespace
{
void
_testSum(IAcceleratorMng* acc_mng)
{
  // Test la somme de deux tableaux 'a' et 'b' dans un tableau 'c'.

  int nb_value = 10000;
  NumArray<Int64,1> a(nb_value);
  NumArray<Int64,1> b(nb_value);
  NumArray<Int64,1> c(nb_value);
  for( int i=0; i<nb_value; ++i ){
    a.s(i) = i+2;
    b.s(i) = i+3;
  }

  {
    auto command = makeCommand(acc_mng->defaultQueue());
    auto in_a = viewIn(command,a);
    auto in_b = viewIn(command,b);
    auto out_c = viewOut(command,c);
    command << RUNCOMMAND_LOOP1(iter,nb_value)
    {
      auto [i] = iter();
      out_c(i) = in_a(i) + in_b(i);
    };
  }

  Int64 total = 0.0;
  for( int i=0; i<nb_value; ++i )
    total += c(i);
  std::cout << "TOTAL=" << total << "\n";
  Int64 expected_total = 100040000;
  if (total!=expected_total)
    ARCANE_FATAL("Bad value for sum={0} (expected={1})",total,expected_total);
}

int
_testStandaloneLauncher(const CommandLineArguments& cmd_line_args,
                        const String& method_name)
{
  ARCANE_UNUSED(method_name);
  ArcaneLauncher::init(cmd_line_args);
  StandaloneAcceleratorMng launcher(ArcaneLauncher::createStandaloneAcceleratorMng());
  IAcceleratorMng* acc_mng = launcher.acceleratorMng();
  if (method_name=="TestSum")
    _testSum(acc_mng);
  else
    ARCANE_FATAL("Unknown method to test");
  return 0;
}
}

extern "C++" ARCANE_EXPORT int
arcaneTestStandaloneLauncher(const CommandLineArguments& cmd_line_args,
                             const String& method_name)
{
  int r =0;
  try{
    r = _testStandaloneLauncher(cmd_line_args,method_name);
  }
  catch(const Exception& ex){
    std::cerr << "EXCEPTION: " << ex << "\n";
    throw;
  }
  return r;
}
