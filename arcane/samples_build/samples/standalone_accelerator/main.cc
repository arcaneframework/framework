// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------

#include <arcane/launcher/ArcaneLauncher.h>

#include "arcane/utils/NumArray.h"
#include "arcane/accelerator/core/IAcceleratorMng.h"

#include "arcane/accelerator/NumArrayViews.h"
#include "arcane/accelerator/RunQueue.h"
#include "arcane/accelerator/RunCommandLoop.h"

using namespace Arcane;

void
_testStandaloneLauncher()
{
  StandaloneAcceleratorMng launcher(ArcaneLauncher::createStandaloneAcceleratorMng());
  IAcceleratorMng* acc_mng = launcher.acceleratorMng();

  constexpr int nb_value = 10000;

  // Teste la somme de deux tableaux 'a' et 'b' dans un tableau 'c'.

  // Définit 2 tableaux 'a' et 'b' et effectue leur initialisation.
  NumArray<Int64,1> a(nb_value);
  NumArray<Int64,1> b(nb_value);
  for( int i=0; i<nb_value; ++i ){
    a.s(i) = i+2;
    b.s(i) = i+3;
  }

  // Defínit le tableau 'c' qui contiendra la somme de 'a' et 'b'
  NumArray<Int64,1> c(nb_value);

  {
    // Noyau de calcul déporté sur accélérateur.
    auto command = makeCommand(acc_mng->defaultQueue());
    // Indique que 'a' et 'b' seront en entrée et 'c' en sortie

    auto in_a = viewIn(command,a);
    auto in_b = viewIn(command,b);
    auto out_c = viewOut(command,c);
    command << RUNCOMMAND_LOOP1(iter,nb_value) {
      out_c(iter) = in_a(iter) + in_b(iter);
    };
  }

  // Vérifie le résultat
  Int64 total = 0.0;
  for( int i=0; i<nb_value; ++i )
    total += c(i);
  std::cout << "TOTAL=" << total << "\n";
  Int64 expected_total = 100040000;
  if (total!=expected_total)
    ARCANE_FATAL("Bad value for sum={0} (expected={1})",total,expected_total);
}

int
main(int argc,char* argv[])
{
  try{
    ArcaneLauncher::init(CommandLineArguments(&argc,&argv));
    _testStandaloneLauncher();
  }
  catch(const Exception& ex){
    std::cerr << "EXCEPTION: " << ex << "\n";
    return 1;
  }
  return 0;
}
