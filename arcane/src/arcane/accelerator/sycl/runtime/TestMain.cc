// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* TestMain.cc                                                 (C) 2000-2024 */
/*                                                                           */
/* Fichier main pour lancer les tests de base SYCL.                          */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/ArcaneGlobal.h"
#include "arcane/utils/Exception.h"

#include <iostream>

extern "C" ARCANE_EXPORT void
arcaneRegisterAcceleratorRuntimesycl();

int
main(int argc,char* argv[])
{
  std::cout << "TESTMAIN\n";
  int r = 0;
  try{
    arcaneRegisterAcceleratorRuntimesycl();
    ARCANE_UNUSED(argc);
    ARCANE_UNUSED(argv);
    r = 0;
  }
  catch(const Arcane::Exception& e){
    std::cerr << "Exception e=" << e << "\n";
  }
  catch(const std::exception& e){
    std::cerr << "Exception e=" << e.what() << "\n";
  }
  return r;
}
