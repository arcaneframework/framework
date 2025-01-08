// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* TestMain.cc                                                 (C) 2000-2025 */
/*                                                                           */
/* Fichier main pour lancer les tests CUDA.                                  */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/ArcaneGlobal.h"
#include "arcane/utils/Exception.h"
#include "arcane/accelerator/core/internal/RegisterRuntimeInfo.h"

#include <iostream>

extern "C" ARCANE_IMPORT
int func0();

extern "C" ARCANE_EXPORT void
arcaneRegisterAcceleratorRuntimecuda(Arcane::Accelerator::RegisterRuntimeInfo& init_info);

int
main(int argc,char* argv[])
{
  int r = 0;
  try{
    Arcane::Accelerator::RegisterRuntimeInfo runtime_info;
    runtime_info.setVerbose(true);
    arcaneRegisterAcceleratorRuntimecuda(runtime_info);
    ARCANE_UNUSED(argc);
    ARCANE_UNUSED(argv);
    r = func0();
  }
  catch(const Arcane::Exception& e){
    std::cerr << "Exception e=" << e << "\n";
  }
  catch(const std::exception& e){
    std::cerr << "Exception e=" << e.what() << "\n";
  }
  return r;
}
