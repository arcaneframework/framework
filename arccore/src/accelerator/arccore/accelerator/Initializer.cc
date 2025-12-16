// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* Initializer.h                                               (C) 2000-2025 */
/*                                                                           */
/* Classe pour initialiser le runtime accélérateur.                          */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/accelerator/internal/Initializer.h"

#include "arccore/base/String.h"
#include "arccore/base/PlatformUtils.h"

#include "arccore/common/accelerator/internal/RuntimeLoader.h"
#include "arccore/common/accelerator/AcceleratorRuntimeInitialisationInfo.h"

#include <iostream>
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Accelerator
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Initializer::
Initializer(bool use_accelerator, Int32 max_allowed_thread)
{
  std::cout << "INIT_ACCELERATOR use?=" << use_accelerator << "\n";
  if (use_accelerator) {
    AcceleratorRuntimeInitialisationInfo init_info;
    String default_runtime_name;
#if defined(ARCCORE_HAS_CUDA)
    default_runtime_name = "cuda";
#elif defined(ARCCORE_HAS_HIP)
    default_runtime_name = "hip";
#elif defined(ARCCORE_HAS_SYCL)
    default_runtime_name = "sycl";
#endif

    String dll_full_path = Platform::getLoadedSharedLibraryFullPath("arccore_accelerator");
    String library_path;
    if (!dll_full_path.null())
      library_path = Platform::getFileDirName(dll_full_path);
    if (library_path.null())
      library_path = Platform::getCurrentDirectory();

    std::cout << "INIT_ACCELERATOR default_runtime=" <<  default_runtime_name << " lib_path=" << library_path << "\n";
    bool has_accelerator = false;
    init_info.setIsUsingAcceleratorRuntime(true);
    int r = Impl::RuntimeLoader::loadRuntime(init_info, default_runtime_name, library_path, has_accelerator);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::Accelerator

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
