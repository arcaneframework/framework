// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* RuntimeLoader.cc                                            (C) 2000-2025 */
/*                                                                           */
/* Management of the accelerator runtime loading.                            */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/common/accelerator/internal/RuntimeLoader.h"

#include "arccore/base/PlatformUtils.h"
#include "arccore/base/FatalErrorException.h"
#include "arccore/base/internal/IDynamicLibraryLoader.h"

#include "arccore/common/ExceptionUtils.h"
#include "arccore/common/MemoryUtils.h"
#include "arccore/common/internal/MemoryUtilsInternal.h"
#include "arccore/common/accelerator/AcceleratorRuntimeInitialisationInfo.h"
#include "arccore/common/accelerator/internal/RegisterRuntimeInfo.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Accelerator::Impl
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Detects and loads the accelerator runtime management library.
 *
 * This method must only be called once.
 *
 * If not null, \a default_runtime_name will be used if
 * init_info.acceleratorRuntime() is null.
 *
 * In return, \a has_accelerator is true if an accelerator runtime
 * (Cuda, Hip or Sycl) has been loaded.
 *
 * \retval 0 if everything is OK
 */
int RuntimeLoader::
loadRuntime(AcceleratorRuntimeInitialisationInfo& init_info,
            const String& default_runtime_name,
            const String& library_path,
            bool& has_accelerator)
{
  has_accelerator = false;
  //AcceleratorRuntimeInitialisationInfo& init_info = si->m_accelerator_init_info;
  if (!init_info.isUsingAcceleratorRuntime())
    return 0;
  String runtime_name = init_info.acceleratorRuntime();
  if (runtime_name == "sequential")
    return 0;
  if (runtime_name.empty())
    runtime_name = default_runtime_name;
  if (runtime_name.empty())
    return 0;
  init_info.setAcceleratorRuntime(runtime_name);
  try {
    // For now, only 'cuda', 'hip' and 'sycl' runtimes are allowed
    if (runtime_name != "cuda" && runtime_name != "hip" && runtime_name != "sycl")
      ARCCORE_FATAL("Invalid accelerator runtime '{0}'. Only 'cuda', 'hip' or 'sycl' is allowed", runtime_name);

    // To automatically register an accelerator runtime named \a NAME,
    // you must call the method 'arcaneRegisterAcceleratorRuntime${NAME}' which is found
    // in the dynamic library 'arcane_${NAME}'.

    typedef void (*ArcaneAutoDetectAcceleratorFunctor)(Accelerator::RegisterRuntimeInfo&);

    IDynamicLibraryLoader* dll_loader = IDynamicLibraryLoader::getDefault();

    String os_dir(library_path);
    String dll_name = "arccore_accelerator_" + runtime_name + "_runtime";
    String symbol_name = "arcaneRegisterAcceleratorRuntime" + runtime_name;
    IDynamicLibrary* dl = dll_loader->open(os_dir, dll_name);
    if (!dl)
      ARCCORE_FATAL("Can not found dynamic library '{0}' for using accelerator runtime", dll_name);

    bool is_found = false;
    void* functor_addr = dl->getSymbolAddress(symbol_name, &is_found);
    if (!is_found || !functor_addr)
      ARCCORE_FATAL("Can not find symbol '{0}' in library '{1}'", symbol_name, dll_name);

    auto my_functor = reinterpret_cast<ArcaneAutoDetectAcceleratorFunctor>(functor_addr);
    Accelerator::RegisterRuntimeInfo runtime_info;

    String verbose_str = Platform::getEnvironmentVariable("ARCANE_DEBUG_ACCELERATOR");
    if (!verbose_str.null())
      runtime_info.setVerbose(true);

    (*my_functor)(runtime_info);
    has_accelerator = true;

    // Allows overriding the data allocator choice
    String data_allocator_str = Platform::getEnvironmentVariable("ARCANE_DEFAULT_DATA_MEMORY_RESOURCE");
    if (!data_allocator_str.null()) {
      eMemoryResource v = MemoryUtils::getMemoryResourceFromName(data_allocator_str);
      if (v != eMemoryResource::Unknown)
        MemoryUtils::setDefaultDataMemoryResource(v);
    }
  }
  catch (const Exception& ex) {
    return ExceptionUtils::print(ex, nullptr);
  }
  catch (const std::exception& ex) {
    return ExceptionUtils::print(ex, nullptr);
  }
  catch (...) {
    return ExceptionUtils::print(nullptr);
  }
  return 0;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::Accelerator::Impl

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
