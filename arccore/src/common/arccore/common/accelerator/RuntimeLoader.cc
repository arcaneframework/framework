// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* RuntimeLoader.cc                                            (C) 2000-2025 */
/*                                                                           */
/* Gestion du chargement du runtime accélérateur.                            */
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
 * \brief Détecte et charge la bibliothèque de gestion du runtime des accélérateurs.
 *
 * Cette méthode ne doit être appelée qu'une seule fois.
 *
 * Si non nul, \a default_runtime_name sera utilisé si init_info.acceleratorRuntime()
 * est nul.
 *
 * En retour, \a has_accelerator est vrai si on a chargé un runtime accélérateur
 * (Cuda, Hip ou Sycl)
 *
 * \retval 0 si tout est OK
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
    // Pour l'instant, seuls les runtimes 'cuda', 'hip' et 'sycl' sont autorisés
    if (runtime_name != "cuda" && runtime_name != "hip" && runtime_name != "sycl")
      ARCCORE_FATAL("Invalid accelerator runtime '{0}'. Only 'cuda', 'hip' or 'sycl' is allowed", runtime_name);

    // Pour pouvoir automatiquement enregistrer un runtime accélérateur de nom \a NAME,
    // il faut appeler la méthode 'arcaneRegisterAcceleratorRuntime${NAME}' qui se trouve
    // dans la bibliothèque dynamique 'arcane_${NAME}'.

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

    // Permet de surcharger le choix de l'allocateur des données
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
