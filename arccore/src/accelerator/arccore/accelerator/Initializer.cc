// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* Initializer.h                                               (C) 2000-2026 */
/*                                                                           */
/* Classe pour initialiser le runtime accélérateur.                          */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/accelerator/internal/Initializer.h"

#include "arccore/base/String.h"
#include "arccore/base/PlatformUtils.h"
#include "arccore/base/CoreArray.h"

#include "arccore/common/List.h"
#include "arccore/common/accelerator/internal/AcceleratorCoreGlobalInternal.h"
#include "arccore/common/accelerator/internal/RuntimeLoader.h"
#include "arccore/common/accelerator/internal/RunnerInternal.h"
#include "arccore/common/accelerator/AcceleratorRuntimeInitialisationInfo.h"

#include <iostream>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{
namespace
{
  Impl::CoreArray<String>
  _stringListToCoreArray(const StringList& slist)
  {
    Impl::CoreArray<String> a;
    for (const String& s : slist)
      a.add(s);
    return a;
  }
} // namespace
} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Accelerator
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Initializer::
Initializer(bool use_accelerator, Int32 max_allowed_thread)
: m_trace_mng(arccoreCreateDefaultTraceMng())
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

    std::cout << "INIT_ACCELERATOR default_runtime=" << default_runtime_name << " lib_path=" << library_path << "\n";
    bool has_accelerator = false;
    init_info.setIsUsingAcceleratorRuntime(true);
    int r = Impl::RuntimeLoader::loadRuntime(init_info, default_runtime_name, library_path, has_accelerator);
    if (r == 0) {
      if (Impl::isUsingCUDARuntime())
        m_policy = eExecutionPolicy::CUDA;
      else if (Impl::isUsingHIPRuntime())
        m_policy = eExecutionPolicy::HIP;
      else if (Impl::isUsingSYCLRuntime())
        m_policy = eExecutionPolicy::SYCL;
    }
  }
  {
    m_concurrency_application.setTraceMng(m_trace_mng);
    m_application_build_info.setDefaultValues();
    m_application_build_info.setDefaultServices();
    if (max_allowed_thread > 1)
      m_application_build_info.setNbTaskThread(max_allowed_thread);
    {
      const auto& b = m_application_build_info;
      auto task_names = _stringListToCoreArray(b.taskImplementationServices());
      auto thread_names = _stringListToCoreArray(b.threadImplementationServices());
      Int32 nb_task_thread = b.nbTaskThread();
      ConcurrencyApplicationBuildInfo c(task_names.constView(), thread_names.constView(), nb_task_thread);
      m_concurrency_application.setCoreServices(c);
    }
    if (max_allowed_thread > 1)
      m_policy = eExecutionPolicy::Thread;
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Initializer::
~Initializer() noexcept(false)
{
  Accelerator::RunnerInternal::finalize(nullptr);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::Accelerator

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
