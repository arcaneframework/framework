// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* AcceleratorCoreGlobalInternal.h                             (C) 2000-2025 */
/*                                                                           */
/* General declarations for accelerator support.                             */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_COMMON_ACCELERATOR_INTERNAL_ACCELERATORCOREGLOBALINTERNAL_H
#define ARCCORE_COMMON_ACCELERATOR_INTERNAL_ACCELERATORCOREGLOBALINTERNAL_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/common/accelerator/CommonAcceleratorGlobal.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Accelerator::Impl
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! Indicates if CUDA runtime is being used
extern "C++" ARCCORE_COMMON_EXPORT bool isUsingCUDARuntime();

//! Sets the usage of the CUDA runtime
extern "C++" ARCCORE_COMMON_EXPORT void setUsingCUDARuntime(bool v);

//! Retrieves the CUDA implementation of RunQueue (may be null)
extern "C++" ARCCORE_COMMON_EXPORT IRunnerRuntime*
getCUDARunQueueRuntime();

//! Sets the CUDA implementation of RunQueue.
extern "C++" ARCCORE_COMMON_EXPORT void setCUDARunQueueRuntime(IRunnerRuntime* v);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! Indicates if HIP runtime is being used
extern "C++" ARCCORE_COMMON_EXPORT bool isUsingHIPRuntime();

//! Sets the usage of the HIP runtime
extern "C++" ARCCORE_COMMON_EXPORT void setUsingHIPRuntime(bool v);

//! Retrieves the HIP implementation of RunQueue (may be null)
extern "C++" ARCCORE_COMMON_EXPORT IRunnerRuntime*
getHIPRunQueueRuntime();

//! Sets the HIP implementation of RunQueue.
extern "C++" ARCCORE_COMMON_EXPORT void setHIPRunQueueRuntime(IRunnerRuntime* v);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! Indicates if SYCL runtime is being used
extern "C++" ARCCORE_COMMON_EXPORT bool isUsingSYCLRuntime();

//! Sets the usage of the SYCL runtime
extern "C++" ARCCORE_COMMON_EXPORT void setUsingSYCLRuntime(bool v);

//! Retrieves the SYCL implementation of RunQueue (may be null)
extern "C++" ARCCORE_COMMON_EXPORT IRunnerRuntime*
getSYCLRunQueueRuntime();

//! Sets the SYCL implementation of RunQueue.
extern "C++" ARCCORE_COMMON_EXPORT void setSYCLRunQueueRuntime(IRunnerRuntime* v);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Retrieves the current accelerator implementation (may be null).
 *
 * The returned pointer is null if no accelerator runtime is set.
 * If isUsingCUDARuntime() is true, returns the runtime associated with CUDA.
 * If isUsingHIPRuntime() is true, returns the runtime associated with HIP.
 */
extern "C++" ARCCORE_COMMON_EXPORT IRunnerRuntime*
getAcceleratorRunnerRuntime();

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! Retrieves the Sequential RunQueue implementation
extern "C++" ARCCORE_COMMON_EXPORT IRunnerRuntime*
getSequentialRunQueueRuntime();

//! Retrieves the Thread RunQueue implementation
extern "C++" ARCCORE_COMMON_EXPORT IRunnerRuntime*
getThreadRunQueueRuntime();

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! Prints the UUID of an accelerator
extern "C++" ARCCORE_COMMON_EXPORT void
printUUID(std::ostream& o, char bytes[16]);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! Static class providing internal functions for Arcane.
class ARCCORE_COMMON_EXPORT RuntimeStaticInfo
{
 public:

  static ePointerAccessibility
  getPointerAccessibility(eExecutionPolicy policy, const void* ptr, PointerAttribute* ptr_attr);

  static void
  checkPointerIsAcccessible(eExecutionPolicy policy, const void* ptr,
                            const char* name, const TraceInfo& ti);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Initializes \a runner with the information in \a acc_info.
 *
 * This function calls runner.setAsCurrentDevice() after
 * initialization.
 */
extern "C++" ARCCORE_COMMON_EXPORT void
arccoreInitializeRunner(Runner& runner, ITraceMng* tm,
                        const AcceleratorRuntimeInitialisationInfo& acc_info);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::Accelerator::Impl

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
