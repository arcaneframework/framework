// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* AcceleratorGlobal.h                                         (C) 2000-2026 */
/*                                                                           */
/* General declarations for accelerator support.                             */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_ACCELERATOR_ACCELERATORGLOBAL_H
#define ARCCORE_ACCELERATOR_ACCELERATORGLOBAL_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/common/accelerator/CommonAcceleratorGlobal.h"

#include <iosfwd>
#include <type_traits>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#ifdef ARCCORE_COMPONENT_arccore_accelerator
#define ARCCORE_ACCELERATOR_EXPORT ARCCORE_EXPORT
#else
#define ARCCORE_ACCELERATOR_EXPORT ARCCORE_IMPORT
#endif

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Accelerator
{
namespace Impl
{
  class CudaHipKernelRemainingArgsHelper;
  class SyclKernelRemainingArgsHelper;
  /*!
 * \brief Template to determine if a type used as a loop in kernels always
 * requires sycl::nb_item as an argument.
 *
 * If so, this template must be specialized by deriving it from
 * std::true_type. This is the case, for example, for WorkGroupLoopRange.
 */
  template <typename T>
  class IsAlwaysUseSyclNdItem
  : public std::false_type
  {
  };
} // namespace Impl

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <typename T, Int32 Extent = DynExtent> class LocalMemory;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! Supported atomic operation type
enum class eAtomicOperation
{
  //! Add
  Add,
  //! Minimum
  Min,
  //! Maximum
  Max
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::Accelerator

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Accelerator::Impl
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

extern "C++" ARCCORE_ACCELERATOR_EXPORT String
getBadPolicyMessage(eExecutionPolicy policy);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::Accelerator::Impl

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! Macro to indicate that a kernel was not compiled with HIP
#define ARCCORE_FATAL_NO_HIP_COMPILATION() \
  ARCCORE_FATAL(Arcane::Accelerator::Impl::getBadPolicyMessage(Arcane::Accelerator::eExecutionPolicy::HIP));

//! Macro to indicate that a kernel was not compiled with CUDA
#define ARCCORE_FATAL_NO_CUDA_COMPILATION() \
  ARCCORE_FATAL(Arcane::Accelerator::Impl::getBadPolicyMessage(Arcane::Accelerator::eExecutionPolicy::CUDA));

//! Macro to indicate that a kernel was not compiled with SYCL
#define ARCCORE_FATAL_NO_SYCL_COMPILATION() \
  ARCCORE_FATAL(Arcane::Accelerator::Impl::getBadPolicyMessage(Arcane::Accelerator::eExecutionPolicy::SYCL));

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
