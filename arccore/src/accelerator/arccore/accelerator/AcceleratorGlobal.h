// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* AcceleratorGlobal.h                                         (C) 2000-2025 */
/*                                                                           */
/* Déclarations générales pour le support des accélérateurs.                 */
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
 * \brief Template pour savoir si un type utilisé comme boucle dans les
 * kernels nécessite toujours sycl::nb_item comme argument.
 *
 * Si c'est le cas, il faut spécialiser cette template en la faisant
 * dériver de std::true_type. C'est le cas par exemple pour WorkGroupLoopRange.
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

//! Type d'opération atomique supportée
enum class eAtomicOperation
{
  //! Ajout
  Add,
  //! Minimum
  Min,
  //! Maximum
  Max
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Initialise \a runner en fonction de
 * la valeur de \a acc_info.
 */
extern "C++" ARCCORE_ACCELERATOR_EXPORT void
initializeRunner(Runner& runner, ITraceMng* tm,
                 const AcceleratorRuntimeInitialisationInfo& acc_info);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::Accelerator

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Accelerator::impl
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

extern "C++" ARCCORE_ACCELERATOR_EXPORT String
getBadPolicyMessage(eExecutionPolicy policy);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::Accelerator::impl

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! Macro pour indiquer qu'un noyau n'a pas été compilé avec HIP
#define ARCCORE_FATAL_NO_HIP_COMPILATION() \
  ARCCORE_FATAL(Arcane::Accelerator::impl::getBadPolicyMessage(Arcane::Accelerator::eExecutionPolicy::HIP));

//! Macro pour indiquer qu'un noyau n'a pas été compilé avec CUDA
#define ARCCORE_FATAL_NO_CUDA_COMPILATION() \
  ARCCORE_FATAL(Arcane::Accelerator::impl::getBadPolicyMessage(Arcane::Accelerator::eExecutionPolicy::CUDA));

//! Macro pour indiquer qu'un noyau n'a pas été compilé avec SYCL
#define ARCCORE_FATAL_NO_SYCL_COMPILATION() \
  ARCCORE_FATAL(Arcane::Accelerator::impl::getBadPolicyMessage(Arcane::Accelerator::eExecutionPolicy::SYCL));

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
