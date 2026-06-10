// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* AcceleratorInitializer.h                                    (C) 2000-2026 */
/*                                                                           */
/* Initializer for an accelerator runtime.                                   */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_ACCELERATOR_ACCELERATORINITIALIZER_H
#define ARCCORE_ACCELERATOR_ACCELERATORINITIALIZER_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/accelerator/AcceleratorGlobal.h"

#include <memory>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Accelerator
{
class Initializer;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Class to initialize a runtime for the accelerator API.
 *
 * \warning Experimental API currently being defined.
 *
 * Only one instance of this class can exist at any given time.
 */
class ARCCORE_ACCELERATOR_EXPORT AcceleratorInitializer
{
 public:

  //! Initializes a sequential runtime
  AcceleratorInitializer();

  /*!
   * \brief Initializes a runtime.
   *
   * If \a use_accelerator is true, the accelerator runtime is initialized
   * used to compile Arcane. In this case, executionPolicy() will return
   * this runtime.
   *
   * If \a nb_thread is greater than 1, the multi-threaded runtime is also
   * initialized.
   */
  explicit AcceleratorInitializer(bool use_accelerator, Int32 nb_thread = 1);

  ~AcceleratorInitializer();

 public:

  AcceleratorInitializer(const AcceleratorInitializer&) = delete;
  AcceleratorInitializer(AcceleratorInitializer&&) = delete;
  AcceleratorInitializer& operator=(const AcceleratorInitializer&) = delete;
  AcceleratorInitializer& operator=(AcceleratorInitializer&&) = delete;

 public:

  //! Default initialized execution policy
  eExecutionPolicy executionPolicy() const;

  //! Associated trace manager
  ITraceMng* traceMng() const;

 private:

  std::unique_ptr<Initializer> m_initializer;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::Accelerator

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
