// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* AcceleratorRuntimeInitialisationInfo.h                      (C) 2000-2025 */
/*                                                                           */
/* Information for accelerator runtime initialization.                       */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_COMMON_ACCELERATOR_ACCELERATORRUNTIMEINITIALISATIONINFO_H
#define ARCCORE_COMMON_ACCELERATOR_ACCELERATORRUNTIMEINITIALISATIONINFO_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/common/accelerator/CommonAcceleratorGlobal.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Accelerator
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Information for accelerator initialization.
 */
class ARCCORE_COMMON_EXPORT AcceleratorRuntimeInitialisationInfo
{
 private:

  class Impl;

 public:

  AcceleratorRuntimeInitialisationInfo();
  AcceleratorRuntimeInitialisationInfo(const AcceleratorRuntimeInitialisationInfo& rhs);
  ~AcceleratorRuntimeInitialisationInfo();
  AcceleratorRuntimeInitialisationInfo& operator=(const AcceleratorRuntimeInitialisationInfo& rhs);

 public:

  //! Indicates if an accelerator runtime is used
  void setIsUsingAcceleratorRuntime(bool v);
  bool isUsingAcceleratorRuntime() const;

  //! Name of the runtime used (for now only 'cuda', 'hip' or 'sycl')
  void setAcceleratorRuntime(StringView name);
  String acceleratorRuntime() const;

  //! Positions the device associated with the Runner.
  void setDeviceId(DeviceId name);
  //! Device associated with the Runner
  DeviceId deviceId() const;

  //! Execution policy associated with acceleratorRuntime()
  eExecutionPolicy executionPolicy() const;

 private:

  Impl* m_p;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::Accelerator

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
