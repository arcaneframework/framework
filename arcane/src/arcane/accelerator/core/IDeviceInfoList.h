// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IDeviceInfoList.h                                           (C) 2000-2022 */
/*                                                                           */
/* Interface d'une liste de devices.                                         */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_ACCELERATOR_CORE_IDEVICEINFOLIST_H
#define ARCANE_ACCELERATOR_CORE_IDEVICEINFOLIST_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/accelerator/core/AcceleratorCoreGlobal.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Accelerator
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Interface d'une liste de devices.
 */
class ARCANE_ACCELERATOR_CORE_EXPORT IDeviceInfoList
{
 public:

  virtual ~IDeviceInfoList() = default;

 public:

  //! Nombre de device de la liste
  virtual Int32 nbDevice() const = 0;

  //! Informations du i-ème device.
  virtual const DeviceInfo& deviceInfo(Int32 i) const = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
