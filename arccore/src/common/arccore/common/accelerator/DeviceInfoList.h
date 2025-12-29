// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* DeviceInfoList.h                                            (C) 2000-2025 */
/*                                                                           */
/* Liste de devices.                                                         */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_COMMON_ACCELERATOR_DEVICEINFOLIST_H
#define ARCCORE_COMMON_ACCELERATOR_DEVICEINFOLIST_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/common/Array.h"

#include "arccore/common/accelerator/IDeviceInfoList.h"
#include "arccore/common/accelerator/DeviceInfo.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Accelerator::Impl
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Interface d'une liste de devices.
 */
class ARCCORE_COMMON_EXPORT DeviceInfoList
: public IDeviceInfoList
{
 public:

  //! Nombre de device de la liste
  Int32 nbDevice() const override { return m_devices.size(); }

  //! Informations du i-ème device.
  const DeviceInfo& deviceInfo(Int32 i) const override { return m_devices[i]; }

 public:

  void addDevice(const DeviceInfo& d) { m_devices.add(d); }

 private:

  UniqueArray<DeviceInfo> m_devices;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::Accelerator::Impl

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
