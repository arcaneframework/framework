// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* DeviceInfo.h                                                (C) 2000-2025 */
/*                                                                           */
/* Information sur un device.                                                */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_COMMON_ACCELERATOR_DEVICEINFO_H
#define ARCCORE_COMMON_ACCELERATOR_DEVICEINFO_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/base/String.h"
#include "arccore/common/accelerator/DeviceId.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Accelerator
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Information sur un device.
 */
class ARCCORE_COMMON_EXPORT DeviceInfo
{
 public:

  //! Informations du i-ème device.
  DeviceId deviceId() const { return m_device_id; }

  //! Nom du device
  String name() const { return m_name; }

  //! UUID sous forme de chaîne de caractères. Peut-être nul.
  String uuidAsString() const { return m_uuid_as_string; }

  //! Description du device.
  String description() const { return m_description; }

  //! Taille d'un warp
  Int32 warpSize() const { return m_warp_size; }

 public:

  void setDeviceId(DeviceId id) { m_device_id = id; }
  void setUUIDAsString(const String& v) { m_uuid_as_string = v; }
  void setDescription(const String& v) { m_description = v; }
  void setName(const String& v) { m_name = v; }
  void setWarpSize(Int32 v) { m_warp_size = v; }

 private:

  DeviceId m_device_id;
  String m_name;
  String m_uuid_as_string;
  String m_description;
  Int32 m_warp_size = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::Accelerator

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
