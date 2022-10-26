// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* DeviceInfo.h                                                (C) 2000-2022 */
/*                                                                           */
/* Information sur un device.                                                */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_ACCELERATOR_CORE_DEVICEINFO_H
#define ARCANE_ACCELERATOR_CORE_DEVICEINFO_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/String.h"
#include "arcane/accelerator/core/DeviceId.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Accelerator
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Information sur un device.
 */
class ARCANE_ACCELERATOR_CORE_EXPORT DeviceInfo
{
 public:

  //! Informations du i-ème device.
  DeviceId id() const { return m_device_id; }

  //! UUID sous forme de chaîne de caractères. Peut-être nul.
  String uuidAsString() const { return m_uuid_as_string; }

 public:

  void setId(DeviceId id) { m_device_id = id; }
  void setUUIDAsString(const String& v) { m_uuid_as_string = v; }

 private:

  DeviceId m_device_id;
  String m_uuid_as_string;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::Accelerator

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
