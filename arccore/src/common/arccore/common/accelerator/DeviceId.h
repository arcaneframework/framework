// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* DeviceId.h                                                  (C) 2000-2025 */
/*                                                                           */
/* Identifier of a system component.                                         */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_COMMON_ACCELERATOR_DEVICEID_H
#define ARCCORE_COMMON_ACCELERATOR_DEVICEID_H
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
 * \brief Identifier of a system component.
 *
 * The component can be an accelerator or the host.
 */
class ARCCORE_COMMON_EXPORT DeviceId
{
 private:

  static constexpr Int32 HOST_ID = (-1);
  static constexpr Int32 NULL_ID = (-2);

 public:

  //! Default accelerator (Device number 0)
  DeviceId() = default;

  explicit DeviceId(Int32 id)
  : m_device_id(id)
  {
  }

 public:

  //! Device representing the host.
  static DeviceId hostDevice() { return DeviceId(HOST_ID); }

  //! Null or invalid device.
  static DeviceId nullDevice() { return DeviceId(NULL_ID); }

 public:

  //! Indicates if the instance is associated with the host.
  bool isHost() const { return m_device_id == HOST_ID; }

  //! Indicates if the instance is not associated with any device
  bool isNull() const { return m_device_id == NULL_ID; }

  //! Indicates if the instance is associated with an accelerator
  bool isAccelerator() const { return m_device_id >= 0; }

  //! Numerical value of the device.
  Int32 asInt32() const { return m_device_id; }

 public:

  friend ARCCORE_COMMON_EXPORT
  std::ostream&
  operator<<(std::ostream& o, const DeviceId& device_id);

 private:

  Int32 m_device_id = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
