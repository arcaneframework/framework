// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* DeviceInfo.h                                                (C) 2000-2025 */
/*                                                                           */
/* Information about an accelerator.                                         */
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
 * \brief Information about an accelerator.
 */
class ARCCORE_COMMON_EXPORT DeviceInfo
{
 public:

  //! Information of the i-th device.
  DeviceId deviceId() const { return m_device_id; }

  //! Device name
  String name() const { return m_name; }

  //! UUID as a string. May be null.
  String uuidAsString() const { return m_uuid_as_string; }

  //! Device description.
  String description() const { return m_description; }

  //! Warp size
  Int32 warpSize() const { return m_warp_size; }

  //! Local memory (in bytes) per block
  Int32 sharedMemoryPerBlock() const { return m_shared_memory_per_block; }
  //! Local memory (in bytes) per SM
  Int32 sharedMemoryPerMultiprocessor() const { return m_shared_memory_per_multiprocessor; }
  //! Local memory (in bytes) per block which can be optionally enabled
  Int32 sharedMemoryPerBlockOptin() const { return m_shared_memory_per_block_optin; }
  //! Constant memory (in bytes)
  Int32 totalConstMemory() const { return m_total_const_memory; }
  //! PCI domain ID (-1 if unknown)
  int pciDomainID() const { return m_pci_domain_id; }
  //! PCI bus ID (-1 if unknown)
  int pciBusID() const { return m_pci_bus_id; }
  //! Accelerator PCI ID (-1 if unknown)
  int pciDeviceID() const { return m_pci_device_id; }

 public:

  void setDeviceId(DeviceId id) { m_device_id = id; }
  void setUUIDAsString(const String& v) { m_uuid_as_string = v; }
  void setDescription(const String& v) { m_description = v; }
  void setName(const String& v) { m_name = v; }
  void setWarpSize(Int32 v) { m_warp_size = v; }
  void setSharedMemoryPerBlock(Int32 v) { m_shared_memory_per_block = v; }
  void setSharedMemoryPerMultiprocessor(Int32 v) { m_shared_memory_per_multiprocessor = v; }
  void setSharedMemoryPerBlockOptin(Int32 v) { m_shared_memory_per_block_optin = v; }
  void setTotalConstMemory(Int32 v) { m_total_const_memory = v; }
  void setPCIDomainID(int v) { m_pci_domain_id = v; }
  void setPCIBusID(int v) { m_pci_bus_id = v; }
  void setPCIDeviceID(int v) { m_pci_device_id = v; }

 private:

  DeviceId m_device_id;
  String m_name;
  String m_uuid_as_string;
  String m_description;
  Int32 m_warp_size = 0;
  Int32 m_shared_memory_per_block = 0;
  Int32 m_shared_memory_per_multiprocessor = 0;
  Int32 m_shared_memory_per_block_optin = 0;
  Int32 m_total_const_memory = 0;
  int m_pci_domain_id = -1;
  int m_pci_bus_id = -1;
  int m_pci_device_id = -1;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::Accelerator

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
