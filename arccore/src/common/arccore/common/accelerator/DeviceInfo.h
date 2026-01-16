// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* DeviceInfo.h                                                (C) 2000-2025 */
/*                                                                           */
/* Information sur un accélérateur.                                          */
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
 * \brief Information sur un accélérateur.
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

  //! Mémoire locale (en octet) par bloc
  Int32 sharedMemoryPerBlock() const { return m_shared_memory_per_block; }
  //! Mémoire locale (en octet) par SM
  Int32 sharedMemoryPerMultiprocessor() const { return m_shared_memory_per_multiprocessor; }
  //! Mémoire locale (en octet) par bloc qui peut s'activer sur option
  Int32 sharedMemoryPerBlockOptin() const { return m_shared_memory_per_block_optin; }
  //! Mémoire constante (en octet)
  Int32 totalConstMemory() const { return m_total_const_memory; }
  //! Identifiant du domaine PCI (-1 si inconnu)
  int pciDomainID() const { return m_pci_domain_id; }
  //! Identifiant du bus PCI (-1 si inconnu)
  int pciBusID() const { return m_pci_bus_id; }
  //! Identifiant PCI de l'accélérateur (-1 si inconnu)
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
  void setPCIDeviceID(int v ){ m_pci_device_id = v; }

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
