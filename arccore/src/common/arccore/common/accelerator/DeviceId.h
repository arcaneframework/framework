// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* DeviceId.h                                                  (C) 2000-2025 */
/*                                                                           */
/* Identifiant d'un composant du système.                                    */
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
 * \brief Identifiant d'un composant du système.
 *
 * Le composant peut être un accélérateur ou l'hôte.
 */
class ARCCORE_COMMON_EXPORT DeviceId
{
 private:

  static constexpr Int32 HOST_ID = (-1);
  static constexpr Int32 NULL_ID = (-2);

 public:

  //! Accélérateur par défaut (Device de numéro 0)
  DeviceId() = default;

  explicit DeviceId(Int32 id)
  : m_device_id(id)
  {
  }

 public:

  //! Device représentant l'hôte.
  static DeviceId hostDevice() { return DeviceId(HOST_ID); }

  //! Device nulle ou invalide.
  static DeviceId nullDevice() { return DeviceId(NULL_ID); }

 public:

  //! Indique si l'instance est associée à l'hôte.
  bool isHost() const { return m_device_id == HOST_ID; }

  //! Indique si l'instance n'est associée à aucune device
  bool isNull() const { return m_device_id == NULL_ID; }

  //! Indique si l'instance est associée à un accélérateur
  bool isAccelerator() const { return m_device_id >= 0; }

  //! Valeur numérique du device.
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
