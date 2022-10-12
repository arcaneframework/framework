// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* DeviceId.h                                                  (C) 2000-2022 */
/*                                                                           */
/* Identifiant d'un composant du système.                                    */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_ACCELERATOR_CORE_DEVICEID_H
#define ARCANE_ACCELERATOR_CORE_DEVICEID_H
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
 * \brief Identifiant d'un composant du système.
 *
 * Le composant peut être un accélérateur ou l'hôte.
 */
class ARCANE_ACCELERATOR_CORE_EXPORT DeviceId
{
 private:

  static constexpr Int32 HOST_ID = (-1);

 public:

  //! Accélérateur par défaut
  DeviceId() = default;

  explicit DeviceId(Int32 id)
  : m_device_id(id)
  {
  }

 public:

  //! Device représentant l'hôte.
  static DeviceId host() { return DeviceId(HOST_ID); }

 public:

  bool isHost() const { return m_device_id == HOST_ID; }
  Int32 asInt32() const { return m_device_id; }

 private:

  Int32 m_device_id = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
