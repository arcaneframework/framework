// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2023 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* DataAllocationInfo.h                                        (C) 2000-2023 */
/*                                                                           */
/* Informations sur l'allocation d'une donnée.                               */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_DATATYPES_DATAALLOCATIONINFO_H
#define ARCANE_DATATYPES_DATAALLOCATIONINFO_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/ArcaneTypes.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Informations sur l'allocation d'une donnée.
 */
class ARCANE_DATATYPE_EXPORT DataAllocationInfo
{
 public:

  //! Constructeur.
  DataAllocationInfo() = default;
  DataAllocationInfo(eMemoryLocationHint hint)
  : m_location_hint(hint)
  {}

 public:

  eMemoryLocationHint memoryLocationHint() const { return m_location_hint; }
  void setMemoryLocationHint(eMemoryLocationHint hint) { m_location_hint = hint; }

 public:

  friend bool operator==(const DataAllocationInfo& a, const DataAllocationInfo& b)
  {
    return a.m_location_hint == b.m_location_hint;
  }

 private:

  eMemoryLocationHint m_location_hint = eMemoryLocationHint::None;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
