// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ConstituentItemLocalId.h                                    (C) 2000-2024 */
/*                                                                           */
/* Index for material variables.                                             */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_MATERIALS_CONSTITUENTITEMLOCALID_H
#define ARCANE_CORE_MATERIALS_CONSTITUENTITEMLOCALID_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/materials/MatVarIndex.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Materials
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \ingroup ArcaneMaterials
 * \brief Index of a ConstituentItem in a variable.
 */
class ConstituentItemLocalId
{
 public:

  constexpr ARCCORE_HOST_DEVICE ConstituentItemLocalId()
  : m_local_id(-1, -1)
  {}
  constexpr ARCCORE_HOST_DEVICE explicit ConstituentItemLocalId(MatVarIndex mvi)
  : m_local_id(mvi)
  {}

 public:

  //! Generic index to access variable values.
  constexpr ARCCORE_HOST_DEVICE MatVarIndex localId() const { return m_local_id; }

 public:

  ARCANE_CORE_EXPORT friend std::ostream&
  operator<<(std::ostream& o, const ConstituentItemLocalId& mvi);

 private:

  MatVarIndex m_local_id;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \ingroup ArcaneMaterials
 * \brief Index of a MatItem in a variable.
 */
class MatItemLocalId
: public ConstituentItemLocalId
{
 public:

  MatItemLocalId() = default;
  constexpr ARCCORE_HOST_DEVICE explicit MatItemLocalId(MatVarIndex mvi)
  : ConstituentItemLocalId(mvi)
  {}
  constexpr ARCCORE_HOST_DEVICE MatItemLocalId(ComponentItemLocalId lid)
  : ConstituentItemLocalId(lid)
  {}
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \ingroup ArcaneMaterials
 * \brief Index of an EnvItem in a variable.
 */
class EnvItemLocalId
: public ConstituentItemLocalId
{
 public:

  EnvItemLocalId() = default;
  constexpr ARCCORE_HOST_DEVICE explicit EnvItemLocalId(MatVarIndex mvi)
  : ConstituentItemLocalId(mvi)
  {}
  constexpr ARCCORE_HOST_DEVICE EnvItemLocalId(ComponentItemLocalId lid)
  : ConstituentItemLocalId(lid)
  {}
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::Materials

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
