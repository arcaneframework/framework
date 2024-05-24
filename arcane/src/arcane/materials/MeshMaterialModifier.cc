// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MeshMaterialModifier.cc                                     (C) 2000-2024 */
/*                                                                           */
/* Objet permettant de modifier les matériaux.                               */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/materials/MeshMaterialModifier.h"

#include "arcane/utils/ArrayView.h"
#include "arcane/utils/TraceInfo.h"
#include "arcane/utils/FatalErrorException.h"

#include "arcane/core/materials/IMeshMaterialMng.h"
#include "arcane/core/materials/internal/IMeshMaterialMngInternal.h"

#include "arcane/materials/internal/MeshMaterialModifierImpl.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Materials
{
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MeshMaterialModifier::
MeshMaterialModifier(IMeshMaterialMng* mm)
: m_impl(mm->_internalApi()->modifier())
, m_has_update(false)
{
  if (!m_impl)
    ARCANE_FATAL("Can not create 'MeshMaterialModifier' because IMeshMaterialMng is not yet initialized");
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MeshMaterialModifier::
~MeshMaterialModifier() ARCANE_NOEXCEPT_FALSE
{
  endUpdate();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshMaterialModifier::
_checkHasUpdate()
{
  if (!m_has_update) {
    m_impl->beginUpdate();
    m_has_update = true;
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshMaterialModifier::
addCells(IMeshMaterial* mat, SmallSpan<const Int32> ids)
{
  _checkHasUpdate();
  m_impl->addCells(mat, ids);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshMaterialModifier::
removeCells(IMeshMaterial* mat, SmallSpan<const Int32> ids)
{
  _checkHasUpdate();
  m_impl->removeCells(mat, ids);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshMaterialModifier::
endUpdate()
{
  if (m_has_update) {
    // IMPORTANT: il faut mettre la m_has_update à \a false avant
    // l'appel car si m_impl->endUpdate() lève une exception, on va rejouer
    // la mise à jour dans le destructeur de cette classe.
    m_has_update = false;
    m_impl->endUpdate();
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshMaterialModifier::
setDoCopyBetweenPartialAndPure(bool v)
{
  m_impl->setDoCopyBetweenPartialAndPure(v);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshMaterialModifier::
setDoInitNewItems(bool v)
{
  m_impl->setDoInitNewItems(v);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::Materials

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
