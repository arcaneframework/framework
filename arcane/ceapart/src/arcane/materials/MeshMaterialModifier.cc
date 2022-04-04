﻿// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MeshMaterialModifier.cc                                     (C) 2000-2015 */
/*                                                                           */
/* Objet permettant de modifier les matériaux.                               */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/ArrayView.h"
#include "arcane/utils/TraceInfo.h"

#include "arcane/materials/MeshMaterialModifier.h"
#include "arcane/materials/IMeshMaterialMng.h"
#include "arcane/materials/IMeshMaterialModifierImpl.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE
MATERIALS_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MeshMaterialModifier::
MeshMaterialModifier(IMeshMaterialMng* mm)
: m_impl(mm->modifier())
, m_has_update(false)
{
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
  if (!m_has_update){
    m_impl->beginUpdate();
    m_has_update = true;
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshMaterialModifier::
addCells(IMeshMaterial* mat,Int32ConstArrayView ids)
{
  _checkHasUpdate();
  m_impl->addCells(mat,ids);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshMaterialModifier::
removeCells(IMeshMaterial* mat,Int32ConstArrayView ids)
{
  _checkHasUpdate();
  m_impl->removeCells(mat,ids);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshMaterialModifier::
setCells(IMeshMaterial* mat,Int32ConstArrayView ids)
{
  _checkHasUpdate();
  m_impl->setCells(mat,ids);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshMaterialModifier::
endUpdate()
{
  if (m_has_update){
    // IMPORTANT: il faut mettre la m_has_update à \a false avant
    // l'appel car si m_impl->endUpdate() lève une exception, on va rejouer
    // la mise à jour dans le destructeur de cette classe.
    m_has_update = false;
    m_impl->endUpdate();
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MATERIALS_END_NAMESPACE
ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
