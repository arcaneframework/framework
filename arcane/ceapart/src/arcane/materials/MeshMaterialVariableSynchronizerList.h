// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MeshMaterialVariableSynchronizerList.h                      (C) 2000-2016 */
/*                                                                           */
/* Liste de variables à synchroniser.                                        */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_MATERIALS_MESHMATERIALVARIABLESYNCHRONIZERLIST_H
#define ARCANE_MATERIALS_MESHMATERIALVARIABLESYNCHRONIZERLIST_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/TraceAccessor.h"
#include "arcane/utils/Array.h"

#include "arcane/materials/IMeshMaterialVariableSynchronizer.h"
#include "arcane/materials/MatVarIndex.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MATERIALS_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class MeshMaterialVariable;
class IMeshMaterialMng;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * \brief Liste de variables à synchroniser.
 */
class ARCANE_MATERIALS_EXPORT MeshMaterialVariableSynchronizerList
{
 private:

  class Impl;

 public:

  MeshMaterialVariableSynchronizerList(IMeshMaterialMng* material_mng);

  virtual ~MeshMaterialVariableSynchronizerList();

 public:

  void apply();
  void add(MeshMaterialVariable* var);

 private:

  Impl* m_p;

  void _synchronizeMultiple(ConstArrayView<MeshMaterialVariable*> vars,
                            IMeshMaterialVariableSynchronizer* mmvs);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MATERIALS_END_NAMESPACE
ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

