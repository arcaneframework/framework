// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MeshMaterialVariableSynchronizerList.h                      (C) 2000-2022 */
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

namespace Arcane::Materials
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * \brief Synchronisation d'une liste de variables matériaux.
 *
 * La méthode add() permet d'ajouter des variables à synchroniser.
 * Il faut ensuite appeler apply() pour effectuer la synchronisation.
 *
 * Une instance de ce cette classe peut-être utilisée plusieurs fois.
 */
class ARCANE_MATERIALS_EXPORT MeshMaterialVariableSynchronizerList
{
 private:

  class Impl;

 public:

  MeshMaterialVariableSynchronizerList(IMeshMaterialMng* material_mng);

  virtual ~MeshMaterialVariableSynchronizerList();

 public:

  //! Effectue la synchronisation
  void apply();

  //! Ajoute la variable \a var à la liste des variables à synchroniser
  void add(MeshMaterialVariable* var);

  //! Après appel à apply(), contient la taille des messages envoyés
  Int64 totalMessageSize() const;

 private:

  Impl* m_p;

 private:

  void _synchronizeMultiple(ConstArrayView<MeshMaterialVariable*> vars,
                            IMeshMaterialVariableSynchronizer* mmvs);
  void _synchronizeMultiple2(ConstArrayView<MeshMaterialVariable*> vars,
                             IMeshMaterialVariableSynchronizer* mmvs,
                             IMeshMaterialSynchronizeBuffer* buf_list);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::Materials

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

