// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IMeshMaterialVariableSynchronizer.h                         (C) 2000-2016 */
/*                                                                           */
/* Interface du synchroniseur de variables matériaux.                        */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_MATERIALS_IMESHMATERIALVARIABLESYNCHRONIZER_H
#define ARCANE_MATERIALS_IMESHMATERIALVARIABLESYNCHRONIZER_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/materials/MaterialsGlobal.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class IVariableSynchronizer;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MATERIALS_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class MatVarIndex;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * \brief Interface du synchroniseur de variables matériaux.
 */
class ARCANE_MATERIALS_EXPORT IMeshMaterialVariableSynchronizer
{
 public:

  virtual ~IMeshMaterialVariableSynchronizer(){}

 public:

  //! Synchroniseur des variables classiques associé.
  virtual IVariableSynchronizer* variableSynchronizer() =0;

  /*!
   * \brief Liste des MatVarIndex partagés pour le rang d'indice \a index
   * dans le tableau variableSynchronizer::communicatingRanks();
   */
  virtual ConstArrayView<MatVarIndex> sharedItems(Int32 index) =0;

  /*!
   * \brief Liste des MatVarIndex fantômes pour le rang d'indice \a index
   * dans le tableau variableSynchronizer::communicatingRanks();
   */
  virtual ConstArrayView<MatVarIndex> ghostItems(Int32 index) =0;

  //! Recalcule les infos de synchronisation.
  virtual void recompute() =0;

  //! Recalcule les infos de synchronisation si nécessaire.
  virtual void checkRecompute() =0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MATERIALS_END_NAMESPACE
ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

