// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2023 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IMeshMaterialVariableSynchronizer.h                         (C) 2000-2023 */
/*                                                                           */
/* Interface du synchroniseur de variables matériaux.                        */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_MATERIALS_IMESHMATERIALVARIABLESYNCHRONIZER_H
#define ARCANE_MATERIALS_IMESHMATERIALVARIABLESYNCHRONIZER_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/materials/MaterialsGlobal.h"

#include "arcane/utils/ArrayView.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{
class IVariableSynchronizer;
}

namespace Arcane::Materials
{

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

  //! Buffer commun pour les messages.
  virtual Ref<IMeshMaterialSynchronizeBuffer> commonBuffer() =0;

  //! Ressource mémoire à utiliser pour les buffers de communication
  virtual eMemoryRessource bufferMemoryRessource() const =0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::Materials

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

