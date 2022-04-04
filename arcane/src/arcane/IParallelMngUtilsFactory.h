﻿// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IParallelMngUtilsFactory.h                                  (C) 2000-2021 */
/*                                                                           */
/* Interface d'une fabrique pour les fonctions utilitaires de IParallelMng.  */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_IPARALLELMNGUTILSFACTORY_H
#define ARCANE_IPARALLELMNGUTILSFACTORY_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/Ref.h"

#include "arcane/Parallel.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{
class IItemFamily;
class ItemGroup;
}

namespace Arcane
{
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * \brief Interface d'une fabrique pour les fonctions utilitaires de IParallelMng.
 */
class ARCANE_CORE_EXPORT IParallelMngUtilsFactory
{
 public:
  virtual ~IParallelMngUtilsFactory() = default;
 public:

  /*!
   * \brief Retourne une opération pour récupérer les valeurs d'une variable
   * sur les entités d'un autre sous-domaine.
   */
  virtual Ref<IGetVariablesValuesParallelOperation>
  createGetVariablesValuesOperation(IParallelMng* pm) =0;

  //! Retourne une opération pour transférer des valeurs entre rangs.
  virtual Ref<ITransferValuesParallelOperation>
  createTransferValuesOperation(IParallelMng* pm) =0;

  //! Retourne une interface pour transférer des messages entre rangs
  virtual Ref<IParallelExchanger>
  createExchanger(IParallelMng* pm) =0;

  /*!
   * \brief Retourne une interface pour synchroniser des
   * variables sur le groupe de la famille \a family
   */
  virtual Ref<IVariableSynchronizer>
  createSynchronizer(IParallelMng* pm,IItemFamily* family) =0;

  /*!
   * \brief Retourne une interface pour synchroniser des
   * variables sur le groupe \a group.
   */
  virtual Ref<IVariableSynchronizer>
  createSynchronizer(IParallelMng* pm,const ItemGroup& group) =0;

  /*!
   * \brief Créé une instance contenant les infos sur la topologie des rangs de ce gestionnnaire.
   *
   * Cette opération est collective.
   */
  virtual Ref<IParallelTopology>
  createTopology(IParallelMng* pm) =0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::impl

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
