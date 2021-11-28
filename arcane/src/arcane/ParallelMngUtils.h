// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ParallelMngUtils.h                                          (C) 2000-2021 */
/*                                                                           */
/* Fonctions utilitaires associées aux 'IParallelMng'.                       */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_PARALLELMNGUTILS_H
#define ARCANE_PARALLELMNGUTILS_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/UtilsTypes.h"

#include "arcane/Parallel.h"

namespace Arcane
{
class IItemFamily;
class ItemGroup;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Fonctions utilitaires associées à IParallelMng.
 */

namespace Arcane::ParallelMngUtils
{
/*!
 * \brief Retourne une opération pour récupérer les valeurs d'une variable
 * sur les entités d'un autre sous-domaine.
 */
extern "C++" ARCANE_CORE_EXPORT Ref<IGetVariablesValuesParallelOperation>
createGetVariablesValuesOperationRef(IParallelMng* pm);

//! Retourne une opération pour transférer des valeurs entre rangs.
extern "C++" ARCANE_CORE_EXPORT Ref<ITransferValuesParallelOperation>
createTransferValuesOperationRef(IParallelMng* pm);

//! Retourne une interface pour transférer des messages entre rangs
extern "C++" ARCANE_CORE_EXPORT Ref<IParallelExchanger>
createExchangerRef(IParallelMng* pm);

/*!
 * \brief Retourne une interface pour synchroniser des
 * variables sur le groupe de la famille \a family
 */
extern "C++" ARCANE_CORE_EXPORT Ref<IVariableSynchronizer>
createSynchronizerRef(IParallelMng* pm,IItemFamily* family);

/*!
 * \brief Retourne une interface pour synchroniser des
 * variables sur le groupe \a group.
 */
extern "C++" ARCANE_CORE_EXPORT Ref<IVariableSynchronizer>
createSynchronizerRef(IParallelMng* pm,const ItemGroup& group);

/*!
 * \brief Créé une instance contenant les infos sur la topologie des rangs de ce gestionnnaire.
 *
 * Cette opération est collective.
 */
extern "C++" ARCANE_CORE_EXPORT Ref<IParallelTopology>
createTopologyRef(IParallelMng* pm);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::ParallelMngUtils

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
