// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ParallelMngUtils.h                                          (C) 2000-2025 */
/*                                                                           */
/* Fonctions utilitaires associées aux 'IParallelMng'.                       */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_PARALLELMNGUTILS_H
#define ARCANE_CORE_PARALLELMNGUTILS_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/UtilsTypes.h"

#include "arcane/core/ArcaneTypes.h"
#include "arcane/core/Parallel.h"

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
createSynchronizerRef(IParallelMng* pm, IItemFamily* family);

/*!
 * \brief Retourne une interface pour synchroniser des
 * variables sur le groupe \a group.
 */
extern "C++" ARCANE_CORE_EXPORT Ref<IVariableSynchronizer>
createSynchronizerRef(IParallelMng* pm, const ItemGroup& group);

/*!
 * \brief Créé une instance contenant les infos sur la topologie des rangs de ce gestionnnaire.
 *
 * Cette opération est collective.
 */
extern "C++" ARCANE_CORE_EXPORT Ref<IParallelTopology>
createTopologyRef(IParallelMng* pm);

/*!
 * \brief Créé un nouveau gestionnaire de parallélisme pour un sous-ensemble
 * des rangs.
 *
 * Cette opération est collective et est équivalent à MPI_Comm_split.
 *
 * Les rangs dont \a color vaut la même valeur seront dans le même communicateur.
 * \a key permet d'ordonner les rangs dans le sous-communicateur créé. S'il vaut
 * pm->commRank() alors les rangs dans le sous-communicateur auront le même ordre
 * que dans \a pm.
 *
 * * Si \a color est négatif, alors le rang actuel ne sera associé à aucun
 * communicateur et la valeur retournée sera nulle.
 */
extern "C++" ARCANE_CORE_EXPORT Ref<IParallelMng>
createSubParallelMngRef(IParallelMng* pm, Int32 color, Int32 key);

/*!
 * \brief Créé un message de sérialisation non bloquant en envoi au rang \a rank.
 *
 * Le message est traité uniquement lors de l'appel à IParallelMng::processMessages().
 */
extern "C++" ARCANE_CORE_EXPORT Ref<ISerializeMessage>
createSendSerializeMessageRef(IParallelMng* pm, Int32 rank);

/*!
 * \brief Créé un message de sérialisation non bloquant en réception du rang \a rank.
 *
 * Le message est traité uniquement lors de l'appel à IParallelMng::processMessages().
 */
extern "C++" ARCANE_CORE_EXPORT Ref<ISerializeMessage>
createReceiveSerializeMessageRef(IParallelMng* pm, Int32 rank);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::ParallelMngUtils

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
