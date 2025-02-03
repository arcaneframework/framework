// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* Parallel.h                                                  (C) 2000-2025 */
/*                                                                           */
/* Espace de nom des types gérant le parallélisme.                           */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_PARALLEL_H
#define ARCANE_CORE_PARALLEL_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/ArcaneGlobal.h"
#include "arcane/utils/ArrayView.h"

#include "arccore/base/RefDeclarations.h"
#include "arccore/message_passing/Request.h"
#include "arccore/message_passing/Communicator.h"
#include "arccore/message_passing/PointToPointMessageInfo.h"
#include "arccore/message_passing/IControlDispatcher.h"

#include "arcane/core/ArcaneTypes.h"

#define ARCANE_BEGIN_NAMESPACE_PARALLEL \
  namespace Parallel \
  {
#define ARCANE_END_NAMESPACE_PARALLEL }

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Types des classes du parallélisme.
 */
namespace Arcane::Parallel
{
class IStat;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Déclarations des types et méthodes utilisés par les mécanismes
 * d'échange de messages.
 */
namespace Arcane::MessagePassing
{
/*!
 * \brief Effectue une barrière nommée de nom \a name
 *
 * Effectue une barrière de nom \a name en utilisant le gestionnaire
 * \a pm.
 *
 * Tous les rangs de \a pm bloquent dans cette barrière et vérifient
 * que tous les rangs utilisent le même nom de barrière. Si un des rangs
 * utilise un nom différent une exception est levée.
 *
 * Cette opération permet de vérifier que tous les rangs utilisent
 * la même barrière contrairement à l'opération IParallelMng::barrier().
 *
 * \note Seuls les 1024 premiers caractères de \a name sont utilisés.
 */
extern "C++" ARCANE_CORE_EXPORT void
namedBarrier(IParallelMng* pm, const String& name);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Filtre les chaînes de caractères communes à tous les rangs de \a pm.
 *
 * Prend en entrée une liste \a input_string de chaînes de caractères et retourne
 * dans \a common_strings celles qui sont communes à tous les rangs de \a pm.
 * Les chaînes de caractères retournées dans \a common_strings sont triées
 * par ordre alphabétique.
 */
extern "C++" ARCANE_CORE_EXPORT void
filterCommonStrings(IParallelMng* pm, ConstArrayView<String> input_strings,
                    Array<String>& common_strings);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Écrit dans \a tm la date et la mémoire consommée.
 *
 * L'opération est collective sur \a pm et affiche la mémoire minimimale,
 * moyenne et maximale consommée ainsi que les rangs de ceux qui consomment
 * le moins et le plus de mémoire.
 */
extern "C++" ARCANE_CORE_EXPORT void
dumpDateAndMemoryUsage(IParallelMng* pm, ITraceMng* tm);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::MessagePassing

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

inline std::ostream&
operator<<(std::ostream& o, const Parallel::Request prequest)
{
  prequest.print(o);
  return o;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

