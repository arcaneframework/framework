// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ITimeHistoryAdder.h                                         (C) 2000-2024 */
/*                                                                           */
/* Interface de classe permettant d'ajouter un historique de valeur lié à    */
/* un maillage.                                                              */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#ifndef ARCANE_ITIMEHISTORYMNGADDER_H
#define ARCANE_ITIMEHISTORYMNGADDER_H

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/UtilsTypes.h"
#include "arcane/utils/FatalErrorException.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Classe gérant un historique de valeurs.
 *
 Le gestionnaire d'historique gère l'historique d'un ensemble de valeur au
 cours du temps.
 
 L'historique est basée par itération (VariablesCommon::globalIteration()).
 Pour chaque itération, il est possible de sauver une valeur par
 l'intermédiaire des méthodes addValue(). Il n'est pas obligatoire d'avoir
 une valeur pour chaque itération. Lorsqu'on effectue plusieurs addValue()
 pour le même historique à la même itération, seule la dernière valeur
 est prise en compte.

 Chaque historique est associée à un nom qui est le nom du fichier dans
 lequel la liste des valeurs sera sauvegardée.

 Seul l'instance associée au sous-domaine tel que parallelMng()->isMasterIO()
 est vrai enregistre les valeurs. Pour les autres, les appels à addValue()
 sont sans effet.

 Les valeurs ne sont sauvées que si active() est vrai. Il est possible
 de modifier l'état d'activation en appelant isActive().

 En mode debug, l'ensemble des historiques est sauvé à chaque pas de temps.
 En exécution normale, cet ensemble est sauvé toute les \a n itérations, \a n
 étant donné par l'option du jeu de donné
 <module-main/time-history-iteration-step>. Dans tous les cas, une sortie
 est effectuée à la fin de l'exécution.

 Le format de ces fichiers dépend de l'implémentation.

 \since 0.4.38
 */
class ITimeHistoryAdder
{
 public:
  virtual ~ITimeHistoryAdder() = default; //!< Libère les ressources

 public:
  virtual void addValue(const String& name, Real value, bool end_time, bool is_local) = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

