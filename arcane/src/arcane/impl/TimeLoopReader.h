// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* TimeLoopReader.h                                            (C) 2000-2006 */
/*                                                                           */
/* Chargement d'une boucle en temps.                                         */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_MAIN_TIMELOOPREADER_H
#define ARCANE_MAIN_TIMELOOPREADER_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/List.h"
#include "arcane/utils/String.h"
#include "arcane/utils/TraceAccessor.h"

#include "arcane/ArcaneTypes.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ITimeLoop;
class IApplication;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Functor de chargement d'une boucle en temps.
 *
 * A partir du jeu de donnée et des options générales, lit le nom de la
 * boucle en temps et l'indique au gestionnaire #m_mng.
 */
class ARCANE_IMPL_EXPORT TimeLoopReader
: public TraceAccessor
{
 public:

  //! Crée une instance associée au gestionnaire \a sm
  TimeLoopReader(IApplication* sm);
  ~TimeLoopReader(); //!< Libère les ressources

 public:

  //! Effectue la lecture des boucles en temps disponible.
  void readTimeLoops();

  //! Enregistre la liste des boucles en temps dans le gestionnaire \a sd
  void registerTimeLoops(ISubDomain* sd);

  //! Positionne la boucle en temps utilisée dans le gestionnaire \a sd
  void setUsedTimeLoop(ISubDomain* sd);

  //! nom de la boucle en temps à exécuter.
  const String& timeLoopName() const { return m_time_loop_name; }

  //! Liste des boucles en temps lues
  TimeLoopCollection timeLoops() const { return m_time_loops; }

 private:

  IApplication* m_application; //!< Superviseur.
  TimeLoopList m_time_loops;
  String m_time_loop_name;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  
