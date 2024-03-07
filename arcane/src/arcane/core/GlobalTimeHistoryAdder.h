// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* GlobalTimeHistoryAdder.h                                    (C) 2000-2024 */
/*                                                                           */
/* Classe permettant d'ajouter un historique de valeur global.               */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_GLOBALTIMEHISTORYADDER_H
#define ARCANE_CORE_GLOBALTIMEHISTORYADDER_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/ITimeHistoryAdder.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Classe permettant d'ajouter une ou plusieurs valeurs à un
 * historique de valeurs.
 *
 * Cette classe enregistrera les courbes de manière globale, sans support.
 * C'est-à-dire que les courbes ne seront liée qu'au domaine complet ou au
 * sous-domaine demandé, par opposition au MeshTimeHistoryAdder qui lie les
 * courbes au maillage désiré.
 *
 * Pour un nom d'historique donné, il ne peut y avoir qu'une courbe de une
 * ou plusieurs valeurs par sous-domaine (et une globale à tous les
 * sous-domaines).
 *
 * Exemple : plusieurs courbes de moyennes des pressions (appelons-les
 * "avg_pressure") et deux sous-domaines (0 et 1). Une valeur par itération.
 * - Une courbe "avg_pressure" liée au sous-domaine 0. Chaque valeur est la
 *   moyenne des pressions de chaque maille du sous-domaine 0.
 * - Une courbe "avg_pressure" liée au sous-domaine 1. Chaque valeur est la
 *   moyenne des pressions de chaque maille du sous-domaine 1.
 * - Une courbe "avg_pressure" liée au domaine complet. Chaque valeur est la
 *   moyenne des pressions de chaque sous-domaine.
 *
 * On peut remarquer qu'il est possible d'avoir plusieurs courbes
 * indépendantes avec le même nom mais liée à des sous-domaines différents
 * (+1 courbe globale).
 */
class ARCANE_CORE_EXPORT GlobalTimeHistoryAdder
: public ITimeHistoryAdder
{
 public:
  /*!
   * \brief Constructeur.
   *
   * \param time_history_mng Un pointeur vers une instance de ITimeHistoryMng.
   */
  explicit GlobalTimeHistoryAdder(ITimeHistoryMng* time_history_mng);
  ~GlobalTimeHistoryAdder() override = default;

 public:
  void addValue(const TimeHistoryAddValueArg& thp, Real value) override;
  void addValue(const TimeHistoryAddValueArg& thp, Int32 value) override;
  void addValue(const TimeHistoryAddValueArg& thp, Int64 value) override;
  void addValue(const TimeHistoryAddValueArg& thp, RealConstArrayView values) override;
  void addValue(const TimeHistoryAddValueArg& thp, Int32ConstArrayView values) override;
  void addValue(const TimeHistoryAddValueArg& thp, Int64ConstArrayView values) override;

 private:
  ITimeHistoryMng* m_thm;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
