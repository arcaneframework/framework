// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MeshTimeHistoryAdder.h                                      (C) 2000-2024 */
/*                                                                           */
/* Classe permettant d'ajouter un historique de valeur lié à un maillage.    */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_MESHTIMEHISTORYADDER_H
#define ARCANE_CORE_MESHTIMEHISTORYADDER_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/ITimeHistoryAdder.h"
#include "arcane/core/IParallelMng.h"
#include "arcane/core/MeshHandle.h"

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
 * Cette classe enregistrera les courbes avec comme support un maillage.
 * C'est-à-dire que les courbes seront liées à un maillage, en plus d'être
 * liées au domaine complet ou au sous-domaine demandé.
 * Si le lien au maillage n'est pas désiré, la classe GlobalTimeHistoryAdder
 * peut être plus intéressante.
 *
 * Pour un nom d'historique donné, il peut y avoir qu'une courbe de une
 * ou plusieurs valeurs par maillage et par sous-domaine (et une globale à
 * tous les sous-domaines).
 *
 * Exemple : plusieurs courbes de moyennes des pressions (appelons-les
 * "avg_pressure"), deux sous-domaines (0 et 1) et deux maillages (mesh0 et mesh1).
 * Une valeur par itération.
 * - Une courbe "avg_pressure" liée au sous-domaine 0 et au maillage 0. Chaque
 *   valeur est la moyenne des pressions de chaque maille du maillage 0 et
 *   du sous-domaine 0.
 * - Une courbe "avg_pressure" liée au sous-domaine 0 et au maillage 1. Chaque
 *   valeur est la moyenne des pressions de chaque maille du maillage 1 et
 *   du sous-domaine 0.
 * - Une courbe "avg_pressure" liée au sous-domaine 1 et au maillage 0. Chaque
 *   valeur est la moyenne des pressions de chaque maille du maillage 0 et
 *   du sous-domaine 1.
 * - Une courbe "avg_pressure" liée au sous-domaine 1 et au maillage 1. Chaque
 *   valeur est la moyenne des pressions de chaque maille du maillage 1 et
 *   du sous-domaine 1.
 * - Une courbe "avg_pressure" liée au domaine complet et au maillage 0.
 *   Chaque valeur est la moyenne des pressions du maillage 0 de chaque
 *   sous-domaine.
 * - Une courbe "avg_pressure" liée au domaine complet et au maillage 1.
 *   Chaque valeur est la moyenne des pressions du maillage 1 de chaque
 *   sous-domaine.
 *
 * On peut remarquer qu'il est possible d'avoir plusieurs courbes
 * indépendantes avec le même nom mais liée à des maillages différents et à
 * des sous-domaines différents (+1 courbe globale). Et il est important de
 * souligner que ce même nom peut être aussi utilisé avec les courbes de
 * GlobalTimeHistoryAdder indépendement, donc l'exemple ci-dessus peut être
 * complémentaire avec celui donné dans la description de GlobalTimeHistoryAdder !
 * (donc possiblement 9 courbes indépendantes mais de même nom !)
 */
class ARCANE_CORE_EXPORT MeshTimeHistoryAdder
: public ITimeHistoryAdder
{
 public:
  /*!
   * \brief Constructeur.
   *
   * \param time_history_mng Un pointeur vers une instance de ITimeHistoryMng.
   * \param mesh_handle Le maillage à lier aux courbes.
   */
  MeshTimeHistoryAdder(ITimeHistoryMng* time_history_mng, const MeshHandle& mesh_handle);
  ~MeshTimeHistoryAdder() override = default;

 public:
  void addValue(const TimeHistoryAddValueArg& thp, Real value) override;
  void addValue(const TimeHistoryAddValueArg& thp, Int32 value) override;
  void addValue(const TimeHistoryAddValueArg& thp, Int64 value) override;
  void addValue(const TimeHistoryAddValueArg& thp, RealConstArrayView values) override;
  void addValue(const TimeHistoryAddValueArg& thp, Int32ConstArrayView values) override;
  void addValue(const TimeHistoryAddValueArg& thp, Int64ConstArrayView values) override;

 private:
  ITimeHistoryMng* m_thm;
  MeshHandle m_mesh_handle;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
