// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ITimeHistoryMngInternal.h                                   (C) 2000-2024 */
/*                                                                           */
/* Interface de classe interne gérant un historique de valeurs.              */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#ifndef ARCANE_ITIMEHISTORYMNGINTERNAL_H
#define ARCANE_ITIMEHISTORYMNGINTERNAL_H

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/UtilsTypes.h"
#include "arcane/utils/FatalErrorException.h"
#include "arcane/core/ITimeHistoryMng.h"
#include "arcane/core/IMesh.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ITimeHistoryTransformer;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ARCANE_CORE_EXPORT TimeHistoryAddValueArgInternal
{
 public:
  explicit TimeHistoryAddValueArgInternal(const TimeHistoryAddValueArg& thp)
  : m_thp(thp)
  , m_mesh_handle()
  {}

  TimeHistoryAddValueArgInternal(const TimeHistoryAddValueArg& thp, const MeshHandle& mesh_handle)
  : m_thp(thp)
  , m_mesh_handle(mesh_handle)
  {}

  TimeHistoryAddValueArgInternal(const String& name, bool end_time, bool is_local)
  : m_thp(name, end_time, is_local)
  , m_mesh_handle()
  {}

 public:
  const TimeHistoryAddValueArg& thp() const { return m_thp; }
  const MeshHandle& meshHandle() const { return m_mesh_handle; }

 private:
  TimeHistoryAddValueArg m_thp;
  MeshHandle m_mesh_handle;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ARCANE_CORE_EXPORT ITimeHistoryMngInternal
{
 public:
  virtual ~ITimeHistoryMngInternal() = default; //!< Libère les ressources

 public:

  // TODO com
  virtual void addValue(const TimeHistoryAddValueArgInternal& thpi, Real value) =0;

  virtual void addValue(const TimeHistoryAddValueArgInternal& thpi, Int32 value) =0;

  virtual void addValue(const TimeHistoryAddValueArgInternal& thpi, Int64 value) =0;

  virtual void addValue(const TimeHistoryAddValueArgInternal& thpi, RealConstArrayView values) =0;

  virtual void addValue(const TimeHistoryAddValueArgInternal& thpi, Int32ConstArrayView values) =0;

  virtual void addValue(const TimeHistoryAddValueArgInternal& thpi, Int64ConstArrayView values) =0;

  /*!
   * \brief Méthode permettant d'ajouter le GlobalTime actuel au tableau des GlobalTimes.
   */
  virtual void addNowInGlobalTime() = 0;

  /*!
   * Méthode permettant de copier le tableau de GlobalTime dans la variable globale GlobalTime.
   */
  virtual void updateGlobalTimeCurve() = 0;

  /*!
   * Méthode permettant de redimensionner les tableaux de valeurs après une reprise.
   */
  virtual void resizeArrayAfterRestore() = 0;

  /*!
   * Méthode permettant d"écrire les courbes à l'aide du writer fourni.
   * @param writer Le writer avec lequel les courbes doivent être écrites.
   */
  virtual void dumpCurves(ITimeHistoryCurveWriter2* writer) =0;

  /*!
   * Méthode permettant d'écrire toutes les courbes à l'aide de tous les writers enregistrés
   * @param is_verbose Active ou non les messages supplémentaires.
   */
  virtual void dumpHistory(bool is_verbose) =0;

  /*!
   * Méthode permettant de mettre à jour les méta-données des courbes.
   */
  virtual void updateMetaData() =0;

  /*!
   * Méthode permettant de récupérer les courbes lors d'une reprise.
   */
  virtual void readVariables() =0;

  /*!
   * Ajoute un écrivain
   */
  virtual void addCurveWriter(Ref<ITimeHistoryCurveWriter2> writer) =0;

  /*!
   * Retire l'écrivain avec le nom name.
   */
  virtual void removeCurveWriter(const String& name) =0;

  /*!
   * \brief Applique la transformation \a v à l'ensemble des courbes.
   */
  virtual void applyTransformation(ITimeHistoryTransformer* v) =0;

  /*!
   * \brief Retourne un booléen indiquant si l'historique est compressé
   */
  virtual bool isShrinkActive() const =0;
  /*!
   * \brief Positionne le booléen indiquant si l'historique est compressé
   */
  virtual void setShrinkActive(bool is_active) =0;

  /*!
   * \brief Indique l'état d'activation.
   *
   * Les fonctions addValue() ne sont prises en compte que si l'instance
   * est active. Dans le cas contraire, les appels à addValue() sont
   * ignorés.
   */
  virtual bool active() const =0;
  /*!
   * \brief Positionne l'état d'activation.
   * \sa active().
   */
  virtual void setActive(bool is_active) =0;

  /*!
   * \brief Indique l'état d'activation des sorties.
   *
   * La fonction dumpHistory() est inactives
   * si isDumpActive() est faux.
   */
  virtual bool isDumpActive() const =0;
  /*!
   * \brief Positionne l'état d'activation des sorties.
   */
  virtual void setDumpActive(bool is_active) =0;

  /*!
   * Méthode permettant de savoir si notre processus est l'écrivain.
   * @return True si nous sommes l'écrivain.
   */
  virtual bool isMasterIO() = 0;

  /*!
   * Méthode permettant de savoir si tous les processus peuvent avoir un historique de valeurs.
   */
  virtual bool isNonIOMasterCurvesEnabled() = 0;

  virtual void addObservers() = 0;

};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

