// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ITimeHistoryMngInternal.h                                   (C) 2000-2025 */
/*                                                                           */
/* Interface de classe interne gérant un historique de valeurs.              */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#ifndef ARCANE_CORE_INTERNAL_ITIMEHISTORYMNGINTERNAL_H
#define ARCANE_CORE_INTERNAL_ITIMEHISTORYMNGINTERNAL_H

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/UtilsTypes.h"

#include "arcane/core/ITimeHistoryMng.h"
#include "arcane/core/IPropertyMng.h"
#include "arcane/core/Directory.h"
#include "arcane/core/MeshHandle.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ITimeHistoryTransformer;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Classe étendant les arguments lors d'un ajout de valeur
 * dans un historique de valeur.
 */
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

  TimeHistoryAddValueArgInternal(const String& name, bool end_time, Integer subdomain_id)
  : m_thp(name, end_time, subdomain_id)
  , m_mesh_handle()
  {}

 public:

  const TimeHistoryAddValueArg& timeHistoryAddValueArg() const { return m_thp; }
  const MeshHandle& meshHandle() const { return m_mesh_handle; }

 private:

  TimeHistoryAddValueArg m_thp;
  MeshHandle m_mesh_handle;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Interface de la partie interne d'un gestionnaire d'historique de valeur.
 */
class ARCANE_CORE_EXPORT ITimeHistoryMngInternal
{
 public:

  virtual ~ITimeHistoryMngInternal() = default; //!< Libère les ressources

 public:

  /*!
   * \brief Méthode permettant d'ajouter une valeur à un historique.
   *
   * \param thpi Les paramètres de historique.
   * \param value La valeur à ajouter.
   */
  virtual void addValue(const TimeHistoryAddValueArgInternal& thpi, Real value) = 0;

  /*!
   * \brief Méthode permettant d'ajouter une valeur à un historique.
   *
   * \param thpi Les paramètres de historique.
   * \param value La valeur à ajouter.
   */
  virtual void addValue(const TimeHistoryAddValueArgInternal& thpi, Int32 value) = 0;

  /*!
   * \brief Méthode permettant d'ajouter une valeur à un historique.
   *
   * \param thpi Les paramètres de historique.
   * \param value La valeur à ajouter.
   */
  virtual void addValue(const TimeHistoryAddValueArgInternal& thpi, Int64 value) = 0;

  /*!
   * \brief Méthode permettant d'ajouter des valeurs à un historique.
   *
   * \param thpi Les paramètres de historique.
   * \param value Les valeurs à ajouter.
   */
  virtual void addValue(const TimeHistoryAddValueArgInternal& thpi, RealConstArrayView values) = 0;

  /*!
   * \brief Méthode permettant d'ajouter des valeurs à un historique.
   *
   * \param thpi Les paramètres de historique.
   * \param value Les valeurs à ajouter.
   */
  virtual void addValue(const TimeHistoryAddValueArgInternal& thpi, Int32ConstArrayView values) = 0;

  /*!
   * \brief Méthode permettant d'ajouter des valeurs à un historique.
   *
   * \param thpi Les paramètres de historique.
   * \param value Les valeurs à ajouter.
   */
  virtual void addValue(const TimeHistoryAddValueArgInternal& thpi, Int64ConstArrayView values) = 0;

  /*!
   * \brief Méthode permettant d'ajouter le GlobalTime actuel au tableau des GlobalTimes.
   */
  virtual void addNowInGlobalTime() = 0;

  /*!
   * \brief Méthode permettant de copier le tableau de GlobalTime dans la variable globale GlobalTime.
   */
  virtual void updateGlobalTimeCurve() = 0;

  /*!
   * \brief Méthode permettant de redimensionner les tableaux de valeurs après une reprise.
   */
  virtual void resizeArrayAfterRestore() = 0;

  /*!
   * \brief Méthode permettant d"écrire les courbes à l'aide du writer fourni.
   *
   * \param writer Le writer avec lequel les courbes doivent être écrites.
   * \param master_only Si tous les historiques doivent être transférés sur
   *                    le masterIO avant la copie.
   */
  virtual void dumpCurves(ITimeHistoryCurveWriter2* writer) = 0;

  /*!
   * \brief Méthode permettant d'écrire toutes les courbes à l'aide de tous les writers enregistrés.
   */
  virtual void dumpHistory() = 0;

  /*!
   * \brief Méthode permettant de mettre à jour les méta-données des courbes.
   */
  virtual void updateMetaData() = 0;

  /*!
   * \brief Méthode permettant de récupérer les courbes précédemment écrites lors d'une reprise.
   *
   * \param mesh_mng Un pointeur vers un meshMng.
   * \param default_mesh Un pointeur vers le maillage par défaut (nécessaire uniquement pour
   *                     la récupération d'anciens checkpoints).
   */
  virtual void readVariables(IMeshMng* mesh_mng, IMesh* default_mesh) = 0;

  /*!
   * \brief Méthode permettant d'ajouter un écrivain pour la sortie des courbes.
   *
   * \param writer Une ref vers l'écrivain.
   */
  virtual void addCurveWriter(Ref<ITimeHistoryCurveWriter2> writer) = 0;

  /*!
   * \brief Méthode permettant de retirer un écrivain.
   *
   * \param writer Le nom de l'écrivain.
   */
  virtual void removeCurveWriter(const String& name) = 0;

  /*!
   * \brief Applique la transformation \a v à l'ensemble des courbes.
   *
   * \param v La transformation à appliquer.
   */
  virtual void applyTransformation(ITimeHistoryTransformer* v) = 0;

  /*!
   * \brief Retourne un booléen indiquant si l'historique est compressé
   */
  virtual bool isShrinkActive() const = 0;
  /*!
   * \brief Positionne le booléen indiquant si l'historique est compressé
   */
  virtual void setShrinkActive(bool is_active) = 0;

  /*!
   * \brief Indique l'état d'activation.
   *
   * Les fonctions addValue() ne sont prises en compte que si l'instance
   * est active. Dans le cas contraire, les appels à addValue() sont
   * ignorés.
   */
  virtual bool active() const = 0;
  /*!
   * \brief Positionne l'état d'activation.
   * \sa active().
   */
  virtual void setActive(bool is_active) = 0;

  /*!
   * \brief Indique l'état d'activation des sorties.
   *
   * La fonction dumpHistory() est inactives
   * si isDumpActive() est faux.
   */
  virtual bool isDumpActive() const = 0;
  /*!
   * \brief Positionne l'état d'activation des sorties.
   */
  virtual void setDumpActive(bool is_active) = 0;

  /*!
   * \brief Méthode permettant de savoir si notre processus est l'écrivain.
   * \return True si nous sommes l'écrivain.
   */
  virtual bool isMasterIO() = 0;

  /*!
   * \brief Méthode permettant de savoir si notre processus est l'écrivain pour notre sous-domaine.
   * Dans le cas où la réplication est activée, un seul processus parmi les réplicats peut
   * écrire (et uniquement dans le cas où isNonIOMasterCurvesEnabled() == true).
   *
   * La variable d'environnement ARCANE_ENABLE_ALL_REPLICATS_WRITE_CURVES permet de bypasser cette
   * protection et permet à tous les processus d'écrire.
   *
   * \return True si nous sommes l'écrivain pour notre sous-domaine.
   */
  virtual bool isMasterIOOfSubDomain() = 0;

  /*!
   * \brief Méthode permettant de savoir si tous les processus peuvent avoir un historique de valeurs.
   */
  virtual bool isNonIOMasterCurvesEnabled() = 0;

  /*!
   * \brief Méthode permettant de savoir s'il n'y a que le processus maitre qui appelle les écrivains.
   *
   * \return true si oui
   */
  virtual bool isIOMasterWriteOnly() = 0;

  /*!
   * \brief Méthode permettant de définir si seul le processus maitre appelle les écrivains.
   *
   * \param is_active true si oui
   */
  virtual void setIOMasterWriteOnly(bool is_active) = 0;

  /*!
   * \brief Méthode permettant de rajouter les observers sauvegardant l'historique avant une protection.
   *
   * \param prop_mng Un pointeur vers un IPropertyMng.
   */
  virtual void addObservers(IPropertyMng* prop_mng) = 0;

  /*!
   * \brief Méthode permettant de changer le répertoire de sortie des courbes.
   *
   * À noter que le répertoire sera créé s'il n'existe pas.
   *
   * \param directory Le nouveau répertoire de sortie.
   */
  virtual void editOutputPath(const Directory& directory) = 0;

  /*!
   * \brief Méthode permettant de sortir les itérations et les valeurs d'un historique.
   *
   * Méthode utile pour du debug/test. Attention en mode réplication de domaine : il n'y
   * a que les masterRank des sous-domaines qui possèdent les valeurs.
   *
   * \param thpi Les informations nécessaire à la récupération de l'historique.
   * \param iterations [OUT] Les itérations où ont été récupéré chaque valeur.
   * \param values [OUT] Les valeurs récupérées.
   */
  virtual void iterationsAndValues(const TimeHistoryAddValueArgInternal& thpi, UniqueArray<Int32>& iterations, UniqueArray<Real>& values) = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
