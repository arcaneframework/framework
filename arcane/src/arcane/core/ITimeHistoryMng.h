// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ITimeHistoryMng.h                                           (C) 2000-2025 */
/*                                                                           */
/* Interface de la classe gérant un historique de valeurs.                   */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_ITIMEHISTORYMNG_H
#define ARCANE_CORE_ITIMEHISTORYMNG_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/ArcaneTypes.h"
#include "arcane/utils/UtilsTypes.h"
#include "arcane/utils/FatalErrorException.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * Classe contenant les arguments pour les méthodes utilisateurs 'addValue'.
 */
class TimeHistoryAddValueArg
{
 public:

  /*!
   * \brief Constructeur avec trois paramètres.
   *
   * \param name Le nom de la courbe.
   * \param end_time Doit-on écrire la valeur à notre itération ou à notre itération-1 ?
   * \param subdomain_id L'id du sous-domaine qui doit sauver la valeur (-1 pour global).
   */
  TimeHistoryAddValueArg(const String& name, bool end_time, Integer subdomain_id)
  : m_name(name)
  , m_end_time(end_time)
  , m_subdomain_id(subdomain_id)
  {}

  /*!
   * \brief Constructeur avec deux paramètres.
   *
   * La valeur sera sauvegardé globalement, et non sur un sous-domaine en particulier.
   *
   * \param name Le nom de la courbe.
   * \param end_time Doit-on écrire la valeur à notre itération ou à notre itération-1 ?
   */
  TimeHistoryAddValueArg(const String& name, bool end_time)
  : TimeHistoryAddValueArg(name, end_time, NULL_SUB_DOMAIN_ID)
  {}

  /*!
   * \brief Constructeur avec un paramètre.
   *
   * La valeur sera sauvegardé globalement, et non sur un sous-domaine en particulier.
   * La valeur sera sauvegardé à notre itération.
   *
   * \param name Le nom de la courbe.
   */
  explicit TimeHistoryAddValueArg(const String& name)
  : TimeHistoryAddValueArg(name, true)
  {}

 public:

  const String& name() const { return m_name; }
  bool endTime() const { return m_end_time; }
  bool isLocal() const { return m_subdomain_id != NULL_SUB_DOMAIN_ID; }
  Integer localSubDomainId() const { return m_subdomain_id; }

 private:

  String m_name;
  bool m_end_time;
  Integer m_subdomain_id;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ITimeHistoryCurveWriter;
class ITimeHistoryCurveWriter2;
class ITimeHistoryTransformer;
class ITimeHistoryMngInternal;
class ITimeHistoryAdder;

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
class ITimeHistoryMng
{
 public:

  virtual ~ITimeHistoryMng() = default; //!< Libère les ressources

 public:

  // TODO Deprecated
  /*! \brief Ajoute la valeur \a value à l'historique \a name.
   *
   * \deprecated Cette méthode est dépréciée et est remplacée par l'utilisation de l'objet GlobalTimeHistoryAdder.
   *
   * La valeur est celle au temps de fin de l'itération si \a end_time est vrai,
   * au début sinon.
   * le booleen is_local indique si la courbe est propre au process ou pas pour pouvoir écrire des courbes meme
   * par des procs non io_master quand la varariable ARCANE_ENABLE_NON_IO_MASTER_CURVES
   */
  virtual void addValue(const String& name, Real value, bool end_time = true, bool is_local = false) = 0;
  /*! \brief Ajoute la valeur \a value à l'historique \a name.
   *
   * \deprecated Cette méthode est dépréciée et est remplacée par l'utilisation de l'objet GlobalTimeHistoryAdder.
   *
   * La valeur est celle au temps de fin de l'itération si \a end_time est vrai,
   * au début sinon.
   * le booleen is_local indique si la courbe est propre au process ou pas pour pouvoir écrire des courbes meme
   * par des procs non io_master quand la varariable ARCANE_ENABLE_NON_IO_MASTER_CURVES
   */
  virtual void addValue(const String& name, Int32 value, bool end_time = true, bool is_local = false) = 0;
  /*! Ajoute la valeur \a value à l'historique \a name.
   *
   * \deprecated Cette méthode est dépréciée et est remplacée par l'utilisation de l'objet GlobalTimeHistoryAdder.
   *
   * La valeur est celle au temps de fin de l'itération si \a end_time est vrai,
   * au début sinon.
   * le booleen is_local indique si la courbe est propre au process ou pas pour pouvoir écrire des courbes meme
   * par des procs non io_master quand la varariable ARCANE_ENABLE_NON_IO_MASTER_CURVES
   */
  virtual void addValue(const String& name, Int64 value, bool end_time = true, bool is_local = false) = 0;
  /*! \brief Ajoute la valeur \a value à l'historique \a name.
   *
   * \deprecated Cette méthode est dépréciée et est remplacée par l'utilisation de l'objet GlobalTimeHistoryAdder.
   *
   * Le nombre d'éléments de \a value doit être constant au cours du temps.
   * La valeur est celle au temps de fin de l'itération si \a end_time est vrai,
   * au début sinon.
   * le booleen is_local indique si la courbe est propre au process ou pas pour pouvoir écrire des courbes meme
   * par des procs non io_master quand la varariable ARCANE_ENABLE_NON_IO_MASTER_CURVES
   */
  virtual void addValue(const String& name, RealConstArrayView value, bool end_time = true, bool is_local = false) = 0;
  /*! \brief Ajoute la valeur \a value à l'historique \a name.
   *
   * \deprecated Cette méthode est dépréciée et est remplacée par l'utilisation de l'objet GlobalTimeHistoryAdder.
   *
   * Le nombre d'éléments de \a value doit être constant au cours du temps.
   * La valeur est celle au temps de fin de l'itération si \a end_time est vrai,
   * au début sinon.
   * le booleen is_local indique si la courbe est propre au process ou pas pour pouvoir écrire des courbes meme
   * par des procs non io_master quand la varariable ARCANE_ENABLE_NON_IO_MASTER_CURVES
   */
  virtual void addValue(const String& name, Int32ConstArrayView value, bool end_time = true, bool is_local = false) = 0;
  /*! Ajoute la valeur \a value à l'historique \a name.
   *
   * \deprecated Cette méthode est dépréciée et est remplacée par l'utilisation de l'objet GlobalTimeHistoryAdder.
   *
   * Le nombre d'éléments de \a value doit être constant au cours du temps.
   * La valeur est celle au temps de fin de l'itération si \a end_time est vrai,
   * au début sinon.
   * le booleen is_local indique si la courbe est propre au process ou pas pour pouvoir écrire des courbes meme
   * par des procs non io_master quand la varariable ARCANE_ENABLE_NON_IO_MASTER_CURVES
   */
  virtual void addValue(const String& name, Int64ConstArrayView value, bool end_time = true, bool is_local = false) = 0;

 public:

  virtual void timeHistoryBegin() = 0;
  virtual void timeHistoryEnd() = 0;
  virtual void timeHistoryInit() = 0;
  virtual void timeHistoryStartInit() = 0;
  virtual void timeHistoryContinueInit() = 0;
  virtual void timeHistoryRestore() = 0;

 public:

  //! Ajoute un écrivain
  virtual ARCANE_DEPRECATED void addCurveWriter(ITimeHistoryCurveWriter* writer)
  {
    ARCANE_UNUSED(writer);
    ARCANE_FATAL("No longer supported. Use 'ITimeHistoryCurveWriter2' interface");
  }

  //! Supprime un écrivain
  virtual ARCANE_DEPRECATED void removeCurveWriter(ITimeHistoryCurveWriter* writer)
  {
    ARCANE_UNUSED(writer);
    ARCANE_FATAL("No longer supported. Use 'ITimeHistoryCurveWriter2' interface");
  }

  //! Ajoute un écrivain
  virtual void addCurveWriter(ITimeHistoryCurveWriter2* writer) = 0;

  //! Supprime un écrivain
  virtual void removeCurveWriter(ITimeHistoryCurveWriter2* writer) = 0;

  //! Supprime l'écrivain de nom \a name
  virtual void removeCurveWriter(const String& name) = 0;

 public:

  /*!
   * \internal
   * \brief Sauve l'historique.
   *
   * Cela consiste à appelé dumpCurves() pour chaque écrivain enregistré.
   */
  virtual void dumpHistory(bool is_verbose) = 0;

  /*!
   * \brief Utilise l'écrivain \a writer pour sortir toutes les courbes.
   *
   * Le chemin de sortie est le répertoire courant.
   */
  virtual void dumpCurves(ITimeHistoryCurveWriter2* writer) = 0;

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
   * \brief Applique la transformation \a v à l'ensemble des courbes.
   */
  virtual void applyTransformation(ITimeHistoryTransformer* v) = 0;

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
   * \brief Retourne un booléen indiquant si l'historique est compressé
   */
  virtual bool isShrinkActive() const = 0;

  /*!
   * \brief Positionne le booléen indiquant si l'historique est compressé
   */
  virtual void setShrinkActive(bool is_active) = 0;

 public:

  //! API interne à Arcane
  virtual ITimeHistoryMngInternal* _internalApi() { ARCANE_FATAL("Invalid usage"); };
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
