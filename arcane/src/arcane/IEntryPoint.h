﻿// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IEntryPoint.h                                               (C) 2000-2012 */
/*                                                                           */
/* Interface du point d'entrée d'un module.                                  */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_IENTRYPOINT_H
#define ARCANE_IENTRYPOINT_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/ArcaneTypes.h"
#include "arcane/Timer.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class IModule;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Interface d'un point d'entrée d'un module.
 * \ingroup Module
 */
class ARCANE_CORE_EXPORT IEntryPoint
{
 public:
  /*! @name Point d'appel
    Endroit ou est utilisé le point d'entrée.
   */
  //@{
  //! appelé pendant la boucle de calcul
  static const char* WComputeLoop;
  //! appelé pour la construction du module
  static const char* WBuild;
  //! appelé pendant l'initialisation
  static const char* WInit;
  //! appelé pendant l'initialisation d'une reprise
  static const char* WContinueInit;
  //! appelé pendant l'initialisation d'un nouveau cas
  static const char* WStartInit;
  //! appelé pour restaurer les variables lors d'un retour arrière
  static const char* WRestore;
  //! appelé après un changement de maillage
  static const char* WOnMeshChanged;
  //! appelé après un raffinement de maillage
  static const char* WOnMeshRefinement;
  //!< appelé lors de la terminaison du code.
  static const char* WExit;
  //@}

  /*!
   * \brief Propriétés d'un point d'entrée.
   */
  enum
  {
    PNone = 0, //!< Pas de propriétés
    /*!
     * \brief Chargé automatiquement au début.
     * Cela signifie qu'un module possédant un point d'entrée avec cette
     * propriété sera toujours chargé, et que le point d'entrée sera ajouté
     * à la liste des points d'entrées s'exécutant en début de boucle en temps.
     */
    PAutoLoadBegin = 1,
    /*!
     * \brief Chargé automatiquement à la fin.
     * Cela signifie qu'un module possédant un point d'entrée avec cette
     * propriété sera toujours chargé, et que le point d'entrée sera ajouté
     * à la liste des points d'entrées s'exécutant en fin de boucle en temps.
     */
    PAutoLoadEnd   = 2
  };

 public:

  virtual ~IEntryPoint() {} //!< Libère les ressources

 public:
	
  //! Retourne le nom du point d'entrée
  virtual const String& name() const =0;

  //! Nom complete du point d'entrée. Ce nom est unique
  virtual const String& fullName() const =0;

 public:

  //! Retourne le gestionnaire principal
  virtual ISubDomain* subDomain() const =0;

  //! Retourne le module associé au point d'entrée
  virtual IModule* module() const =0;

  //! Appelle le point d'entrée
  virtual void executeEntryPoint() =0;

  /*!
   * \brief Temps d'exécution passé dans ce point d'entrée en (en milli-s)
   * \deprecated Utiliser totalCPUTime() à la place.
   */
  ARCANE_DEPRECATED_118 virtual Real totalTime() const =0;

  /*!
   * \brief Temps d'exécution de la dernière itération (en milli-s).
   * \deprecated Utiliser lsatCPUTime() à la place.
   */
  ARCANE_DEPRECATED_118 virtual Real lastTime() const =0;

  /*!
   * \brief Consommation CPU totale passé dans ce point d'entrée en (en milli-s)
   */
  virtual Real totalCPUTime() const =0;

  /*!
   * \brief Consommation CPU de la dernière itération (en milli-s).
   */
  virtual Real lastCPUTime() const =0;

  /*!
   * \brief Temps d'exécution passé (temps horloge) dans ce point d'entrée en (en milli-s)
   */
  virtual Real totalElapsedTime() const =0;

  /*!
   * \brief Temps d'exécution (temps horloge) de la dernière itération (en milli-s).
   */
  virtual Real lastElapsedTime() const =0;

  /*!
   * \brief Temps total associé au timer \a type.
   */
  virtual Real totalTime(Timer::eTimerType type) const =0;

  /*!
   * \brief Temps d'exécution de la dernière itération associé au timer \a type.
   */
  virtual Real lastTime(Timer::eTimerType type) const =0;

  //! Retourne le nombre de fois que le point d'entrée a été exécuté
  virtual Integer nbCall() const =0;

  //! Retourne l'endroit ou est appelé le point d'entrée.
  virtual const String& where() const =0;

  //! Retourne les propriétés du point d'entrée.
  virtual int property() const =0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

