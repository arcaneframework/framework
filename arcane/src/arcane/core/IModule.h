// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IModule.h                                                   (C) 2000-2025 */
/*                                                                           */
/* Interface de la classe Module.                                            */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_IMODULE_H
#define ARCANE_CORE_IMODULE_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/ArcaneTypes.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class IVariableRef;
class IParallelMng;
class CaseOptionsMain;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Interface d'un module.
 * \ingroup Module
 */
class ARCANE_CORE_EXPORT IModule
{
 public:

  //! Destructeur
  virtual ~IModule() {}

 public:

  //! Nom du module
  virtual String name() const =0;

  //! Version du module
  virtual VersionInfo versionInfo() const =0;

 public:
	
  //! Session du module
  virtual ISession* session() const =0;

  //! Gestionnaire de sous-domaine.
  virtual ISubDomain* subDomain() const =0;

  //! Maillage associé. Peut être nul. Utiliser defaultMeshHandle() à la place
  virtual IMesh* defaultMesh() const =0;

  //! Maillage associé
  virtual MeshHandle defaultMeshHandle() const =0;

  //! Gestionnaire du parallélisme par échange de message
  virtual IParallelMng* parallelMng() const =0;

 //! Gestionnaire des accélérateurs
  virtual IAcceleratorMng* acceleratorMng() const =0;

  //! Gestionnaire de traces.
  virtual ITraceMng* traceMng() const =0;

 public:
  
  /*!
   * \brief Indique si un module est utilisé ou non (interne).
   *
   * Un module est utilisé si et seulement si au moins un de ses
   * points d'entrée est utilisé dans la boucle en temps.
   */
  virtual void setUsed(bool v) =0;

  //! \a true si le module est utilisé.
  virtual bool used() const =0;

  /*!
   * \brief Active ou désactive temporairement le module (interne).
   *
   * Lorsqu'un module est désactivé, ses points d'entrée de la boucle
   * de calcul ne sont plus appelés (mais les autres comme ceux
   * d'initialisation ou de terminaison le sont toujours).
   */
  virtual void setDisabled(bool v) =0;

  //! \a true si le module est désactivé
  virtual bool disabled() const =0;

  /*! \internal
   * \brief Indique si le module est géré par un ramasse miette auquel
   * cas il ne faut pas appeler l'operateur delete dessus.
   */
  virtual bool isGarbageCollected() const =0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

