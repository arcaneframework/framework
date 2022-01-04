// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IModuleFactory.h                                            (C) 2000-2019 */
/*                                                                           */
/* Interface de la manufacture des modules.                                  */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_IMODULEFACTORY_H
#define ARCANE_IMODULEFACTORY_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/base/ReferenceCounter.h"
#include "arcane/utils/ArcaneGlobal.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{
class MeshHandle;
class IModuleFactory2;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Informations sur la fabrique d'un module.
 *
 * Cette interface contient les informations nécessaire sur une fabrique
 * d'un module.
 *
 * On peut directement créer le module via la méthode createModule().
 *
 * Cette classe utilise un compteur de référence pour gérer sa durée de vie
 * (voir la classe ReferenceCounter).
 */
class ARCANE_CORE_EXPORT IModuleFactoryInfo
{
 protected:

  //! Libère les ressources
  virtual ~IModuleFactoryInfo() {}

 public:

  virtual void addReference() =0;
  virtual void removeReference() =0;
  /*!
   * \brief Indique si le module et doit être chargé automatiquement.
   *
   * Si cette propriété est vrai, le module sera toujours chargé même
   * s'il n'apparait pas dans la boucle en temps.
   */
  virtual bool isAutoload() const =0;

  /*!
   * \brief Si la fabrique est un pour un module,
   * l'initialise sur le sous-domaine \a sub_domain.
   *
   * Cette méthode est appelée lorsque le sous-domaine est créé, pour
   * effectuer certaines initialisations spécifiques du module avant
   * que celui-ci ne soit fabriqué. Par exemple, pour ajouter des boucles
   * en temps propres au module.
   */
  virtual void initializeModuleFactory(ISubDomain* sub_domain) =0;

  /*!
   * \brief Créé un module.
   *
   * L'implémentation doit appeler parent->moduleMng()->addModule()
   * pour le module créé.
   *
   * \param parent Parent de ce module. 
   * \param mesh maillage associé au module.
   * \return le module créé
   */
  virtual Ref<IModule> createModule(ISubDomain* parent,const MeshHandle& mesh_handle) =0;

  //! Nom du module créé par cette fabrique.
  virtual String moduleName() const =0;

  /*!
   * \brief Informations sur le module pouvant être créé par cette fabrique.
   *
   * L'instance retournée reste la propriété de l'application l'ayant créée
   * et ne doit ni être modifiée, ni être détruite.
   */
  virtual const IServiceInfo* serviceInfo() const =0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * \brief Interface d'une fabrique de module (V2).
 *
 * Cette interface est réservée à IModuleFactoryInfo et ne doit pas
 * être utilisée directement.
 */
class ARCANE_CORE_EXPORT IModuleFactory2
{
 public:
  virtual ~IModuleFactory2(){}
 public:
  /*!
   * \brief Créé un module.
   *
   * \param sd sous-domaine associé.
   * \param mesh maillage associé au module.
   * \return le module créé
   */
  virtual Ref<IModule> createModuleInstance(ISubDomain* sd,const MeshHandle& mesh_handle) =0;

  /*!
   * \brief Initialisation statique du module.
   *
   * Cette méthode est appelée lorsque le sous-domaine est créé, pour
   * effectuer certaines initialisations spécifiques du module avant
   * que celui-ci ne soit fabriqué. Par exemple, pour ajouter des boucles
   * en temps propres au module.
   */
  virtual void initializeModuleFactory(ISubDomain* sd) =0;

  //! Nom du module créé par cette fabrique.
  virtual String moduleName() const =0;

  //! Informations sur le module pouvant être créé par cette fabrique.
  virtual const IServiceInfo* serviceInfo() const =0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Compteur de référence sur une fabrique de module.
 */
class ARCANE_CORE_EXPORT ModuleFactoryReference
: ReferenceCounter<IModuleFactoryInfo>
{
 public:
  typedef ReferenceCounter<IModuleFactoryInfo> Base;
 public:
  explicit ModuleFactoryReference(IModuleFactoryInfo* f)
  : Base(f){}
  ModuleFactoryReference(Ref<IModuleFactory2> factory,bool is_autoload);
 public:
  IModuleFactoryInfo* factory() const { return get(); }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  
