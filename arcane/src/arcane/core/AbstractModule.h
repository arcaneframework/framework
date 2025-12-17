// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* AbstractModule.h                                            (C) 2000-2025 */
/*                                                                           */
/* Classe abstraite de base d'un module.                                     */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_ABSTRACTMODULE_H
#define ARCANE_CORE_ABSTRACTMODULE_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/String.h"
#include "arcane/utils/VersionInfo.h"
#include "arcane/utils/TraceAccessor.h"

#include "arcane/core/ArcaneTypes.h"
#include "arcane/core/IModule.h"
#include "arcane/core/ModuleBuildInfo.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ModuleBuildInfo;
typedef ModuleBuildInfo ModuleBuilder;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Classe représentant un module.
 *
 * Cette classe est LA classe d'implémentation bas niveau de l'interface \a IModule.
 *
 * \ingroup Module
 */
class ARCANE_CORE_EXPORT AbstractModule
: public TraceAccessor
, public IModule
{
 public:

  //! Constructeur à partir d'un \a ModuleBuildInfo
  AbstractModule(const ModuleBuildInfo&);

 public:
	
  //! Destructeur
  virtual ~AbstractModule();
  
 public:

  //! Version du module
  VersionInfo versionInfo() const override { return m_version_info; }

 public:

  /*! \brief Initialisation du module pour le sous-domaine \a sd.
   *
   * Cette méthode statique peut être redéfinie dans une classe dérivée
   * pour effectuer des initialisations pour le sous-domaine \a sd
   * même si le module n'est pas utilisé.
   *
   * Une utilisation classique est l'enregistrement de points d'entrée
   * pour des modules sans .axl
   *
   * Cette méthode sera appelé pendant la phase de création du
   * sous-domaine sur tous les Modules (même non utilisés).
   */
  static void staticInitialize(ISubDomain* sd) { ARCANE_UNUSED(sd); }

 public:

  //! Nom du module
  String name() const override { return m_name; }
  //! Session associé au module
  ISession* session() const override { return m_session; }
  //! Sous-domaine associé au module
  ISubDomain* subDomain() const override { return m_sub_domain; }
  //! Maillage par défaut pour ce module
  IMesh* defaultMesh() const override { return m_default_mesh_handle.mesh(); }
  //! Maillage par défaut pour ce module
  MeshHandle defaultMeshHandle() const override { return m_default_mesh_handle; }
  //! Gestionnaire du parallélisme par échange de message
  IParallelMng* parallelMng() const override;
  //! Gestionnaire des accélérateurs.
  IAcceleratorMng* acceleratorMng() const override;
  //! Gestionnaire de traces
  ITraceMng* traceMng() const override;
  //! Positionne le flag d'utilisation du module
  void setUsed(bool v) override { m_used = v; }
  //! Retourne l'état d'utilisation du module
  bool used() const override { return m_used; }
  //! Positionne le flag d'activation du module
  void setDisabled(bool v) override { m_disabled = v; }
  //! Retourne l'état d'activation du module
  bool disabled() const override { return m_disabled; }
  //! Indique si le module utilise un système de Garbage collection
  /*! 
   *  <ul>
   *  <li>si \a true, indique une destruction par un Garbage collecteur et non une destruction explicite</li>
   *  <li>si \a false, ce module sera détruit explicitement par un appel à son destructeur</li>
   *  </ul>
   *
   * Le système de Garbage collection est usuellement activé pour les
   * modules issus d'une implémentation en C#. Les modules classiques
   * en C++ n'ont pas se mécanisme.
   *
   * \todo Vérifier dans ModuleMng::removeModule l'utilisation de
   * cette indication. Un appel au Deleter comme dans
   * ModuleMng::removeAllModules est peut-être nécessaire.
   */
  bool isGarbageCollected() const override { return false; }

 protected:

  void _setVersionInfo(const VersionInfo& vi)
  {
    m_version_info = vi;
  }

 private:

  ISession* m_session; //!< Sesion
  ISubDomain* m_sub_domain; //!< sous-domaine
  MeshHandle m_default_mesh_handle; //!< Maillage par défaut du module
  String m_name; //!< Nom du module
  bool m_used; //!< \a true si le module est utilisé
  bool m_disabled; //!< Etat d'activation du module
  VersionInfo m_version_info; //!< Version du module
  IAcceleratorMng* m_accelerator_mng; //!< Gestionnaire des accélérateurs
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

