// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2023 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IMainFactory.h                                              (C) 2000-2023 */
/*                                                                           */
/* Interface des AbstractFactory d'Arcane.                                   */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_IMAINFACTORY_H
#define ARCANE_IMAINFACTORY_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/ArcaneTypes.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class IBase;
class ISubDomain;
class ApplicationInfo;
class IArcaneMain;
class IParallelSuperMng;
class IApplication;
class IRegistry;
class IVariableMng;
class IModuleMng;
class IEntryPointMng;
class ITimeHistoryMng;
class ICaseMng;
class ICaseDocument;
class ITimerMng;
class ITimeLoopMng;
class ITimeLoop;
class IIOMng;
class IServiceMng;
class IServiceLoader;
class IXmlDocumentHolder;
class IMesh;
class IDataFactory;
class ITimeStats;
class IParallelMng;
class ItemGroup;
class IPrimaryMesh;
class ITraceMngPolicy;
class IModuleMaster;
class ILoadBalanceMng;
class ICheckpointMng;
class IPropertyMng;
class IDataFactoryMng;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * \brief Manufacture des classes d'Arcane.
 *
 Il s'agit d'une classe virtuelle comprenant les méthodes pour fabriquer
 les différentes instances des gestionnaires de l'architecture
 (Design Pattern: AbstractFactory).

 Arcane fournit des fabriques par défaut pour la plupart des gestionnaires
 (IApplication, IParallelSuperMng, ...). La classe gérant le code doit par contre
 être spécifiée en implémentant la méthode createArcaneMain() dans une
 classe dérivée.

 Le point d'entrée général du code se fait par l'appel à la fonction
 arcaneMain().

 Par exemple, si on définit une classe <tt>ConcreteMainFactory</tt> qui
 dérive de IMainFactory, on lance le code comme suit:
 
 * \code
 * int
 * main(int argc,char** argv)
 * {
 *   ApplicationInfo exe_info = ... // Création des infos de l'exécutable.
 *   ConcreteMainFactory cmf; // Création de la manufacture
 *   return IMainFactory::arcaneMain(exe_info,&cmf);
 * }
 * \endcode
 */
class IMainFactory
{
 public:

  virtual ~IMainFactory() {} //!< Libère les ressources.

 public:

 public:

  //! Crée une instance de IArcaneMain
  virtual IArcaneMain* createArcaneMain(const ApplicationInfo& app_info) =0;

 public:

  //! Crée une instance d'un superviseur
  virtual IApplication* createApplication(IArcaneMain*) =0;

  //! Crée une instance du gestionnaire de variable
  virtual IVariableMng* createVariableMng(ISubDomain*) =0;

  //! Crée une instance du gestionnaire de module
  virtual IModuleMng* createModuleMng(ISubDomain*) =0;

  //! Crée une instance du gestionnaire des points d'entrée
  virtual IEntryPointMng* createEntryPointMng(ISubDomain*) =0;

  //! Crée une instance du gestionnaire d'historique en temps
  virtual ITimeHistoryMng* createTimeHistoryMng(ISubDomain*) =0;

  //! Crée une instance du gestionnaire du jeu de données
  virtual ICaseMng* createCaseMng(ISubDomain*) =0;

  //! Crée une instance d'un document du jeu de données
  virtual ICaseDocument* createCaseDocument(IApplication*) =0;

  //! Crée une instance d'un document du jeu de données pour une langue donnée \a lang
  virtual ICaseDocument* createCaseDocument(IApplication*,const String& lang) =0;

  //! Crée une instance d'un document du jeu de données
  virtual ICaseDocument* createCaseDocument(IApplication*,IXmlDocumentHolder* doc) =0;

  /*!
   * \brief Crée une instance des statistiques de temps d'exécution.
   *
   * Utiliser la surchage createTimeStats(ITimerMng*,ITraceMng*,const String& name).
   */
  virtual ARCANE_DEPRECATED_116 ITimeStats* createTimeStats(ISubDomain*) =0;

  //! Crée une instance des statistiques de temps d'exécution
  virtual ITimeStats* createTimeStats(ITimerMng* tim,ITraceMng* trm,const String& name) =0;

  //! Crée une instance du gestionnaire de la boucle en temps
  virtual ITimeLoopMng* createTimeLoopMng(ISubDomain*) =0;

  //! Crée une boucle en temps de nom \a name
  virtual ITimeLoop* createTimeLoop(IApplication* sm,const String& name) =0;

  //! Crée une instance du gestionnaire d'entrée/sortie
  virtual IIOMng* createIOMng(IApplication*) =0;

  //! Crée une instance du gestionnaire d'entrée/sortie pour le gestionnaire de parallélisme \a pm
  virtual IIOMng* createIOMng(IParallelMng* pm) =0;

  //! Crée une instance du chargeur de services
  virtual IServiceLoader* createServiceLoader() =0;

  //! Crée une instance du gestionnaire de services
  virtual IServiceMng* createServiceMng(IBase*) =0;

  //! Crée une instance du gestionnaire de protections
  virtual ICheckpointMng* createCheckpointMng(ISubDomain*) =0;

  //! Crée une instance du gestionnaire de propriétés
  ARCCORE_DEPRECATED_2020("Use createPropertyMngReference() instead")
  virtual IPropertyMng* createPropertyMng(ISubDomain*) =0;

  //! Crée une instance du gestionnaire de propriétés
  virtual Ref<IPropertyMng> createPropertyMngReference(ISubDomain*) =0;

  /*!
   * \brief Créé ou récupère un maillage.
   *
   * Créé ou récupère un maillage de nom \a name pour le sous-domaine \a sub_domain.
   *
   * Si le sous-domaine possède déjà un maillage avec le nom \a name,
   * c'est ce dernier qui est retourné.
   */
  virtual IPrimaryMesh* createMesh(ISubDomain* sub_domain,const String& name) =0;

  /*!
   * \brief Créé ou récupère un maillage.
   *
   * Créé ou récupère un maillage de nom \a name pour le sous-domaine \a sub_domain.
   *
   * Si le sous-domaine possède déjà un maillage avec le nom \a name,
   * c'est ce dernier qui est retourné.
   */
  ARCANE_DEPRECATED_REASON("Y2023: Use createMesh(..., eMeshAMRKind amr_type) instead")
  virtual IPrimaryMesh* createMesh(ISubDomain* sub_domain,const String& name,bool is_amr) =0;

  /*!
   * \brief Créé ou récupère un maillage.
   *
   * Créé ou récupère un maillage de nom \a name pour le sous-domaine \a sub_domain.
   *
   * Si le sous-domaine possède déjà un maillage avec le nom \a name,
   * c'est ce dernier qui est retourné.
   */
  virtual IPrimaryMesh* createMesh(ISubDomain* sub_domain,const String& name,eMeshAMRKind amr_type) =0;

  /*!
   * \brief Créé ou récupère un maillage.
   *
   * Créé ou récupère un maillage de nom \a name pour le sous-domaine \a sub_domain
   * associé au gestionnaire de parallélisme \a pm. Si le sous-domaine possède
   * déjà un maillage avec le nom \a name, c'est ce dernier qui est retourné.
   *
   * Le gestionnaire de parallélisme doit être le même que celui du sous-domaine
   * ou issu de celui-ci.
   */
  virtual IPrimaryMesh* createMesh(ISubDomain* sub_domain, IParallelMng* pm,
                                   const String& name) =0;

  /*!
   * \brief Créé ou récupère un maillage.
   *
   * Créé ou récupère un maillage de nom \a name pour le sous-domaine \a sub_domain
   * associé au gestionnaire de parallélisme \a pm. Si le sous-domaine possède
   * déjà un maillage avec le nom \a name, c'est ce dernier qui est retourné.
   *
   * Le gestionnaire de parallélisme doit être le même que celui du sous-domaine
   * ou issu de celui-ci.
   */
  ARCANE_DEPRECATED_REASON("Y2023: Use createMesh(..., eMeshAMRKind amr_type) instead")
  virtual IPrimaryMesh* createMesh(ISubDomain* sub_domain, IParallelMng* pm,
                                   const String& name, bool is_amr) =0;

  /*!
   * \brief Créé ou récupère un maillage.
   *
   * Créé ou récupère un maillage de nom \a name pour le sous-domaine \a sub_domain
   * associé au gestionnaire de parallélisme \a pm. Si le sous-domaine possède
   * déjà un maillage avec le nom \a name, c'est ce dernier qui est retourné.
   *
   * Le gestionnaire de parallélisme doit être le même que celui du sous-domaine
   * ou issu de celui-ci.
   */
  virtual IPrimaryMesh* createMesh(ISubDomain* sub_domain, IParallelMng* pm,
                                   const String& name, eMeshAMRKind amr_type) =0;

  /*!
   * \brief Créé un sous-maillage pour le maillage \a mesh, de nom \a name.
   *
   * Le sous-maillage est initialisé avec les items du groupe \a group.
   * Actuellement, ce groupe ne peut ni être un groupe complet (isAllItems())
   * ni un groupe calculé (si non incrémental).
   */
  virtual IMesh* createSubMesh(IMesh* mesh, const ItemGroup& group, const String& name) =0;

  //! Créé une fabrique pour les données
  ARCCORE_DEPRECATED_2020("Use createDataFactoryMngRef() instead")
  virtual IDataFactory* createDataFactory(IApplication*) =0;

  //! Créé un gestionnaire de fabrique pour les données
  virtual Ref<IDataFactoryMng> createDataFactoryMngRef(IApplication*) =0;

  //! Créé un gestionnaire pour les accélérateurs
  virtual Ref<IAcceleratorMng> createAcceleratorMngRef(ITraceMng* tm) =0;

  /*!
   * \brief Créé un gestionnaire de trace.
   *
   * L'instance retournée doit être initialisée via un ITraceMngPolicy.
   */
  virtual ITraceMng* createTraceMng() =0;

  /*!
   * \brief Créé un gestionnaire de configuration pour un gestion de trace.
   */
  virtual ITraceMngPolicy* createTraceMngPolicy(IApplication* app) =0;

  /*!
   * \brief Créé le module maitre pour le sous-domaine \a sd.
   */
  virtual IModuleMaster* createModuleMaster(ISubDomain* sd) =0;

  /*!
   * \brief Cree un gestionnaire de description pour l'equilibrage.
   */
  virtual ILoadBalanceMng* createLoadBalanceMng(ISubDomain* sd) =0;

 private:
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif

