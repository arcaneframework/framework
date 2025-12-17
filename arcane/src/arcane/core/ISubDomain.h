// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ISubDomain.h                                                (C) 2000-2025 */
/*                                                                           */
/* Interface d'un sous-domaine.                                              */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_ISUBDOMAIN_H
#define ARCANE_CORE_ISUBDOMAIN_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/ArcaneTypes.h"

#include "arcane/core/IBase.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class IVariableMng;
class IModuleMng;
class IServiceMng;
class IEntryPointMng;
class IModule;
class IMeshIOService;
class IMesh;
class IMeshMng;
class ApplicationInfo;
class IIOMng;
class ITimeLoopMng;
class CaseOptionsMain;
class IParallelMng;
class IThreadMng;
class IDirectory;
class ITimeHistoryMng;
class ICaseMng;
class IInterfaceMng;
class ITimerMng;
class ITimeStats;
class IRessourceMng;
class CommonVariables;
class IMainFactory;
class ICaseDocument;
class XmlNode;
class IMemoryInfo;
class IObservable;
class IInitialPartitioner;
class IDirectExecution;
class IPhysicalUnitSystem;
class ILoadBalanceMng;
class IModuleMaster;
class ICheckpointMng;
class IPropertyMng;
class IConfiguration;
class MeshHandle;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Interface du gestionnaire d'un sous-domaine.
 */
class ARCANE_CORE_EXPORT ISubDomain
: public IBase
{
 protected:

  virtual ~ISubDomain() {} //!< Libère les ressources.

 public:

  virtual void destroy() =0;

 public:

  //! Manufacture principale.
  virtual IMainFactory* mainFactory() =0;

  //! Session
  virtual ISession* session() const =0;

  //! Application
  virtual IApplication* application() =0;

  //! Retourne le gestionnaire de variables
  virtual IVariableMng* variableMng() =0;

  //! Retourne le gestionnaire de modules
  virtual IModuleMng* moduleMng() =0;

  //! Retourne le gestionnaire de points d'entrée
  virtual IEntryPointMng* entryPointMng() =0;

  //! Retourne le gestionnaire de parallélisme
  virtual IParallelMng* parallelMng() =0;

  /*!
   * \brief Retourne le gestionnaire de parallélisme pour tous les réplicats.
   *
   * En règle général, il faut utiliser parallelMng(). Ce gestionnaire
   * sert essentiellement à effectuer des opérations sur l'ensemble
   * des sous-domaines et leur réplicats. S'il n'y a pas de réplication,
   * ce gestionnaire est le même que parallelMng().
   */
  virtual IParallelMng* allReplicaParallelMng() const =0;

  //! Retourne le gestionnaire de thread
  virtual IThreadMng* threadMng() =0;

  //! Retourne le gestionnaire d'historique
  virtual ITimeHistoryMng* timeHistoryMng() =0;

  //! Retourne le gestionnaire de la boucle en temps
  virtual ITimeLoopMng* timeLoopMng() =0;

  //! Retourne le gestionnaire des entrées/sorties.
  virtual IIOMng* ioMng() =0;

  //! Retourne le gestionnaire du jeu de données.
  virtual ICaseMng* caseMng() =0;

  //! Retourne le gestionnaire de timers
  virtual ITimerMng* timerMng() const =0;

  //! Gestionnaire de protections
  virtual ICheckpointMng* checkpointMng() const =0;

  //! Gestionnaire de propriétés
  virtual IPropertyMng* propertyMng() const =0;

  //! Statistiques des temps d'exécution
  virtual ITimeStats* timeStats() const =0;

  //! Gestionnaire d'informations mémoire
  virtual IMemoryInfo* memoryInfo() const =0;

  //! Système d'unité du sous-domaine.
  virtual IPhysicalUnitSystem* physicalUnitSystem() =0;

  //! Retourne le gestionnaire d'équilibrage de charge.
  virtual ILoadBalanceMng* loadBalanceMng() =0;

  //! Retourne le gestionnaire de maillage.
  virtual IMeshMng* meshMng() const =0;

  //! Interface du module maître.
  virtual IModuleMaster* moduleMaster() const =0;

  //! Configuration associée.
  virtual const IConfiguration* configuration() const =0;

  //! Configuration associée.
  virtual IConfiguration* configuration() =0;

  //! Gestionnaire de l'accélérateur associé
  virtual IAcceleratorMng* acceleratorMng() =0;

 public:

  //! Numéro du sous-domaine associé à ce gestionnaire.
  virtual Int32 subDomainId() const =0;

  //! Nombre total de sous-domaines
  virtual Int32 nbSubDomain() const =0;

  //! Lit les informations de maillage du jeu de données
  virtual void readCaseMeshes() =0;

  /*!
   * \internal
   * \brief Positionne un flag indiquant qu'on effectue une
   * reprise.
   *
   * Cette méthode doit être appelée avant d'allouer le maillage (allocateMeshes()).
   */
  virtual void setIsContinue() =0;

  //! Vrai si on effectue une reprise, faux sinon.
  virtual bool isContinue() const =0;

  /*!
   * \internal
   * \brief Alloue les instances.
   *
   * Les instances de maillages sont simplements alloués mais ne contiennent pas d'entités.
   * Cette méthode doit être appelée avant toute autre opération impliquant le maillage,
   * en particulier avant la lecture des options du jeu de données ou la lecture des protections.
   */
  virtual void allocateMeshes() =0;

  /*!
   * \internal
   * \brief Lit ou relit les maillages.
   *
   * Au démarrage, les maillages sont relues à partir des informations du jeu de données.
   * En reprise, les maillages sont rechargés depuis une protection.
   * Cette méthode doit être appelée après l'appel à allocateMeshes().
   */
  virtual void readOrReloadMeshes() =0;

  /*!
   * \internal
   * \brief Initialise les variables dont les valeurs sont spécifiées dans
   * le jeu de données.
   */
  virtual void initializeMeshVariablesFromCaseFile() =0;

  /*!
   * \internal
   * \brief Applique le partitionnement de maillage de l'initialisation.
   */
  virtual void doInitMeshPartition() =0;

  //! Ajoute un maillage au sous-domaine
  ARCCORE_DEPRECATED_2020("Use meshMng()->meshFactoryMng() to create and add mesh")
  virtual void addMesh(IMesh* mesh) =0;

  //! Listes des maillages du sous-domaine
  virtual ConstArrayView<IMesh*> meshes() const =0;

  /*!
   * \internal
   * \brief Exécution des modules d'initialisation
   * \deprecated Cette méthode ne fait plus rien.
   */
  virtual ARCANE_DEPRECATED_2018 void doInitModules() =0;

  //! Exécution des modules de fin d'exécution
  virtual void doExitModules() =0;

  //! Affiche des informations sur l'instance
  virtual void dumpInfo(std::ostream&) =0;

  /*!
   * \brief Maillage par défaut.
   *
   * Le maillage par défaut n'existe pas tant que le jeu
   * de données n'a pas été lu. Il est en général préférable
   * d'utiliser defautMeshHandle() à la place.
   */
  virtual IMesh* defaultMesh() =0;
  
  /*!
   * \brief Handle sur le maillage par défaut.
   *
   * Ce handle existe toujours même si le maillage associé n'a pas
   * encore été créé.
   */
  virtual const MeshHandle& defaultMeshHandle() =0;

  virtual ARCANE_DEPRECATED IMesh* mesh() =0;

  /*! \brief Recherche le maillage de nom \a name.
   *
   * Si le maillage n'est pas trouvé, la méthode lance une exception
   * si \a throw_exception vaut \a true ou retourne 0 si \a throw_exception
   * vaut \a false.
   */
  ARCCORE_DEPRECATED_2019("Use meshMng()->findMeshHandle() instead")
  virtual IMesh* findMesh(const String& name,bool throw_exception=true) =0;
  
  //! Indique si la session a été initialisée.
  virtual bool isInitialized() const =0;

  /*!
   * \internal
   * \brief Indique que le sous-domaine est initialié.
   */
  virtual void setIsInitialized() =0;

  //! Informations sur l'exécutable
  virtual const ApplicationInfo& applicationInfo() const =0;

  //! Document XML du cas.
  virtual ICaseDocument* caseDocument() =0;

  /*!
   * \brief Vérifie qu'un identifiant est valide
   *
   * \exception ExceptionBadName si \a id n'est pas valide comme identifiant.
   */
  virtual void checkId(const String& where,const String& id) =0;

  //! Chemin complet du fichier contenant le jeu de données
  virtual const String& caseFullFileName() const =0;

  //! Nom du cas
  virtual const String& caseName() const =0;

  //! Remplit \a bytes avec le contenue du jeu de données.
  virtual void fillCaseBytes(ByteArray& bytes) const =0;

  /*! \brief Positionne le nom du cas.
   *
   Cette méthode doit être appelée avant l'initialisation.
  */
  virtual void setCaseName(const String& name) =0;

  /*!
   * \brief Positionne le partitionneur initial.
   *
   * Si cette méthode n'est pas appelée, le partitionneur
   * par défaut est utilisé.
   *
   * Cette méthode doit être appelée avant l'initialisation des modules,
   * par exemple dans les points d'entrée de construction.
   *
   * L'instance s'approprie \a partitioner et le détruira par delete
   * à la fin du calcul.
   */
  virtual void setInitialPartitioner(IInitialPartitioner* partitioner) =0;

  //! Options générales du jeu de donnée.
  virtual const CaseOptionsMain* caseOptionsMain() const =0;

  //! Répertoire de base des exportations.
  virtual const IDirectory& exportDirectory() const =0;

  /*! \brief Positionne le chemin de sortie des exportations (protections et reprises)
   
   Le répertoire correspondant à \a dir doit exister.
   
   Cette méthode doit être appelée avant l'initialisation.
  */
  virtual void setExportDirectory(const IDirectory& dir) =0;

  //! Répertoire de base des exportations nécessitant un archivage.
  virtual const IDirectory& storageDirectory() const =0;

  /*! \brief Positionne le chemin de sortie des exportations nécessitant un archivage.
   
    Ce répertoire permet d'indiquer un répertoire qui peut être archivé automatiquement.
    S'il est nul, on utilise le exportDirectory().
   
    Cette méthode doit être appelée avant l'initialisation.
  */
  virtual void setStorageDirectory(const IDirectory& dir) =0;

  //! Répertoire de base des listings (logs, info exécution).
  virtual const IDirectory& listingDirectory() const =0;

  /*! \brief Positionne le chemin de sortie des infos listing
   *
   Le répertoire correspondant à \a dirname doit exister.

   Cette méthode doit être appelée avant l'initialisation.
  */
  virtual void setListingDirectory(const IDirectory& dir) =0;

  //! Informations sur les variables standards
  virtual const CommonVariables& commonVariables() const =0;

  /*!
   * \brief Sort les informations internes de l'architecture.
   * Les informations sont stockées dans un arbre XML ayant pour élément
   * racine \a root.
   * Ces informations sont à usage interne d'Arcane.
   */
  virtual void dumpInternalInfos(XmlNode& elem) =0;

  /*! \brief Dimension du maillage (1D, 2D ou 3D).
   *
   * \deprecated Utiliser mesh()->dimension() à la place.
   */
  virtual Integer ARCANE_DEPRECATED meshDimension() const =0;

  /*!
   * \brief Notification avant destruction du sous-domaine
   */
  virtual IObservable* onDestroyObservable() =0;

  //! Service d'exécution directe (ou null)
  virtual IDirectExecution* directExecution() const =0;

  /*!
   * \brief Positionne le service d'exécution directe.
   *
   * Ce service doit être positionné lors de la création des services lors
   * de la lecture du jeu de donnée.
   */
  virtual void setDirectExecution(IDirectExecution* v) =0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

