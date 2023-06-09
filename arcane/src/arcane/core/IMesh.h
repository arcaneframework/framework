// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2023 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IMesh.h                                                     (C) 2000-2023 */
/*                                                                           */
/* Interface d'un maillage.                                                  */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_IMESH_H
#define ARCANE_CORE_IMESH_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/ArcaneTypes.h"
#include "arcane/core/VariableTypedef.h"
#include "arcane/core/IMeshBase.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class IParallelMng;
class MeshItemInternalList;
class IParticleExchanger;
class XmlNode;
class IMeshUtilities;
class IMeshModifier;
class IMeshMng;
class Properties;
class IMeshPartitionConstraintMng;
class IExtraGhostCellsBuilder;
class IUserData;
class IUserDataList;
class IGhostLayerMng;
class IMeshChecker;
class IMeshCompactMng;
class MeshPartInfo;
class IItemFamilyNetwork;
class MeshHandle;
class IVariableMng;
class ItemTypeMng;
class IMeshUniqueIdMng;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
//INFO: La doc complete est dans Mesh.dox
class IMesh
: public IMeshBase
{
 public:

  virtual ~IMesh() = default; //<! Libère les ressources

 public:

  virtual void build() =0;


  //! Nom de la fabrique utilisée pour créer le maillage
  virtual String factoryName() const =0;

  //! Tableau interne des éléments du maillage de type \a type
  virtual ItemInternalList itemsInternal(eItemKind) =0;

  //! Coordonnées des noeuds
  virtual SharedVariableNodeReal3 sharedNodesCoordinates() =0;

  //! Vérification de la validité des structues internes de maillage (interne)
  virtual void checkValidMesh() =0;

  /*!
   * \brief Vérification de la validité du maillage.
   *
   * Il s'agit d'une vérification globale entre tous les sous-domaines.
   *
   * Elle vérifie notamment que la connectivité est cohérente entre
   * les sous-domaines.
   *
   * La vérification peut-être assez coûteuse en temps CPU.
   * Cette méthode est collective.
   */
  virtual void checkValidMeshFull() =0;

  /*!
   * \brief Synchronise tous les groupes et les variables du maillage.
   *
   * Cette opération est collective
   */
  virtual void synchronizeGroupsAndVariables()=0;

 public:

  /*! \brief Vrai si le maillage est allouée.
   *
   * Un maillage est alloué dès qu'une entité a été ajouté, par allocateCells(),
   *  ou reloadMesh()
   */
  virtual bool isAllocated() =0;

  /*!
   * \brief Compteur indiquant le temps de dernière modification du maillage.
   * Ce compteur augmente à chaque appel à endUpdate(). Il vaut 0 lors
   * de l'initialisation.
   * \node Actuellement, ce compteur n'est pas sauvegardé lors d'une protection.
   */
  virtual Int64 timestamp() =0;
  
 public:

  //! Sous-domaine associé
  ARCANE_DEPRECATED_LONG_TERM("Y2020: Do not use this method. Try to get 'ISubDomain' from another way")
  virtual ISubDomain* subDomain() =0;

 public:

  //! Gestionnaire de parallèlisme
  virtual IParallelMng* parallelMng() =0;

 public:

  //! Descripteur de connectivité
  /*! Cet objet permet de lire/modifier la connectivité */
  virtual VariableScalarInteger connectivity() = 0;

//! AMR
  //! Groupe de toutes les mailles actives
  virtual CellGroup allActiveCells() =0;

  //! Groupe de toutes les mailles actives et propres au domaine
  virtual CellGroup ownActiveCells() =0;

  //! Groupe de toutes les mailles de niveau \p level
  virtual CellGroup allLevelCells(const Integer& level) =0;

  //! Groupe de toutes les mailles propres de niveau \p level
  virtual CellGroup ownLevelCells(const Integer& level) =0;

  //! Groupe de toutes les faces actives
   virtual FaceGroup allActiveFaces() =0;

   //! Groupe de toutes les faces actives propres au domaine.
   virtual FaceGroup ownActiveFaces() =0;

  //! Groupe de toutes les faces actives
  virtual FaceGroup innerActiveFaces() =0;

  //! Groupe de toutes les faces actives sur la frontière.
  virtual FaceGroup outerActiveFaces() =0;

  //!
//  virtual void readAmrActivator(const XmlNode& xml_node) =0;

 public:

  //! Liste des groupes
  virtual ItemGroupCollection groups() =0;

  //! Retourne le groupe de nom \a name ou le groupe nul s'il n'y en a pas.
  virtual ItemGroup findGroup(const String& name) =0;

  //! Détruit tous les groupes de toutes les familles.
  virtual void destroyGroups() =0;

 public:

  virtual MeshItemInternalList* meshItemInternalList() =0;

 public:

  virtual void updateGhostLayers(bool remove_old_ghost) =0;
  /*!
   * \internal
   * \deprecated Utiliser IMesh::cellFamily()->policyMng()->createSerializer() à la place.
   */
  ARCANE_DEPRECATED_240 virtual void serializeCells(ISerializer* buffer,Int32ConstArrayView cells_local_id) =0;

  //! Prépare l'instance en vue d'une protection
  virtual void prepareForDump() =0;
  
  //! Initialize les variables avec les valeurs du fichier de configuration (interne)
  virtual void initializeVariables(const XmlNode& init_node) =0;

  /*!
   * \brief Positionne le niveau de vérification du maillage.
   *
   * 0 - tests désactivés
   * 1 - tests partiels, après les endUpdate()
   * 2 - tests complets, après les endUpdate()
   */
  virtual void setCheckLevel(Integer level) =0;
  
  //! Niveau actuel de vérification
  virtual Integer checkLevel() const =0;

  //! Indique si le maillage est dynamique (peut évoluer)
  virtual bool isDynamic() const =0;

  //!
  virtual bool isAmrActivated() const =0;

 public:

  //! \name Gestions des interfaces semi-conformes
  //@{
  //! Détermine les interfaces de semi-conformités
  virtual void computeTiedInterfaces(const XmlNode& mesh_node) =0;

  //! Vrai s'il existe des interfaces semi-conformes dans le maillage
  virtual bool hasTiedInterface() =0;
  
  //! Liste des interfaces semi-conformes
  virtual TiedInterfaceCollection tiedInterfaces() =0;
  //@}

  //! Gestionnaire des contraintes de partitionnement associées à ce maillage.
  virtual IMeshPartitionConstraintMng* partitionConstraintMng() =0;

 public:
  //! Interface des fonctions utilitaires associée
  virtual IMeshUtilities* utilities() =0;

  //! Propriétés associées à ce maillage
  virtual Properties* properties() =0;

 public:

  //! Interface de modification associée
  virtual IMeshModifier* modifier() =0;

 public:

  //! Coordonnées des noeuds
  /*! Retourne un tableau natif (non partagé comme SharedVariable) des coordonnées.
   *  Cet appel n'est légal que sur un maillage primaire (non sous-maillage).
   */
  virtual VariableNodeReal3& nodesCoordinates() =0;

  //@{ @name Interface des sous-maillages
  //! Définit les maillage et groupe parents
  /*! Doit être positionné sur le maillage en construction _avant_ la phase build() */
  virtual void defineParentForBuild(IMesh * mesh, ItemGroup group) =0;

  //! Accès au maillage parent
  /*! (NULL si n'a pas de maillage parent) */
  virtual IMesh * parentMesh() const = 0;

  //! Accès au maillage prent
  /*! (null() si n'a pas de maillage parent) */
  virtual ItemGroup parentGroup() const = 0;

  //! Ajoute un sous-maillage au maillage parent
  virtual void addChildMesh(IMesh * sub_mesh) = 0;

  //! Liste des sous-maillages du maillage courant
  virtual MeshCollection childMeshes() const = 0;

  //@}

 public:

  /*!
   * \brief Indique si l'instance est un maillage primaire.
   *
   * Pour être un maillage primaire, l'instance doit
   * pouvoir être convertie en un IPrimaryMesh
   * et ne pas être un sous-maillage, c'est à dire ne
   * pas avoir de maillage parent (parentMesh()==nullptr).
   */
  virtual bool isPrimaryMesh() const =0;

  /*!
   * \brief Retourne l'instance sous la forme d'un IPrimaryMesh.
   *
   * Renvoie une exception de type BadCastException si l'instance
   * n'est pas du type IPrimaryMesh et si isPrimaryMesh() est faux.
   */
  virtual IPrimaryMesh* toPrimaryMesh() =0;

 public:

  //! Gestionnnaire de données utilisateurs associé
  virtual IUserDataList* userDataList() =0;

  //! Gestionnnaire de données utilisateurs associé
  virtual const IUserDataList* userDataList() const =0;

 public:

  //! Gestionnare de couche fantômes associé
  virtual IGhostLayerMng* ghostLayerMng() const =0;

  //! Gestionnare de la numérotation des identifiants uniques
  virtual IMeshUniqueIdMng* meshUniqueIdMng() const =0;

  //! Interface du vérificateur.
  virtual IMeshChecker* checker() const =0;

  //! Informations sur les parties du maillage
  virtual const MeshPartInfo& meshPartInfo() const =0;

  //! check if the network itemFamily dependencies is activated
  virtual bool useMeshItemFamilyDependencies() const =0;

  //! Interface du réseau de familles (familles connectées)
  virtual IItemFamilyNetwork* itemFamilyNetwork() =0;

  //! Interface du gestionnaire des connectivités incrémentales indexées.
  virtual IIndexedIncrementalItemConnectivityMng* indexedConnectivityMng() =0;

  //! Caractéristiques du maillage
  virtual const MeshKind meshKind() const =0;

 public:

  //! \internal
  virtual IMeshCompactMng* _compactMng() =0;

  /*!
   * \internal
   * \brief Politique d'utilisation des connectivitées
   */
  virtual InternalConnectivityPolicy _connectivityPolicy() const =0;

 public:

  //! Gestionnaire de maillage associé
  virtual IMeshMng* meshMng() const =0;

  //! Gestionnaire de variable associé
  virtual IVariableMng* variableMng() const =0;

  //! Gestionnaire de types d'entités associé
  virtual ItemTypeMng* itemTypeMng() const =0;

 public:

  //! API interne à Arcane
  virtual IMeshInternal* _internalApi() =0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  
