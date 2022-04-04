﻿// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IItemFamily.h                                               (C) 2000-2020 */
/*                                                                           */
/* Interface d'une famille d'entités.                                        */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_IITEMFAMILY_H
#define ARCANE_IITEMFAMILY_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/ArcaneTypes.h"
#include "arcane/ItemTypes.h"
#include "arcane/VariableTypedef.h"
#include "arcane/Parallel.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class IParallelMng;
class IDataOperation;
class ItemUniqueId;
class IItemInternalSortFunction;
class IVariableSynchronizer;
class IParticleFamily;
class GroupIndexTable;
class IItemConnectivityInfo;
class IItemConnectivityMng;
class IItemConnectivity;
class IExtraGhostItemsBuilder;
class IIncrementalItemConnectivity;
class IItemFamilyPolicyMng;
class IItemFamilyTopologyModifier;
class ItemInternalConnectivityList;
class Properties;
namespace mesh{
  class ItemDataList;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \ingroup Mesh
 * \brief Interface d'une famille d'entités.
 *
 * Une famille d'entité gère toutes les entités de même genre (Item::kind())
 * et est attachée à un maillage (IMesh).
 *
 * Pour tout maillage, il existe une et une seule famille de
 * noeuds (Node), arêtes (Edge), faces (Face) et mailles (Cell).
 * Ces entités sont appelées des entités de \b base du maillage et les
 * familles associées les familles de base du maillage.
 *
 * Suivant l'implémentation, il peut aussi y avoir des familles
 * de particules (Particle), de noeuds duaux (DualNode) ou de liens (Link).
 * Suivant la connectivité demandée, une famille peut ne pas avoir d'éléments.
 * Par exemple, par défaut en 3D, les arêtes (Edge) ne sont pas créées.
 *
 * Chaque entité de la famille possède un identifiant local dans la
 * famille, donnée par Item::localId(). Lorsqu'une famille évolue, cet identifiant
 * peut être modifié. Les Item::localId() des entités d'une famille ne sont
 * pas nécessairement contigus. La méthode maxLocalId() permet de connaître
 * le maximum de ces valeurs. Le compactage permet
 * de garantir que les localId() sont renumérotés de 0 à (nbItem()-1). Pour les entités
 * de base du maillage, le compactage est automatique si le maillage
 * à la propriété \a "sort" à vrai. Pour les autres, il faut appeler
 * compactItems().
 *
 * Par défaut, une famille possède une table de conversion des
 * uniqueId() vers les localId(). Cette table doit exister pour
 * permettre les opérations suivantes:
 * - le uniqueId() est garanti unique sur le sous-domaine et doit
 * l'être par construction sur tous les sous-domaines.
 * - faire appel aux méthodes itemsUniqueIdToLocalId().
 * - les entités de la famille peuvent être présentes dans plusieurs
 * sous-domaines.
 * - faire des synchronisations.
 * - avoir des variables partielles sur cette famille
 *
 * Il est possible d'activer ou désactiver cette table de conversion
 * via la méthode setHasUniqueIdMap() uniquement si aucune entité
 * n'a été créée. Cette opération n'est pas possible sur
 * les familles de noeuds, arêtes, faces et mailles.
 
 * Lorsqu'on modifie une famille par ajout ou suppression d'entités, les
 * variables et les groupes qui reposent sur cette famille ne sont plus utilisables
 * tant qu'on a pas fait d'appel à endUpdate(). Il est possible pour des raisons
 * d'optimisation de faire des mise à jour de certaines variables ou groupes via
 * partialEndUpdateVariable() ou partialEndUpdateGroup(). ATTENTION, un appel
 * à l'une de ces 3 méthodes de mise à jour invalide les instances des entités (Item).
 * Pour conserver une référence sur une entité, il faut soit utiliser un groupe (ItemGroup),
 * soit conserver son numéro unique et utiliser itemsUniqueIdToLocalId().
 *
 */
class IItemFamily
{
 public:

  virtual ~IItemFamily() {} //<! Libère les ressources

 public:

  virtual void build() =0;

 public:

  //! Nom de la famille
  virtual const String& name() const =0;

  //! Nom complet de la famille (avec celui du maillage)
  virtual const String& fullName() const =0;

  //! Genre des entités
  virtual eItemKind itemKind() const =0;
  
  //! Nombre d'entités
  virtual Integer nbItem() const =0;

  /*!
   * \brief Taille nécessaire pour dimensionner les variables sur ces entités.
   * \deprecated Utiliser maxLocalId() à la place.
   */
  virtual ARCANE_DEPRECATED_112 Int32 variableMaxSize() const =0;

  /*!
   * Taille nécessaire pour dimensionner les variables sur ces entités.
   *
   * Il s'agit du maximum des Item::localId() des entités de cette famille plus 1.
   */
  virtual Int32 maxLocalId() const =0;

  //! Tableau interne des entités
  virtual ItemInternalArrayView itemsInternal() =0;

  //! IItemFamily parent
  /*! Issue des imbrications de sous-maillages
   *  \return NULL si n'a pas de famille parente
   */
  virtual IItemFamily* parentFamily() const = 0;

  //! Positionne l'IItemFamily parent
  /*! A utiliser avant build() pour les sous-maillages construit dynamiquement, 
   *   ie pas depuis un reprise */
  virtual void setParentFamily(IItemFamily * parent) = 0;

  //! Donne la profondeur d'imbrication du maillage courant
  virtual Integer parentFamilyDepth() const = 0;

  //! Ajoute d'une famile en dépendance
  /*! Opération en symétrie de setParentFamily */
  virtual void addChildFamily(IItemFamily * family) = 0;

  //! Familles enfantes de cette famille
  virtual IItemFamilyCollection childFamilies() = 0;

  /*!
   * \brief Variable contenant le numéro du nouveau sous-domaine
   * propriétaire de l'entité.
   *
   * Cette variable n'est utilisée que pour un repartitionnement du maillage.
   */
  virtual VariableItemInt32& itemsNewOwner() =0;
  
  //! Vérification de la validité des structures internes (interne)
  virtual void checkValid() =0;

  /*!
   * \brief Vérification de la validité des structures internes concernant
   * la connectivité.
   */
  virtual void checkValidConnectivity() =0;

  /*!
   * \brief Vérifie que les identifiants \a unique_ids sont bien uniques
   * pour tous les sous-domaines.
   *
   * Cette méthode NE vérifie PAS que les \a unique_ids sont identiques
   * à ceux des entités déjà créées. Elle vérifie uniquement l'ensemble des
   * \a unique_ids passés en argument par tous les sous-domaines.
   *
   * Cette opération est collective et doit être appelée par tous les sous-domaines.
   */
  virtual void checkUniqueIds(Int64ConstArrayView unique_ids) =0;

 public:

#if 1
  /*!
   * \brief Alloue des entités.
   *
   * Après appel à cette opération, il faut appeler endUpdate() pour notifier
   * à l'instance la fin des modifications. Il est possible d'enchaîner plusieurs
   * allocations avant d'appeler endUpdate(). Les \a unique_ids doivent
   * être unique sur l'ensemble des sous-domaines. Il est possible de vérifier
   * cela via checkUniqueIds().
   * \a items doit avoir le même nombre d'éléments que \a unique_ids
   * et sera remplit en retour avec les numéros locaux des entités créées.
   * Il est possible après création d'obtenir une vue sur les entités créés via view().
   * \code
   * Int64UniqueArray my_unique_ids;
   * Int32UniqueArray my_local_ids;
   * my_unique_ids.add(25);
   * my_unique_ids.add(32);
   * family->addItems(my_unique_ids,my_local_ids);
   * family->endUpdate();
   * ENUMERATE_ITEM(iitem,family->view(my_local_ids)){
   *   const Item& item = *iitem;
   *   info() << "Item local_id=" << item.localId() << " unique_id=" << item.uniqueId();
   * }
   * \endcode
   */
  virtual ARCANE_DEPRECATED_114 void addItems(Int64ConstArrayView unique_ids,Int32ArrayView items) =0;

  /*!
   * \deprecated.
   * Il faut utiliser addItems(Int64ConstArrayView,Int32ArrayView)
   */
  virtual ARCANE_DEPRECATED_114 void addItems(Int64ConstArrayView unique_ids,ArrayView<Item> items) =0;

  /*!
   * \deprecated.
   * Il faut utiliser addItems(Int64ConstArrayView,Int32ArrayView)
   */
  virtual ARCANE_DEPRECATED_114 void addItems(Int64ConstArrayView unique_ids,ItemGroup item_group) =0;

  /*!
   * \brief Échange des entités.
   *
   * Cette méthode n'est supportée que pour les familles de particule.
   * Pour les éléments du maillage comme les noeuds, faces ou maille, il faut utiliser IMesh::exchangeItems().
   *
   * Les nouveaux propriétaires des entités sont données par la itemsNewOwner().
   *
   * Cette opération est bloquante et collective.
   */
  virtual ARCANE_DEPRECATED_114 void exchangeItems() =0;
#endif

  /*!
   * \brief Vue sur les entités.
   *
   * Retourne une vue sur les entités de numéro locaux \a local_ids.
   * \warning Cette vue n'est valide que tant que la famille n'évolue pas.
   * En particulier, l'ajout, la suppression ou le compactage invalide la vue.
   * Si vous souhaitez conserver une liste même après modification, il faut
   * utiliser les groupes (ItemGroup).
   */
  virtual ItemVectorView view(Int32ConstArrayView local_ids) =0;

  /*!
   * \brief Vue sur toutes les entités de la famille.
   */
  virtual ItemVectorView view() =0;

#if 1
  /*!
   * \brief Supprime des entités.
   */
  virtual ARCANE_DEPRECATED_114 void removeItems(Int32ConstArrayView local_ids,bool keep_ghost=false) =0;
#endif
  
  /*!
     * \brief Supprime des entités.
     *
     * Utilise le graphe (Familles, Connectivités) ItemFamilyNetwork
     */
    virtual void removeItems2(mesh::ItemDataList& item_data_list) =0;

    /*!
     * \brief Supprime des entités et met a jour les connectivites. Ne supprime pas d'eventuels sous items orphelins.
     *
     * Contexte d'utilisation avec un graphe des familles. Les sous items orphelins ont du eux aussi etre marque NeedRemove.
     * Il n'y a donc pas besoin de les gerer dans les familles parentes.
     */
    virtual void removeNeedRemoveMarkedItems() =0;

  /*
   * \brief Entité de numéro unique \a unique_id.
   *
   * Si aucune entité avec cet \a unique_id n'est trouvé, retourne null.
   */
  virtual ItemInternal* findOneItem(Int64 unique_id) =0;
 
  /** 
   * Fusionne deux entités en une plus grande. Par exemple, deux
   * mailles partageant une face. L'intérêt est de conserver ainsi les
   * uniqueIds() en parallèle.
   * 
   * @note La maille résultante est construite en remplacement de la
   * première maille, de numéro \a local_id1
   * 
   * @param local_id1 numéro local de la première entité
   * @param local_id2 numéro local de la seconde entité
   */
  ARCANE_DEPRECATED_240 virtual void mergeItems(Int32 local_id1,Int32 local_id2) = 0;

  /** 
   * Détermine quel sera le localId de la maille apres fusion.
   * 
   * @param local_id1 numéro local de la première entité
   * @param local_id2 numéro local de la seconde entité
   * 
   * @return le local id de la maille fusionnée
   */
  ARCANE_DEPRECATED_240 virtual Int32 getMergedItemLID(Int32 local_id1,Int32 local_id2) = 0;

  /*! \brief Notifie la fin de modification de la liste des entités.
   *
   * Cette méthode doit être appelée après modification de la liste des
   * entités (après ajout ou suppression). Elle met à jour les groupes
   * et redimensionne les variables sur cette famille.
   */
  virtual void endUpdate() =0;

  /*!
   * \brief Mise à jour partielle.
   *
   * Met à jour les structures internes après une modification de la famille.
   * Il s'agit d'une version optimisée de endUpdate() lorsqu'on souhaite
   * faire de multiples modifications de maillage. Cette méthode NE met PAS
   * à jour les groupes ni les variables associées à cette famille. Seul le
   * groupe allItems() est disponible. Il est possible de mettre à jour
   * un groupe via partialEndUpdateGroup() et une variable via partialEndUpdateVariable().
   *
   * Cette méthode est réservée aux utilisateurs expérimentés. Pour les autres,
   * il vaut mieux utiliser endUpdate().
   */
  virtual void partialEndUpdate() =0;

  /*!
   * \brief Met à jour un groupe.
   
   * Met à jour le groupe \a group après une modification de la famille.
   * La mise à jour consiste à supprimer du groupe les entités de la famille
   * éventuellement détruites lors de la modification.
   *
   * \sa partialEndUpdate().
   */
  virtual void partialEndUpdateGroup(const ItemGroup& group) =0;
  
  /*!
   * \brief Met à jour une variable.
   *
   * Met à jour la variable \a variable après une modification de la famille.
   * La mise à jour consiste à redimensionner la variable après un éventuel
   * ajout d'entités.
   *
   * \sa partialEndUpdate().
   */
  virtual void partialEndUpdateVariable(IVariable* variable) =0;

  //! Notifie que les entités propres au sous-domaine de la famille ont été modifiées
  virtual void notifyItemsOwnerChanged() =0;

  //! Notifie que les numéros uniques des entités ont été modifiées
  virtual void notifyItemsUniqueIdChanged() =0;

  /*!
   * \internal
   * \brief Redimensionne les variables
   */
  virtual void resizeVariables(bool force_resize=false) =0;

 public:

  /*!
   * \name Informations sur la connectivité.
   *
   * Ces informations sont recalculées dynamiquement lorsque le maillage évolue.
   *
   *
   */
  //@{
  /*! Nombre maximal de noeuds par entité.
   * \deprecated Utiliser localConnectivityInfos()->maxNodePerItem() à la place.
   */    
  virtual ARCANE_DEPRECATED_122 Integer maxNodePerItem() const =0;
  
  /*! Nombre maximal d'arêtes par entité
   * \deprecated Utiliser localConnectivityInfos()->maxEdgePerItem() à la place.
   */
  virtual ARCANE_DEPRECATED_122 Integer maxEdgePerItem() const =0;

  /*! Nombre maximal de faces par entité
   * \deprecated Utiliser localConnectivityInfos()->maxFacePerItem() à la place.
   */
  virtual ARCANE_DEPRECATED_122 Integer maxFacePerItem() const =0;

  /*! Nombre maximal de mailles par entité
   * \deprecated Utiliser localConnectivityInfos()->maxCellPerItem() à la place.
   */
  virtual ARCANE_DEPRECATED_122 Integer maxCellPerItem() const =0;

  /*! Nombre maximal de noeuds pour un type d'entité
   * \deprecated Utiliser localConnectivityInfos()->maxNodeInItemTypeInfo() à la place.
   */
  virtual ARCANE_DEPRECATED_122 Integer maxLocalNodePerItemType() const =0;
  
  /*! Nombre maximal d'arêtes pour un type d'entité
   * \deprecated Utiliser localConnectivityInfos()->maxEdgeInItemTypeInfo() à la place.
   */
  virtual ARCANE_DEPRECATED_122 Integer maxLocalEdgePerItemType() const =0;

  /*! Nombre maximal de faces pour un type d'entité
   * \deprecated Utiliser localConnectivityInfos()->maxFaceInItemTypeInfo() à la place.
   */
  virtual ARCANE_DEPRECATED_122 Integer maxLocalFacePerItemType() const =0;
  //@}

  //! Informations sur la connectivité locales au sous-domaine pour à cette famille
  virtual IItemConnectivityInfo* localConnectivityInfos() const =0;

  //! Informations sur la connectivité globales à tous les sous-domaines.
  virtual IItemConnectivityInfo* globalConnectivityInfos() const =0;

 public:

  /*!
   * \brief Indique si la famille possède une table de conversion
   * uniqueId vers localId.
   *
   * Cette méthode ne peut être appelée que lorsqu'il n'y a aucune
   * entité de la famille.
   *
   * Les familles de noeuds, arêtes, faces et mailles du maillage
   * ont obligatoirement une table de conversion.
   */
  virtual void setHasUniqueIdMap(bool v) =0;

  //! Indique si la famille possède une table de conversion uniqueId vers localId.
  virtual bool hasUniqueIdMap() const =0;

 public:

  /*!
    \brief Converti un tableau de numéros uniques en numéros locaux.

    Cette opération prend en entrée le tableau \a unique_ids contenant les
    numéros uniques des entités du type \a item_kind et retourne dans
    \a local_ids le numéro local à ce sous-domaine correspondant.

    La complexité de cette opération dépend de l'implémentation.
    L'implémentation par défaut utilise une table de hachage. La complexité
    moyenne est donc constante.

    Si \a do_fatal est vrai, une erreur fatale est générée si une entité n'est
    pas n'est trouvée, sinon l'élément non trouvé a pour valeur NULL_ITEM_ID.
  */
  virtual void itemsUniqueIdToLocalId(Int32ArrayView local_ids,
                                      Int64ConstArrayView unique_ids,
                                      bool do_fatal=true) const =0;

  /*!
    \brief Converti un tableau de numéros uniques en numéros locaux.

    Cette opération prend en entrée le tableau \a unique_ids contenant les
    numéros uniques des entités du type \a item_kind et retourne dans
    \a local_ids le numéro local à ce sous-domaine correspondant.

    La complexité de cette opération dépend de l'implémentation.
    L'implémentation par défaut utilise une table de hachage. La complexité
    moyenne est donc constante.

    Si \a do_fatal est vrai, une erreur fatale est générée si une entité n'est
    pas n'est trouvée, sinon l'élément non trouvé a pour valeur NULL_ITEM_ID.
  */
  virtual void itemsUniqueIdToLocalId(Int32ArrayView local_ids,
                                      ConstArrayView<ItemUniqueId> unique_ids,
                                      bool do_fatal=true) const =0;

 public:

  /*!
   * \brief Positionne la fonction de tri des entités.
   *
   * La méthode par défaut est de trier les entités par uniqueId() croissant.
   * Si \a sort_function est nul, c'est la méthode par défaut qui sera utilisée.
   * Sinon, \a sort_function remplace la fonction précédente qui est détruite
   * (via delete).
   * Le tri est effectué via l'appel à compactItems().
   * \sa itemSortFunction()
   */
  virtual void setItemSortFunction(IItemInternalSortFunction* sort_function) =0;

  /*!
   * \brief Fonction de tri des entités.
   *
   * L'instance de cette classe reste propriétaire de l'objet retournée
   * qui ne doit pas être détruit ni modifié.
   * \sa setItemSortFunction()
   */
  virtual IItemInternalSortFunction* itemSortFunction() const =0;

 public:

  /*!
   * \internal
   * \brief Indique une fin d'allocation.
   *
   * Cette méthode ne doit normalement être appelée que par le maillage (IMesh)
   * au moment de l'allocate.
   *
   * Cette méthode est collective.
   */
  virtual void endAllocate() =0;

  /*!
   * \internal
   * \brief Indique une fin de modification par le maillage.
   *
   * Cette méthode ne doit normalement être appelée que par le maillage (IMesh)
   * à la fin d'un endUpdate().
   *
   * Cette méthode est collective.
   */
  virtual void notifyEndUpdateFromMesh() =0;

 public:

  //! Sous-domaine associé
  ARCCORE_DEPRECATED_2020("Do not use this method. Try to get 'ISubDomain' from another way")
  virtual ISubDomain* subDomain() const =0;

  //! Gestionnaire de trace associé
  virtual ITraceMng* traceMng() const =0;

  //! Maillage associé
  virtual IMesh* mesh() const =0;

  //! Gestionnaire de parallélisme associé
  virtual IParallelMng* parallelMng() const =0;

 public:

  //! Groupe de toutes les entités
  virtual ItemGroup allItems() const =0;

  //! Liste des groupes de cette famille
  virtual ItemGroupCollection groups() const =0;

 public:

  //! @name opérations sur des groupes
  //@{
  /*!
    \brief Recherche un groupe.
    \param name nom du groupe à rechercher
    \return le groupe de nom \a name ou le groupe nul s'il n'y en a pas.
  */
  virtual ItemGroup findGroup(const String& name) const =0;

  /*!
   * \brief Recherche un groupe
   *
   * \param name nom du groupe à rechercher
   *
   * \return le groupe trouvé ou le groupe nul si aucun groupe de nom
   * \a name et de type \a type n'existe et si \a create_if_needed vaut \e false.
   * Si \a create_if_needed vaux \e vrai, un groupe vide de nom \a name est créé et retourné.
   */
  virtual ItemGroup findGroup(const String& name,bool create_if_needed) =0;
  

  /*! 
   * \brief Créé un groupe d'entités de nom \a name contenant les entités \a local_ids.
   *
   * \param name nom du groupe
   * \param local_ids liste des localId() des entités composant le groupe.
   * \param do_override si \e true et q'un groupe de même nom existe déjà,
   * ses éléments sont remplacés par ceux donnés dans \a local_ids. Si \e false,
   * alors une exception est levée.
   * \return le groupe créé
   */
  virtual ItemGroup createGroup(const String& name,Int32ConstArrayView local_ids,bool do_override=false) =0;

  /*!
   * \brief Créé un groupe d'entités de nom \a name
   *
   * Le groupe ne doit pas déjà exister.
   *
   * \param name nom du groupe
   * \return le groupe créé
   */
  virtual ItemGroup createGroup(const String& name) =0;

  /*!
   * \brief Supprime tous les groupes de cette famille.
   */
  virtual void destroyGroups() =0;

  /*!
   * \internal
   * For Internal Use Only
   */
  virtual ItemGroup createGroup(const String& name,const ItemGroup& parent,bool do_override=false) =0;

  //@}

  /*!
   * \brief Recherche la variable de nom \a name associée à cette famille.
   *
   * Si aucune variable de nom \a name n'existe, si \a throw_exception vaut
   * \a false, retourne 0, sinon lève une exception.
   */
  virtual IVariable* findVariable(const String& name,bool throw_exception=false) =0;

  /*!
   * \brief Ajoute à la collection \a collection la liste des variables
   * utilisés de cette famille.
   */
  virtual void usedVariables(VariableCollection collection) =0;

 public:

  //! Prépare les données pour une protection
  virtual void prepareForDump() =0;

  //! Relit les données à partir d'une protection
  virtual void readFromDump() =0;

  /** 
   * Copie les valeurs des entités numéros @a source dans les entités
   * numéros @a destination
   * 
   * @param source liste des @b localId source
   * @param destination liste des @b localId destination
   */
  virtual void copyItemsValues(Int32ConstArrayView source, Int32ConstArrayView destination) =0;

  /** 
   * Copie les moyennes des valeurs des entités numéros
   * @a first_source et @a second_source dans les entités numéros
   * @a destination
   * 
   * @param first_source liste des @b localId de la 1ère source
   * @param second_source  liste des @b localId de la 2ème source
   * @param destination  liste des @b localId destination
   */
  virtual void copyItemsMeanValues(Int32ConstArrayView first_source,
                                   Int32ConstArrayView second_source,
                                   Int32ConstArrayView destination) = 0;

  /*!
   * \brief Supprime toutes les entités de la famille.
   * \warning attention à ne pas détruire des entités qui sont utilisées dans
   * par une autre famille. En général, il est plus prudent d'utiliser IMesh::clearItems()
   * si on souhaite supprimer tous les éléments du maillage.
   */
  virtual void clearItems() =0;

  //! Compacte les entités.
  virtual void compactItems(bool do_sort) =0;

 public:

  /*!
   * \internal
   * \brief Ajoute une variable à cette famille.
   *
   * Cette méthode est appelée par la variable elle même et ne doit pas
   * être apelée dans d'autres conditions.
   */
  virtual void addVariable(IVariable* var) =0;
  
  /*!
   * \internal
   * \brief Supprime une variable à cette famille.
   *
   * Cette méthode est appelée par la variable elle même et ne doit pas
   * être apelée dans d'autres conditions.
   */
  virtual void removeVariable(IVariable* var) =0;

 public:

  /*!
   * \brief Construit les structures nécessaires à la synchronisation.
   *
   Cette opération doit être effectuée à chaque fois que les entités
   du maillage changent de propriétaire (par exemple lors d'un équilibrage de
   charge).
   
   Cette opération est collective.
  */
  virtual void computeSynchronizeInfos() =0;

  //! Liste des sous-domaines communiquants pour les entités.
  virtual void getCommunicatingSubDomains(Int32Array& sub_domains) const =0;

  //! @name opérations de synchronisation d'une variable
  //@{

  //! Synchroniseur sur toutes les entités de la famille
  virtual IVariableSynchronizer* allItemsSynchronizer() =0;
  
  /*!
   * \brief Synchronise les variables \a variables.
   *
   * Les variables \a variables doivent être toutes être issues
   * de cette famille et ne pas être partielles.
   */
  virtual void synchronize(VariableCollection variables) =0;
  //@}
  
  /*!
   * \brief Applique une opération de réduction depuis les entités fantômes.
   *
   * Cette opération est l'opération inverse de la synchronisation.
   *
   * Le sous-domaine récupère les valeurs de la variable \a v sur les entités
   * qu'il partage avec d'autres sous-domaines et l'opération de réduction
   * \a operation est appliquée sur cette variable.
   */
  virtual void reduceFromGhostItems(IVariable* v,IDataOperation* operation) =0;
  /*!
   * \brief Applique une opération de réduction depuis les entités fantômes.
   *
   * Cette opération est l'opération inverse de la synchronisation.
   *
   * Le sous-domaine récupère les valeurs de la variable \a v sur les entités
   * qu'il partage avec d'autres sous-domaines et l'opération de réduction
   * \a operation est appliquée sur cette variable.
   */
  virtual void reduceFromGhostItems(IVariable* v,Parallel::eReduceType operation) =0;

  /*!
   * \brief Cherche une liste d'adjacence.
   *
   * Cherche la liste d'entités de type \a sub_kind, liées par
   * le type d'entité \a link_kind du groupe \a group,
   * sur un nombre de couche \a nb_layer.
   *
   * Si \a group et \a sub_group sont de même genre, une entité est toujours
   * dans sa liste d'adjacence, en tant que premier élément.
   *
   * Si la liste n'existe pas, elle est créée.
   *
   * \note pour l'instant une seule couche est autorisée.
   */
  virtual ItemPairGroup findAdjencyItems(const ItemGroup& group,
                                         const ItemGroup& sub_group,
                                         eItemKind link_kind,
                                         Integer nb_layer) =0;
  
  /*!
   * \brief Retourne l'interface de la famille de particule de cette famille.
   *
   * L'interface IParticleFamily n'existe que si cette famille est
   * une famille de particules (itemKind()==IK_Particle). Pour les
   * autres genres de famille, 0 est retourné.
   */
  virtual IParticleFamily* toParticleFamily() =0;

  /*!
   * \internal
   * \brief Supprime les entités donnés par \a local_ids.
   *
   * Pour usage interne uniquement. Si on souhaite supprimer des entités
   * du maillage, il faut passer par IMeshModifier via l'appel à IMesh::modifier().
   */
  virtual void internalRemoveItems(Int32ConstArrayView local_ids,bool keep_ghost=false) =0;


  /*!
   * \name Enregistre/Supprime un gestionnaire de connectivité.
   *
   * Permet de répercuter les évolutions de la famille dans les
   * connectivites "externes" ou elle est impliquée.
   * Ces connectivites "externes" sont aujourd'hui les connectivités
   * utilisant les degres de liberté.
   *
   * \note Ces méthodes sont internes à %Arcane.
   */
  //@{
  virtual void addSourceConnectivity(IItemConnectivity* connectivity) =0;
  virtual void addSourceConnectivity(IIncrementalItemConnectivity* connectivity) =0;
  virtual void addTargetConnectivity(IItemConnectivity* connectivity) =0;
  virtual void addTargetConnectivity(IIncrementalItemConnectivity* connectivity) =0;
  virtual void removeSourceConnectivity(IItemConnectivity* connectivity) =0;
  virtual void removeTargetConnectivity(IItemConnectivity* connectivity) =0;
  virtual void setConnectivityMng(IItemConnectivityMng* connectivity_mng) =0;
  //@}

  /*!
    * \brief Alloue des entités fantômes.
    *
    * Après appel à cette opération, il faut appeler endUpdate() pour
    * notifier à l'instance la fin des modifications. Il est possible
    * d'enchaîner plusieurs allocations avant d'appeler
    * endUpdate().
    *
    * Les \a unique_ids sont ceux d'items présents sur un autre
    * sous-domaine, dont le numéro est dans le tableau owners (de même
    * taille que le tableau unique_ids). \a items doit avoir le même
    * nombre d'éléments que \a unique_ids et sera remplit en retour
    * avec les numéros locaux des entités créées.
    */
  virtual void addGhostItems(Int64ConstArrayView unique_ids, Int32ArrayView items,
                             Int32ConstArrayView owners) =0;

 public:

  //! Interface des comportements/politiques associées à cette famille.
  virtual IItemFamilyPolicyMng* policyMng() =0;

  //! Propriétés associées à cette famille.
  virtual Properties* properties() =0;

  /*!
   * \internal
   * \brief Interface du modificateur de topologie.
   */
  virtual IItemFamilyTopologyModifier* _topologyModifier() =0;

  /*!
   * \internal
   * \brief Informations sur les connectivités non structurés
   */
  virtual ItemInternalConnectivityList* _unstructuredItemInternalConnectivityList() =0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  
