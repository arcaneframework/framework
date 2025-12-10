// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ItemGroupImpl.h                                             (C) 2000-2025 */
/*                                                                           */
/* Implémentation d'un groupe d'entités du maillage.                         */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_ITEMGROUPIMPL_H
#define ARCANE_ITEMGROUPIMPL_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/ItemTypes.h"
#include "arcane/core/SharedReference.h"
#include "arcane/utils/SharedPtr.h"

#include "arcane/core/GroupIndexTable.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

// Macro pour détecter les modifications de conception de ItemGroupImpl
#define ITEMGROUP_USE_OBSERVERS

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class IItemGroupObserver;
class IObservable;
class ItemGroupComputeFunctor;
class IMesh;
class ItemGroupInternal;
class ItemPairGroupImpl;
class GroupIndexTable;
class IVariableSynchronizer;
class ItemGroupImplInternal;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * \brief Implémentation d'un groupe d'entités de maillage.

 Un groupe est un ensemble d'entité du maillage (noeuds, faces, mailles,...)
 de même genre.

 Une instance de cette classe ne doit pas s'utiliser directement, mais
 par l'intermédiaire d'une instance de ItemGroup.

 Une entité élément ne peut être présente qu'une seule fois.

 Le développeur ne doit pas conserver directement des instances de cette
 class mais passer par un ItemGroup. Certains groupes étant déterminés
 dynamiquement suivant le contenu d'un autre groupe (par exemple, les groupes
 d'entités propres aux sous-domaines sont calculés à partir du groupe de
 toutes les entités du sous-domaine), ceci garantit que les mises à jour
 des groupes se font correctement.

 Cette instance est géré par un compteur de référence et est détruite
 automatiquement lorsqu'il arrive à zéro.
 */
class ARCANE_CORE_EXPORT ItemGroupImpl
: public SharedReference
{
 private:

  friend class ItemGroupChildrenByType;
  friend ItemGroup;
  class ItemSorter;

 public:

  //! Construit un groupe nul
  ItemGroupImpl();

  /*! \brief Construit un groupe.
   * Construit un groupe vide de nom \a name, associé à la famille \a family.
   */
  ItemGroupImpl(IItemFamily* family,const String& name);

  /*! \brief Construit un groupe fils d'un autre groupe.
   * Construit un groupe de nom \a name fils du groupe \a parent. Le genre de ce
   * groupe est le même que celui de la famille à laquelle il appartient.
   */
  ItemGroupImpl(IItemFamily* family,ItemGroupImpl* parent,const String& name);

  virtual ~ItemGroupImpl(); //!< Libère les ressources

 private:

  static ItemGroupImpl* shared_null;

 public:

  static ItemGroupImpl* checkSharedNull();

 public:

  virtual ISharedReference& sharedReference() { return *this; }

 public:

  //! Nom du groupe
  const String& name() const;

  //! Nom complet du groupe (avec maillage + famille)
  const String& fullName() const;

  //! Nombre de références sur le groupe.
  virtual Integer nbRef() const { return refCount(); }

  //! Groupe parent (0 si aucun)
  ItemGroupImpl* parent() const;

  //! Retourne \a true si le groupe est nul.
  bool null() const;

  //! Retourne si le groupe contient uniquement des éléments propres au sous-domaine
  bool isOwn() const;

  //! Positionne la propriété de groupe local ou non.
  void setOwn(bool v);

  //! Groupe des entité propres des entités de ce groupe
  ItemGroupImpl* ownGroup();

  //! Items in the group not owned by the subdomain
  ItemGroupImpl* ghostGroup();

  // Items in the group lying on the boundary between two subdomains
  // Implemented for faces only
  ItemGroupImpl* interfaceGroup();

  //! Groupe des noeuds des éléments de ce groupe
  ItemGroupImpl* nodeGroup();

  //! Groupe des arêtes des éléments de ce groupe
  ItemGroupImpl* edgeGroup();

  //! Groupe des faces des éléments de ce groupe
  ItemGroupImpl* faceGroup();

  //! Groupe des mailles des éléments de ce groupe
  ItemGroupImpl* cellGroup();

  //! Crée un sous-groupe calculé
  /*! Le gestion mémoire du functor est alors délégué au groupe */
  ItemGroupImpl* createSubGroup(const String& suffix, IItemFamily* family, ItemGroupComputeFunctor* functor);

  //! Accède à un sous-groupe calculé
  /*! Le gestion mémoire du functor est alors délégué au groupe */
  ItemGroupImpl* findSubGroup(const String& suffix);

  /*!
   *  \brief Groupe des faces internes des éléments de ce groupe
   *
   * Ce groupe n'existe que pour un groupe de maille (itemKind()==IK_Cell).
   * Une face est interne si elle connectée à deux mailles de ce groupe.
   */
  ItemGroupImpl* innerFaceGroup();

  /*!
   *  \brief Groupe des faces externes des éléments de ce groupe
   *
   * Ce groupe n'existe que pour un groupe de maille (itemKind()==IK_Cell).
   * Une face est externe si elle n'est connectée qu'à une maille de ce groupe.
   */
  ItemGroupImpl* outerFaceGroup();

  //! AMR
  /*!
   *  \brief Groupe des mailles actives de ce groupe
   *
   * Une maille active est une maille feuille dans l'arbre AMR.
   */
  ItemGroupImpl* activeCellGroup();

  /*!
   *  \brief Groupe des mailles propres actives de ce groupe
   */
  ItemGroupImpl* ownActiveCellGroup();

  /*!
   *  \brief Groupe des mailles actives de ce groupe
   *
   * Une maille active est une maille feuille dans l'arbre AMR.
   */
  ItemGroupImpl* levelCellGroup(const Integer& level);

  /*!
   *  \brief Groupe des mailles propres actives de ce groupe
   */
  ItemGroupImpl* ownLevelCellGroup(const Integer& level);

  /*!
   *  \brief Groupe des faces actives propres au domaine
   *
   * Ce groupe n'existe que pour un groupe de maille (itemKind()==IK_Cell).
   * Une face est interne active si elle connectée à deux mailles actives de ce groupe.
   */
  ItemGroupImpl* activeFaceGroup();

  /*!
   *  \brief Groupe des faces externes actives des éléments de ce groupe
   *
   * Ce groupe n'existe que pour un groupe de maille (itemKind()==IK_Cell).
   * Une face est externe active si elle n'est connectée qu'à une maille de ce groupe et est active.
   */
  ItemGroupImpl* ownActiveFaceGroup();

  /*!
   *  \brief Groupe des faces internes actives des éléments de ce groupe
   *
   * Ce groupe n'existe que pour un groupe de maille (itemKind()==IK_Cell).
   * Une face est interne active si elle connectée à deux mailles actives de ce groupe.
   */
  ItemGroupImpl* innerActiveFaceGroup();

  /*!
   * \brief Groupe des faces externes actives des éléments de ce groupe
   *
   * Ce groupe n'existe que pour un groupe de maille (itemKind()==IK_Cell).
   * Une face est externe active si elle n'est connectée qu'à une maille de ce groupe et est active.
   */
  ItemGroupImpl* outerActiveFaceGroup();

  //! AMR OFF

  //! Vrai si le groupe est local au sous-domaine
  bool isLocalToSubDomain() const;

  //! Positionne le booléen indiquant si le groupe est local au sous-domaine.
  void setLocalToSubDomain(bool v);

  //! Maillage auquel appartient le groupe (0 pour le groupe nul).
  IMesh* mesh() const;

  //! Genre du groupe. Il s'agit du genre de ses éléments.
  eItemKind itemKind() const;

  //! Familly à laquelle appartient le groupe (ou 0 si aucune)
  IItemFamily* itemFamily() const;

  //! Nombre d'entités du groupe
  Integer size() const;

  //! Vrai si le groupe est vide
  bool empty() const;

  //! Supprime les entités du groupe
  void clear();

  //! Groupe parent
  ItemGroup parentGroup();

  /*!
   * \brief Invalide le groupe
   *
   * Opération très violente qui induit une invalidation de toutes les
   * dépendances autant des observers que des sous-groupes construits.
   */
  void invalidate(bool force_recompute);

  /*!
   * \brief  Ajoute les entités de numéros locaux \a items_local_id.
   * \sa ItemGroup::addItems()
   */
  void addItems(Int32ConstArrayView items_local_id,bool check_if_present);

  //! Positionne les entités du groupe à \a items_local_id
  void setItems(Int32ConstArrayView items_local_id);

  //! Positionne les entités du groupe à \a items_local_id en les triant éventuellement.
  void setItems(Int32ConstArrayView items_local_id,bool do_sort);

  //! Supprime les entités \a items_local_id du groupe
  void removeItems(Int32ConstArrayView items_local_id,bool check_if_present);
 
  //! Supprime et ajoute les entités \a removed_local_id et \a added_local_id du groupe
  void removeAddItems(Int32ConstArrayView removed_local_id,
                      Int32ConstArrayView added_local_id,
                      bool check_if_present);

  /*!
   * \brief Supprime du groupe les entités dont le flag isSuppressed() est vrai
   */
  void removeSuppressedItems();

  //! Vérifie que le groupe est valide.
  void checkValid();

  /*! \brief Réactualise le groupe si nécessaire.
   *
   Un groupe doit être réactualisée lorsqu'il est devenu invalide, par exemple
   suite à un appel à invalidate().
   \retval true si le groupe a été réactualisé,
   \retval false sinon.
   */
  bool checkNeedUpdate();

  //! Liste des numéros locaux des entités de ce groupe.
  Int32ConstArrayView itemsLocalId() const;

  /*!
   * \brief Débute une transaction.
   *
   * Une transaction permet d'accèder en écriture à des groupes protégés.
   * L'utilisation de ce mécanisme indique a Arcane que l'utilisateur 
   * a conscience qu'il modifie 'à ses risques' un groupe.
   */ 
  void beginTransaction();

  //! Termine une transaction
  void endTransaction();

  ARCANE_DEPRECATED_REASON("Y2022: Use itemInfoListView() instead")
  //! Liste des entités sur lesquelles s'appuie le groupe
  ItemInternalList itemsInternal() const;

  //! Liste des entités sur lesquelles s'appuie le groupe
  ItemInfoListView itemInfoListView() const;

  /*!
   * \internal
   * \brief Indique à ce groupe qu'il s'agit du groupe de toutes les
   * entités de la famille.
   */
  void setIsAllItems();

  //! Indique si le groupe est celui de toutes les entités
  bool isAllItems() const;

  //! Change les indices des entités du groupe
  void changeIds(Int32ConstArrayView old_to_new_ids);

  //! Applique l'opération \a operation sur les entités du groupe.
  void applyOperation(IItemOperationByBasicType* operation);

  //! Indique si le groupe a structurellement besoin d'une synchro parallèle
  bool needSynchronization() const;

  //! Retourne le temps du groupe. Ce temps est incrémenté après chaque modification.
  Int64 timestamp() const;

  /*!
   * \brief Attache un observer.
   *
   * \param ref référence de l'émetteur de l'observer
   * \param obs Observer
   */
  void attachObserver(const void * ref, IItemGroupObserver * obs);

  /*!
   * \brief Détache un observer.
   *
   * \param ref référence de l'émetteur de l'observer
   */
  void detachObserver(const void * ref);

  /*!
   * \brief Indique si le contenu de ce groupe est observé.
   *
   * Ceci a pour effet d'embrayer des mécanismes de modification incrémentaux.
   * 
   *  Un groupe peut n'être observé que pour sa structure 
   *  par des objets recalculés non incrémentalement.
   */
  bool hasInfoObserver() const;

  //! Définit une fonction de calcul de groupe
  void setComputeFunctor(IFunctor* functor);

  //! Indique si le groupe est calculé
  bool hasComputeFunctor() const;

  /*!
   * \brief Détruit le groupe. Après cet appel, le groupe devient un groupe nul.
   *
   * \warning Cette méthode ne doit être appelé qu'avec une extrème précaution
   * même dans le code bas niveau de Arcane. S'il reste des références sur ce groupe
   * le comportement est indéfini.
   */
  void destroy();

  //! Table des local ids vers une position pour toutes les entités du groupe
  SharedPtrT<GroupIndexTable> localIdToIndex();
 
  //! Synchronizer du groupe
  IVariableSynchronizer* synchronizer();
  
  //! Indique si ce groupe possède un synchroniser
  bool hasSynchronizer();

  /*!
   * \brief Vérifie et retourne si le groupe est trié par uniqueId() croissants.
   */
  bool checkIsSorted() const;

  //! \deprecated Utiliser isContiguousLocalIds() à la place
  bool isContigousLocalIds() const { return isContiguousLocalIds(); }

  //! Indique si les entités du groupe ont des localIds() contigüs.
  bool isContiguousLocalIds() const;

  //! \deprecated Utiliser checkLocalIdsAreContiguous() à la place
  void checkLocalIdsAreContigous() const { return checkLocalIdsAreContiguous(); }

  /*!
   * \brief Vérifie si les entités du groupe ont des localIds() contigüs.
   *
   * Si c'est le cas, alors \a isContiguousLocalIds() retournera \a vrai.
   */
  void checkLocalIdsAreContiguous() const;

  /*!
   * \brief Limite au maximum la mémoire utilisée par le groupe.
   *
   * Si le groupe est un groupe calculé, il est invalidé et toute sa mémoire
   * allouée est libérée.
   *
   * Si le groupe est un groupe créé par l'utilisateur (donc persistant),
   * s'assure que la mémoire consommée est minimale. Normalement %Arcane alloue
   * un peu plus d'éléments que nécessaire pour éviter de faire des réallocations
   * trop souvent.
   */
  void shrinkMemory();

  //! Nombre d'éléments alloués
  Int64 capacity() const;

  //! API interne à Arcane
  ItemGroupImplInternal* _internalApi() const;

 public:

  /*!
   * \internal
   * \brief Liste des numéros locaux des entités de ce groupe.
   * \warning a utiliser avec moult précaution, en général
   * uniquement par le functor de recalcul.
   */
  ARCANE_DEPRECATED_REASON("Y2024: This method is internal to Arcane")
  Int32Array& unguardedItemsLocalId(const bool self_invalidate = true);


 public:

  //! \internal
  static void _buildSharedNull();
  //! \internal
  static void _destroySharedNull();

 private:

  //! Initialisation des sous-groupes par types
  void _initChildrenByType();
  //! Méthode de calcul des sous-groupes par type
  void _computeChildrenByType();
  //! Initialisation des sous-groupes par types
  //void _initChildrenByTypeV2();
  //! Méthode de calcul des sous-groupes par type
  //void _computeChildrenByTypeV2();
  //! Invalidation des sous-groupes
  void _executeExtend(const Int32ConstArrayView * info);
  //! Invalidation des sous-groupes
  void _executeReduce(const Int32ConstArrayView * info);
  //! Invalidation des sous-groupes
  void _executeCompact(const Int32ConstArrayView * info);
  //! Invalidation des sous-groupes
  void _executeReorder(const Int32ConstArrayView * info);
  //! Invalidation des sous-groupes
  void _executeInvalidate();
  //! Mise à jour forcée du flag d'information de restructuration
  void _updateNeedInfoFlag(const bool flag);
  //! Invalidate forcée récursive
  /*! Ne notifie pas les observers. Devra être suivi d'un invalidate() normal */
  void _forceInvalidate(const bool self_invalidate);

  void _checkUpdateSimdPadding();
  //! Notification de SharedReference indiquant qu'il faut détruire l'instance.
  virtual void deleteMe();

 private:

 ItemGroupInternal* m_p = nullptr; //!< Implémentation du groupe

 private:

  //! Supprime les entités \a items_local_id du groupe
  void _removeItems(SmallSpan<const Int32> items_local_id);
  bool _checkNeedUpdateNoPadding();
  bool _checkNeedUpdateWithPadding();
  bool _checkNeedUpdate(bool do_padding);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
