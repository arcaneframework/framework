// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* BasicIndexManager                                         (C) 2000-2024   */
/*                                                                           */
/* Basic indexing between algebra and mesh worlds. Depends on Arcane         */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
#ifndef ALIEN_INDEXMANAGER_BASICINDEXMANAGER_H
#define ALIEN_INDEXMANAGER_BASICINDEXMANAGER_H

#include <map>
#include <memory>
#include <vector>
#include "alien/AlienArcaneToolsPrecomp.h"
#include "alien/arcane_tools/IIndexManager.h"
#include "alien/arcane_tools/indexManager/IItemIndexManager.h"

#include <arcane/ItemInternalVectorView.h>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Alien {

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace ArcaneTools {

  /*---------------------------------------------------------------------------*/
  /*---------------------------------------------------------------------------*/

  /*! \todo Il est possible d'optimiser les accès vectorielles en raprochant les
   *  structures internes des interfaces critiques
   *  (en particulier getIndex vectoriel)
   */
  class ALIEN_ARCANE_TOOLS_EXPORT BasicIndexManager : public IItemIndexManager
  {
   public:
    class ALIEN_ARCANE_TOOLS_EXPORT ItemAbstractFamily : public IAbstractFamily
    {
     public:
      ItemAbstractFamily(const Arcane::IItemFamily* family);

      virtual ~ItemAbstractFamily() {}

     public:
      IIndexManager::IAbstractFamily* clone() const;

     public:
      Arccore::Int32 maxLocalId() const;
      void uniqueIdToLocalId(
          Arccore::Int32ArrayView localIds, Arccore::Int64ConstArrayView uniqueIds) const;
      Item item(Arccore::Int32 localId) const;
      Arccore::SharedArray<Arccore::Integer> owners(
          Arccore::Int32ConstArrayView localIds) const;
      Arccore::SharedArray<Arccore::Int64> uids(
          Arccore::Int32ConstArrayView localIds) const;
      Arccore::SharedArray<Arccore::Int32> allLocalIds() const;

     private:
      const Arcane::IItemFamily* m_family;
      const Arcane::ItemInternalArrayView m_item_internals;
    };

   public:
    //! Constructeur de la classe
    BasicIndexManager(Arcane::IParallelMng* parallelMng);

    //! Destructeur de la classe
    virtual ~BasicIndexManager();

    //! Initialisation
    void init();

    //! Indique si la phase de préparation est achevée
    bool isPrepared() const { return m_state == Prepared; }

    //! Définit le gestionnaire de trace
    void setTraceMng(Arcane::ITraceMng* traceMng);

    //! Préparation : fixe l'indexation (fin des définitions)
    void prepare();

    //! Statistiques d'indexation
    /*! Uniquement valide après \a prepare */
    void stats(Arccore::Integer& globalSize, Arccore::Integer& minLocalIndex,
        Arccore::Integer& localSize) const;

    //! Retourne la taille globale
    /*! Uniquement valide après \a prepare */
    Arccore::Integer globalSize() const;

    //! Retourne l'indice minimum local
    /*! Uniquement valide après \a prepare */
    Arccore::Integer minLocalIndex() const;

    //! Retourne l'indice minimum local
    /*! Uniquement valide après \a prepare */
    Arccore::Integer localSize() const;

    //! Construction d'un enumerateur sur les \a Entry
    EntryEnumerator enumerateEntry() const;

    //! Construit une nouvelle entrée scalaire sur des items du maillage
    ScalarIndexSet buildScalarIndexSet(
        const Arccore::String name, Arcane::IItemFamily* item_family);
    void defineIndex(ScalarIndexSet& set, const Arcane::ItemGroup& itemGroup);

    ScalarIndexSet buildScalarIndexSet(
        const Arccore::String name, const Arcane::ItemGroup& itemGroup);

    //! Construit une nouvelle entrée scalaire sur un ensemble d'entités abstraites
    ScalarIndexSet buildScalarIndexSet(const Arccore::String name,
        const Arccore::IntegerConstArrayView localIds, const IAbstractFamily& family);

    //! Construit une nouvelle entrée scalaire sur l'ensemble des entités d'une familles
    //! abstraite
    ScalarIndexSet buildScalarIndexSet(
        const Arccore::String name, const IAbstractFamily& family);

    //! Construit une nouvelle entrée vectorielle sur des items du maillage
    /*! L'implémentation actuelle considére le multi-scalaire comme du vectoriel */
    VectorIndexSet buildVectorIndexSet(const Arccore::String name,
        const Arcane::ItemGroup& itemGroup, const Arccore::Integer n);

    //! Construit une nouvelle entrée vectoriellesur un ensemble d'entités abstraites
    /*! L'implémentation actuelle considére le multi-scalaire comme du vectoriel */
    VectorIndexSet buildVectorIndexSet(const Arccore::String name,
        const Arccore::IntegerConstArrayView localIds, const IAbstractFamily& family,
        const Arccore::Integer n);

    //! Construit une nouvelle entrée scalaire sur l'ensemble des entités d'une familles
    //! abstraite
    /*! L'implémentation actuelle considére le multi-scalaire comme du vectoriel */
    VectorIndexSet buildVectorIndexSet(const Arccore::String name,
        const IAbstractFamily& family, const Arccore::Integer n);

    //! Demande de dé-indexation d'une partie d'une entrée
    /*! Utilisable uniquement avant prepare */
    void removeIndex(const ScalarIndexSet& entry, const Arcane::ItemGroup& itemGroup);

    void removeIndex(const VectorIndexSet& entry, const Arccore::Integer component, const Arcane::ItemGroup& itemGroup)
    {
      removeIndex(entry[component],itemGroup) ;
    }

    Arccore::Integer getIndex(const Entry& entry, const Arcane::Item& item) const;

    //! Consultation vectorielle d'indexation d'une entrée (après prepare)
    void getIndex(const ScalarIndexSet& entry, const Arcane::ItemVectorView& items,
        Arccore::ArrayView<Arccore::Integer> indexes) const;

    //! Fournit une table de translation indexé par les items
    Arccore::UniqueArray<Arccore::Integer> getIndexes(const ScalarIndexSet& entry) const;

    //! Fournit une table de translation vectorielle indexé par les items puis par les
    //! entrées
    Arccore::UniqueArray2<Arccore::Integer> getIndexes(
        const VectorIndexSet& entries) const;

    //! Donne le gestionnaire parallèle ayant servi à l'indexation
    Alien::IMessagePassingMng* parallelMng() const { return m_parallel_mng; }

    //! define null index : default = -1, if true null_index = max_index+1
    void setMaxNullIndexOpt(bool flag) { m_max_null_index_opt = flag; }

    Arccore::Integer nullIndex() const
    {
      ARCANE_ASSERT((m_state == Prepared), ("nullIndex is valid only in Prepared state"));
      if (m_max_null_index_opt)
        return m_global_entry_offset + m_local_entry_count;
      else
        return -1;
    }

   public:
    void keepAlive(const IAbstractFamily* family);

   private:
    Alien::IMessagePassingMng* m_parallel_mng = nullptr;
    Arccore::ITraceMng*        m_trace        = nullptr;


    enum State
    {
      Undef,
      Initialized,
      Prepared
    } m_state;


    Arccore::Integer m_local_owner                = 0; //!< Identifiant du 'propriétaire' courant
    Arccore::Integer m_local_entry_count          = 0;
    Arccore::Integer m_global_entry_count         = 0;
    Arccore::Integer m_global_entry_offset        = 0;
    Arccore::Integer m_local_removed_entry_count  = 0;
    Arccore::Integer m_global_removed_entry_count = 0;

    bool m_max_null_index_opt                     = false;

    class MyEntryImpl;
    class MyEntryEnumeratorImpl;

    struct InternalEntryIndex
    {
      InternalEntryIndex(MyEntryImpl* e, Arccore::Integer lid, Arccore::Integer kind,
                         Arccore::Int64 uid, Arccore::Integer index,
                         [[maybe_unused]] Arccore::Integer creation_index, Arccore::Integer owner)
      : m_entry(e)
      , m_uid(uid)
      , m_localid(lid)
      , m_kind(kind)
      , m_index(index)
      // , m_creation_index(creation_index)
      , m_owner(owner)
      {
      }
      MyEntryImpl* m_entry;
      Arccore::Int64 m_uid;
      Arccore::Integer m_localid, m_kind, m_index;
      // Integer m_creation_index;
      Arccore::Integer m_owner;
      bool operator==(const InternalEntryIndex& m) const
      {
        return m.m_entry == m_entry && m.m_localid == m_localid;
      }
    };

    typedef std::vector<InternalEntryIndex> EntryIndexMap;

    struct EntryIndexComparator
    {
      inline bool operator()(
          const InternalEntryIndex& a, const InternalEntryIndex& b) const;
    };

    //! Table des Entry connues localement
    typedef std::map<Arccore::String, MyEntryImpl*> EntrySet;
    EntrySet m_entry_set;

    //! Index de creation des entrées
    Arccore::Integer m_creation_index = 0;

    //! Famille des familles abstraites associées aux familles du maillage

    Arccore::Integer m_abstract_family_base_kind = 0;
    std::map<Arcane::IMesh*, Arccore::Integer>
        m_item_family_meshes; //!< Table des maillages connues (pour grouper en maillage)
    // TODO: why a shared_ptr here ?
    std::map<Arccore::Integer, std::shared_ptr<IAbstractFamily>>
        m_abstract_families; //!< Table des IAbstractFamily ici gérées
    std::map<const IAbstractFamily*, Arccore::Integer>
        m_abstract_family_to_kind_map; //!< Permet la gestion de la survie des
    //!< IAbstractFamily extérieures

    //! Retourne l'entrée associée à un nom

    Entry getEntry(const Arccore::String name) const;

   protected:
    //! \internal Structure interne de communication dans prepare()
    struct EntrySendRequest;
    struct EntryRecvRequest;

   protected: // Méthodes protègés en attendant une explicitation du besoin
   private:
    Entry buildEntry(const Arccore::String name, const IAbstractFamily* itemFamily,
        const Arccore::Integer kind);
    void defineIndex(const Entry& entry, const Arccore::IntegerConstArrayView localIds);
    void parallel_prepare(EntryIndexMap& entry_index);
    void sequential_prepare(EntryIndexMap& entry_index);
    inline bool isOwn(const IAbstractFamily::Item& item) const
    {
      return item.owner() == m_local_owner;
    }
    inline bool isOwn(const InternalEntryIndex& i) const
    {
      return i.m_owner == m_local_owner;
    }
    void reserveEntries(const EntryIndexMap& entry_index);
    Arccore::Integer addNewAbstractFamily(const IAbstractFamily* family);
    Arccore::Integer kindFromItemFamily(const Arcane::IItemFamily* family);
  };

  /*---------------------------------------------------------------------------*/
  /*---------------------------------------------------------------------------*/

} // namespace Alien

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif /* ARCGEOSIM_ARCGEOSIM_NUMERICS_LINEARALGEBRA2_INDEXMANAGER_BASICINDEXMANAGER_H   \
          */
