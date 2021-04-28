#pragma once

#include <alien/index_manager/IIndexManager.h>
#include <alien/utils/Precomp.h>
#include <map>
#include <memory>
#include <vector>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Alien {

#ifdef USE_ARCANE_PARALLELMNG
namespace ArcaneParallelTest {
#endif
  /*---------------------------------------------------------------------------*/
  /*---------------------------------------------------------------------------*/

  /*---------------------------------------------------------------------------*/
  /*---------------------------------------------------------------------------*/

  /*! \todo Il est possible d'optimiser les accès vectorielles en raprochant les
   *  structures internes des interfaces critiques
   *  (en particulier getIndex vectoriel)
   */
  class ALIEN_EXPORT BasicIndexManager : public IIndexManager
  {
   public:
   public:
    //! Constructeur de la classe
#ifdef USE_ARCANE_PARALLELMNG
    BasicIndexManager(Arcane::IParallelMng* parallelMng);
#else
  BasicIndexManager(Alien::IMessagePassingMng* parallelMng);
#endif
    //! Destructeur de la classe
    virtual ~BasicIndexManager();

    //! Initialisation
    void init();

    //! Indique si la phase de préparation est achevée
    bool isPrepared() const { return m_state == Prepared; }

    //! Définit le gestionnaire de trace
    void setTraceMng(Alien::ITraceMng* traceMng);

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

    //! Construit une nouvelle entrée scalaire sur un ensemble d'entités abstraites
    ScalarIndexSet buildScalarIndexSet(const Arccore::String name,
        const Arccore::IntegerConstArrayView localIds, const IAbstractFamily& family);

    //! Construit une nouvelle entrée scalaire sur l'ensemble des entités d'une familles
    //! abstraite
    ScalarIndexSet buildScalarIndexSet(
        const Arccore::String name, const IAbstractFamily& family);

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

    //! Fournit une table de translation indexé par les items
    Arccore::UniqueArray<Arccore::Integer> getIndexes(const ScalarIndexSet& entry) const;

    //! Fournit une table de translation vectorielle indexé par les items puis par les
    //! entrées
    Arccore::UniqueArray2<Arccore::Integer> getIndexes(
        const VectorIndexSet& entries) const;

    //! Donne le gestionnaire parallèle ayant servi à l'indexation

#ifdef USE_ARCANE_PARALLELMNG
    Arcane::IParallelMng* parallelMng() const { return m_parallel_mng; }
#else
  Alien::IMessagePassingMng* parallelMng() const { return m_parallel_mng; }
#endif
    //! define null index : default = -1, if true null_index = max_index+1
    void setMaxNullIndexOpt(bool flag) { m_max_null_index_opt = flag; }

    Arccore::Integer nullIndex() const
    {
      ALIEN_ASSERT((m_state == Prepared), ("nullIndex is valid only in Prepared state"));
      if (m_max_null_index_opt)
        return m_global_entry_offset + m_local_entry_count;
      else
        return -1;
    }

   public:
    void keepAlive(const IAbstractFamily* family);

   private:
#ifdef USE_ARCANE_PARALLELMNG
    Arcane::IParallelMng* m_parallel_mng;
#else
  Alien::IMessagePassingMng* m_parallel_mng = nullptr;
#endif
    Arccore::Integer m_local_owner = 0; //!< Identifiant du 'propriétaire' courant

    enum State
    {
      Undef,
      Initialized,
      Prepared
    } m_state;

    Arccore::ITraceMng* m_trace = nullptr;

    Arccore::Integer m_local_entry_count = 0;
    Arccore::Integer m_global_entry_count = 0;
    Arccore::Integer m_global_entry_offset = 0;
    Arccore::Integer m_local_removed_entry_count = 0;
    Arccore::Integer m_global_removed_entry_count = 0;

    bool m_max_null_index_opt = false;

    class MyEntryImpl;
    class MyEntryEnumeratorImpl;

    struct InternalEntryIndex
    {
      InternalEntryIndex(MyEntryImpl* e, Arccore::Integer lid, Arccore::Integer kind,
          Arccore::Int64 uid, Arccore::Integer index,
          Arccore::Integer creation_index ALIEN_UNUSED_PARAM, Arccore::Integer owner)
      : m_entry(e)
      , m_uid(uid)
      , m_localid(lid)
      , m_kind(kind)
      , m_index(index)
      // , m_creation_index(creation_index)
      , m_owner(owner)
      {}
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
  };

  /*---------------------------------------------------------------------------*/
  /*---------------------------------------------------------------------------*/

#ifdef USE_ARCANE_PARALLELMNG
}
#endif
} // namespace Alien

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
