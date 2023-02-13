#pragma once

#include <map>
#include <memory>
#include <vector>

#include <alien/index_manager/IIndexManager.h>
#include <alien/utils/Precomp.h>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Alien
{

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
  explicit BasicIndexManager(Alien::IMessagePassingMng* parallelMng);
  //! Destructeur de la classe
  ~BasicIndexManager() override;

  //! Initialisation
  void init() override;

  //! Indique si la phase de préparation est achevée
  [[nodiscard]] bool isPrepared() const override { return m_state == Prepared; }

  //! Définit le gestionnaire de trace
  void setTraceMng(Alien::ITraceMng* traceMng) override;

  //! Préparation : fixe l'indexation (fin des définitions)
  void prepare() override;

  //! Statistiques d'indexation
  /*! Uniquement valide après \a prepare */
  void stats(Arccore::Integer& globalSize, Arccore::Integer& minLocalIndex,
             Arccore::Integer& localSize) const override;

  //! Retourne la taille globale
  /*! Uniquement valide après \a prepare */
  [[nodiscard]] Arccore::Integer globalSize() const override;

  //! Retourne l'indice minimum local
  /*! Uniquement valide après \a prepare */
  [[nodiscard]] Arccore::Integer minLocalIndex() const override;

  //! Retourne l'indice minimum local
  /*! Uniquement valide après \a prepare */
  [[nodiscard]] Arccore::Integer localSize() const override;

  //! Construction d'un enumerateur sur les \a Entry
  [[nodiscard]] EntryEnumerator enumerateEntry() const override;

  //! Construit une nouvelle entrée scalaire sur un ensemble d'entités abstraites
  ScalarIndexSet buildScalarIndexSet(const Arccore::String& name,
                                     Arccore::IntegerConstArrayView localIds,
                                     const IIndexManager::IAbstractFamily& family) override;

  //! Construit une nouvelle entrée scalaire sur l'ensemble des entités d'une familles
  //! abstraite
  ScalarIndexSet buildScalarIndexSet(
  const Arccore::String& name, const IIndexManager::IAbstractFamily& family) override;

  //! Construit une nouvelle entrée vectoriellesur un ensemble d'entités abstraites
  /*! L'implémentation actuelle considére le multi-scalaire comme du vectoriel */
  VectorIndexSet buildVectorIndexSet(const Arccore::String& name,
                                     Arccore::IntegerConstArrayView localIds,
                                     const IIndexManager::IAbstractFamily& family,
                                     Arccore::Integer n) override;

  //! Construit une nouvelle entrée scalaire sur l'ensemble des entités d'une familles
  //! abstraite
  /*! L'implémentation actuelle considére le multi-scalaire comme du vectoriel */
  VectorIndexSet buildVectorIndexSet(const Arccore::String& name,
                                     const IIndexManager::IAbstractFamily& family, Arccore::Integer n) override;

  //! Fournit une table de translation indexé par les items
  [[nodiscard]] Arccore::UniqueArray<Arccore::Integer> getIndexes(
  const ScalarIndexSet& entry) const override;

  //! Fournit une table de translation vectorielle indexé par les items puis par les
  //! entrées
  [[nodiscard]] Arccore::UniqueArray2<Arccore::Integer> getIndexes(
  const VectorIndexSet& entries) const override;

  //! Donne le gestionnaire parallèle ayant servi à l'indexation

  [[nodiscard]] Arccore::MessagePassing::IMessagePassingMng* parallelMng() const override
  {
    return m_parallel_mng;
  }

  //! define null index : default = -1, if true null_index = max_index+1
  void setMaxNullIndexOpt(bool flag) override { m_max_null_index_opt = flag; }

  [[nodiscard]] Arccore::Integer nullIndex() const override
  {
    ALIEN_ASSERT((m_state == Prepared), ("nullIndex is valid only in Prepared state"));
    if (m_max_null_index_opt)
      return m_global_entry_offset + m_local_entry_count;
    else
      return -1;
  }

 public:
  void keepAlive(const IAbstractFamily* family) override;

 private:
  Arccore::MessagePassing::IMessagePassingMng* m_parallel_mng = nullptr;
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
                       [[maybe_unused]] Arccore::Integer creation_index, Arccore::Integer owner)
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

  [[nodiscard]] Entry getEntry(const Arccore::String& name) const override;

 protected:
  //! \internal Structure interne de communication dans prepare()
  struct EntrySendRequest;
  struct EntryRecvRequest;

 protected: // Méthodes protègés en attendant une explicitation du besoin
 private:
  Entry buildEntry(const Arccore::String& name, const IAbstractFamily* itemFamily,
                   Arccore::Integer kind);
  void defineIndex(const Entry& entry, Arccore::IntegerConstArrayView localIds);
  void parallel_prepare(EntryIndexMap& entry_index);
  void sequential_prepare(EntryIndexMap& entry_index);
  [[nodiscard]] inline bool isOwn(const IAbstractFamily::Item& item) const
  {
    return item.owner() == m_local_owner;
  }
  [[nodiscard]] inline bool isOwn(const InternalEntryIndex& i) const
  {
    return i.m_owner == m_local_owner;
  }
  void reserveEntries(const EntryIndexMap& entry_index);
  Arccore::Integer addNewAbstractFamily(const IAbstractFamily* family);

  //! Init datastructure with a non virtual function
  void init_mine();
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
} // namespace Alien

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
