#ifndef ALIEN_INDEX_MANAGER_IMPL_H
#define ALIEN_INDEX_MANAGER_IMPL_H

/*---------------------------------------------------------------------------*/
#include <map>

#include "alien/arcane_tools/IIndexManager.h"

#include <arcane/IParallelMng.h>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Alien {

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace ArcaneTools {

  /*---------------------------------------------------------------------------*/
  /*---------------------------------------------------------------------------*/

  /**
   *
   * Voir \ref secLinearSolver "Description des services de solveurs
   * linéaires" pour plus de détails
   *
   * \todo Pour gérer les inconnues vectoriel, mettre un argument additionnel qui
   * est le sous-indice (vectoriel) avec une valeur par défaut de 0 (ou
   * bien un proto différent)
   */
  class IndexManagerImpl : public IIndexManager
  {
   public:
   public:
    //! Constructeur par défaut
    IndexManagerImpl(Arccore::Integer global_size, Arccore::Integer nproc = 1,
        Arccore::Integer myrank = 0, Arccore::ITraceMng* trace_mng = nullptr)
    : m_global_size(global_size)
    , m_nproc(nproc)
    , m_myrank(myrank)
    , m_trace_mng(trace_mng)
    {
      if (m_nproc > 1)
        _computePartition();
    }

    IndexManagerImpl(Arccore::Integer global_size, Arcane::IParallelMng* parallel_mng,
        Arccore::ITraceMng* trace_mng = NULL)
    : m_global_size(global_size)
    , m_parallel_mng(parallel_mng->messagePassingMng())
    , m_trace_mng(trace_mng)
    {
      m_nproc = m_parallel_mng ? m_parallel_mng->commSize() : 1;
      m_myrank = m_parallel_mng ? m_parallel_mng->commRank() : 0;
      if (m_nproc > 1) {
        _computePartition();
      } else {
        m_offset.resize(m_nproc + 1);
        m_offset[0] = 0;
        m_offset[1] = m_offset[0] + m_global_size;
        m_local_offset = m_offset[m_myrank];
        m_local_size = m_global_size;
        m_ghost_size = 0;
      }
    }

    IndexManagerImpl(Arccore::Integer global_size, Arccore::Integer local_size,
        Arccore::Integer ghost_size, Arccore::ConstArrayView<Arccore::Int64> local_uids,
        Arccore::ConstArrayView<Arccore::Int64> ghost_uids,
        Arccore::ConstArrayView<Arccore::Integer> ghost_owners,
        Arcane::IParallelMng* parallel_mng, Arccore::ITraceMng* trace_mng = nullptr)
    : m_global_size(global_size)
    , m_local_size(local_size)
    , m_ghost_size(ghost_size)
    , m_parallel_mng(parallel_mng->messagePassingMng())
    , m_trace_mng(trace_mng)
    {
      m_nproc = m_parallel_mng ? m_parallel_mng->commSize() : 1;
      m_myrank = m_parallel_mng ? m_parallel_mng->commRank() : 0;
      ALIEN_ASSERT((m_local_size <= local_uids.size()), ("Incompatible local size"));
      m_local_uids.resize(local_size);
      m_local_uids.copy(local_uids);

      ALIEN_ASSERT((m_ghost_size <= ghost_uids.size()), ("Incompatible local size"));
      m_ghost_uids.resize(ghost_size);
      m_ghost_uids.copy(ghost_uids);
      m_ghost_owners.resize(ghost_size);
      m_ghost_owners.copy(ghost_owners);
    }

    IndexManagerImpl(Arccore::Integer global_size, Arccore::Integer local_size,
        Arccore::Integer ghost_size, Arccore::ConstArrayView<Arccore::Integer> local_uids,
        Arccore::ConstArrayView<Arccore::Integer> ghost_uids,
        Arccore::ConstArrayView<Arccore::Integer> ghost_owners,
        Arcane::IParallelMng* parallel_mng, Arccore::ITraceMng* trace_mng = nullptr)
    : m_global_size(global_size)
    , m_local_size(local_size)
    , m_ghost_size(ghost_size)
    , m_parallel_mng(parallel_mng->messagePassingMng())
    , m_trace_mng(trace_mng)
    {
      m_nproc = m_parallel_mng ? m_parallel_mng->commSize() : 1;
      m_myrank = m_parallel_mng ? m_parallel_mng->commRank() : 0;
      // ALIEN_ASSERT( (m_local_size<=local_uids.size()),("Incompatible local size")) ;
      m_local_uids.resize(local_size);
      for (Arccore::Integer i = 0; i < local_size; ++i)
        m_local_uids[i] = local_uids[i];
      if (m_nproc > 1) {
        // ALIEN_ASSERT( (m_ghost_size<=ghost_uids.size()),("Incompatible local size")) ;
        m_ghost_uids.resize(ghost_size);
        for (Arccore::Integer i = 0; i < ghost_size; ++i)
          m_ghost_uids[i] = ghost_uids[i];

        m_ghost_owners.resize(ghost_size);
        // Alien::copy(m_ghost_owners,ghost_owners) ;
        for (Arccore::Integer i = 0; i < ghost_size; ++i)
          m_ghost_owners[i] = ghost_owners[i];
      }
    }

    //! Destructeur
    virtual ~IndexManagerImpl() { ; }

    //! Indique si la phase de préparation est achevée
    virtual bool isPrepared() const { return true; }

    //! Initialisation les structures
    /*! Implicitement appelé par le constructeur */
    virtual void init() {}

    //! Préparation : fixe l'indexation (fin des définitions)
    virtual void prepare(){};

    //! Définit le gestionnaire de trace
    virtual void setTraceMng(Arccore::ITraceMng* traceMng) { m_trace_mng = traceMng; }

    //! Statistiques d'indexation
    /*! Uniquement valide après \a prepare */
    virtual void stats(Arccore::Integer& globalSize, Arccore::Integer& minLocalIndex,
        Arccore::Integer& localSize) const
    {
      globalSize = m_global_size;
      localSize = m_local_size;
      minLocalIndex = m_offset[m_myrank];
    }

    //! Retourne la taille globale
    /*! Uniquement valide après \a prepare */
    virtual Arccore::Integer globalSize() const { return m_global_size; }

    //! Retourne l'indice minimum local
    /*! Uniquement valide après \a prepare */
    virtual Arccore::Integer minLocalIndex() const { return m_offset[m_myrank]; }

    //! Retourne l'indice minimum local
    /*! Uniquement valide après \a prepare */
    virtual Arccore::Integer localSize() const { return m_local_size; }

    virtual Arccore::Integer ghostSize() const { return m_ghost_size; }

    Arccore::UniqueArray<Arccore::Int64> const& localUids() const { return m_local_uids; }

    Arccore::UniqueArray<Arccore::Int64> const ghostUids() const { return m_ghost_uids; }

    Arccore::UniqueArray<Arccore::Int64> const& ghostGids() const { return m_ghost_gids; }

    Arccore::UniqueArray<Arccore::Integer> const& ghostOwners() const
    {
      return m_ghost_owners;
    }

    std::map<Arccore::Int64, Arccore::Integer> const& localLids() const
    {
      return m_local_lids;
    }

    std::map<Arccore::Int64, Arccore::Integer> const& ghostLids() { return m_ghost_lids; }

    void computeIndexes(Arccore::Integer lidMax ALIEN_UNUSED_PARAM,
        Arccore::ArrayView<Arccore::Integer> local_lids ALIEN_UNUSED_PARAM,
        Arccore::ArrayView<Arccore::Integer> guost_lids ALIEN_UNUSED_PARAM,
        Arccore::UniqueArray<Arccore::Integer>& indexes ALIEN_UNUSED_PARAM){};

    //! Construction d'un enumerateur sur les \a Entry
    virtual EntryEnumerator enumerateEntry() const
    {
      return EntryEnumerator(static_cast<EntryEnumeratorImpl*>(nullptr));
    }

    //! Retourne l'entrée associée à un nom
    virtual Entry getEntry(const Arccore::String name ALIEN_UNUSED_PARAM) const
    {
      return Entry();
    }

    typedef Entry ScalarIndexSet;
    typedef Arccore::UniqueArray<ScalarIndexSet> VectorIndexSet;

    //! Construit une nouvelle entrée scalaire sur des items du maillage
    // virtual ScalarIndexSet buildScalarIndexSet(const Arccore::String
    // name,Arcane::IItemFamily * item_family) = 0;// virtual void
    // defineIndex(ScalarIndexSet& set , const Arcane::ItemGroup & itemGroup) = 0; virtual
    // ScalarIndexSet buildScalarIndexSet(const Arccore::String name, const
    // Arcane::ItemGroup & itemGroup) = 0;

    //! Construit une nouvelle entrée scalaire sur un ensemble d'entités abstraites
    virtual ScalarIndexSet buildScalarIndexSet(
        const Arccore::String name ALIEN_UNUSED_PARAM,
        const Arccore::ConstArrayView<Arccore::Integer> localIds ALIEN_UNUSED_PARAM,
        const IAbstractFamily& family ALIEN_UNUSED_PARAM)
    {
      return ScalarIndexSet();
    }

    //! Construit une nouvelle entrée scalaire sur l'ensemble des entités d'une familles
    //! abstraite
    virtual ScalarIndexSet buildScalarIndexSet(
        const Arccore::String name ALIEN_UNUSED_PARAM,
        const IAbstractFamily& family ALIEN_UNUSED_PARAM)
    {
      return ScalarIndexSet();
    }

    //! Construit une nouvelle entrée vectorielle sur des items du maillage
    /*! L'implémentation actuelle considére le multi-scalaire comme du vectoriel */
    // virtual VectorIndexSet buildVectorIndexSet(const Arccore::String name, const
    // Arcane::ItemGroup & itemGroup, const Arccore::Integer n) = 0;
    //! Construit une nouvelle entrée vectoriellesur un ensemble d'entités abstraites
    /*! L'implémentation actuelle considére le multi-scalaire comme du vectoriel */
    virtual VectorIndexSet buildVectorIndexSet(
        const Arccore::String name ALIEN_UNUSED_PARAM,
        const Arccore::ConstArrayView<Arccore::Integer> localIds ALIEN_UNUSED_PARAM,
        const IAbstractFamily& family ALIEN_UNUSED_PARAM,
        const Arccore::Integer n ALIEN_UNUSED_PARAM)
    {
      return VectorIndexSet();
    }

    //! Construit une nouvelle entrée scalaire sur l'ensemble des entités d'une familles
    //! abstraite
    /*! L'implémentation actuelle considére le multi-scalaire comme du vectoriel */
    virtual VectorIndexSet buildVectorIndexSet(
        const Arccore::String name ALIEN_UNUSED_PARAM,
        const IAbstractFamily& family ALIEN_UNUSED_PARAM,
        const Arccore::Integer n ALIEN_UNUSED_PARAM)
    {
      return VectorIndexSet();
    }

    //! Demande de dé-indexation d'une partie d'une entrée
    /*! Utilisable uniquement avant prepare */
    virtual void removeIndex(const ScalarIndexSet & entry,
                             const Arcane::ItemGroup & itemGroup)
    {

    }

    virtual void removeIndex(const VectorIndexSet & entry, const Arccore::Integer icomponent,
                             const Arcane::ItemGroup & itemGroup)
    {

    }

    //! Fournit une table de translation indexé par les items
    virtual Arccore::UniqueArray<Arccore::Integer> getIndexes(
        const ScalarIndexSet& entry ALIEN_UNUSED_PARAM) const
    {
      return Arccore::UniqueArray<Arccore::Integer>();
    }

    //! Fournit une table de translation indexé par les items
    virtual Arccore::UniqueArray2<Arccore::Integer> getIndexes(
        const VectorIndexSet& entry ALIEN_UNUSED_PARAM) const
    {
      return Arccore::UniqueArray2<Arccore::Integer>();
    }

    //! Donne le gestionnaire parallèle ayant servi à l'indexation
    virtual Alien::IMessagePassingMng* parallelMng() const { return m_parallel_mng; }

    //! define null index : default == -1, if true nullIndex() == max index of current
    //! indexation
    virtual void setMaxNullIndexOpt(bool flag ALIEN_UNUSED_PARAM) {}

    //! return value of null index
    virtual Arccore::Integer nullIndex() const { return 0; }

   public:
    //! Permet de gérer la mort d'une famille associée à l'index-manager
    /*! Méthode de bas niveau pour les implémentationsde IAbstractFamily,
     *  usuellement dans le desctructeur des implémentations extérieures de
     * IAbstractFamily
     */
    virtual void keepAlive(const IAbstractFamily* family ALIEN_UNUSED_PARAM) {}

   private:
    void _computePartition()
    {
      Arccore::Integer local_size = m_global_size / m_nproc;
      Arccore::Integer r = m_global_size % m_nproc;
      if (m_myrank < r)
        m_local_size = local_size + 1;
      else
        m_local_size = local_size;
      m_offset.resize(m_nproc + 1);
      m_offset[0] = 0;
      for (Arccore::Integer i = 0; i < r; ++i)
        m_offset[i + 1] = m_offset[i] + local_size + 1;
      for (Arccore::Integer i = r; i < m_nproc; ++i)
        m_offset[i + 1] = m_offset[i] + local_size;
      m_local_offset = m_offset[m_myrank];
    }
    Arccore::Integer m_global_size = 0;
    Arccore::Integer m_local_size = 0;
    Arccore::Integer m_ghost_size = 0;
    Arccore::UniqueArray<Arccore::Int64> m_local_uids;
    Arccore::UniqueArray<Arccore::Int64> m_ghost_uids;
    Arccore::UniqueArray<Arccore::Int64> m_ghost_gids;
    Arccore::UniqueArray<Arccore::Integer> m_ghost_owners;
    std::map<Arccore::Int64, Arccore::Integer> m_local_lids;
    std::map<Arccore::Int64, Arccore::Integer> m_ghost_lids;

    Arccore::UniqueArray<Arccore::Int64> m_offset;
    Arccore::Int64 m_local_offset = 0;
    Arccore::Integer m_nproc = 1;
    Arccore::Integer m_myrank = 0;
    Alien::IMessagePassingMng* m_parallel_mng = nullptr;
    Arccore::ITraceMng* m_trace_mng = nullptr;
  };

  /*---------------------------------------------------------------------------*/
  /*---------------------------------------------------------------------------*/

} // namespace Alien

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif /* ARCGEOSIM_ARCGEOSIM_NUMERICS_LINEARALGEBRA2_IINDEX_MANAGER_H */
