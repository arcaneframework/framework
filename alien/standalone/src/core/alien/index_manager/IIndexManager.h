#pragma once

#include <memory> // for std::shared_ptr.
#include <vector>

#include <alien/utils/Precomp.h>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Alien
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/**
 * Voir \ref secLinearSolver "Description des services de solveurs
 * linéaires" pour plus de détails
 *
 * \todo Pour gérer les inconnues vectoriel, mettre un argument additionnel qui
 * est le sous-indice (vectoriel) avec une valeur par défaut de 0 (ou
 * bien un proto différent)
 */
class IIndexManager
{
 public:
  //! Interface des familles abstraites pour l'indexation de items
  class IAbstractFamily
  {
   public:
    class Item
    {
     public:
      Item(Arccore::Int64 uniqueId, Arccore::Integer owner)
      : m_unique_id(uniqueId)
      , m_owner(owner)
      {}

     public:
      Arccore::Int64 uniqueId() const { return m_unique_id; }
      Arccore::Integer owner() const { return m_owner; }

     private:
      Arccore::Int64 m_unique_id;
      Arccore::Integer m_owner;
    };

   public:
    virtual ~IAbstractFamily() = default;

   public:
    //! Construit un clone de cet objet
    virtual IAbstractFamily* clone() const = 0;

   public:
    //! Identifiant maximal des localIds pour cette famille
    virtual Arccore::Int32 maxLocalId() const = 0;
    //! Convertit des uniqueIds en localIds. Erreur fatale si un item n'est pas retrouvé
    virtual void uniqueIdToLocalId(Arccore::ArrayView<Arccore::Int32> localIds,
                                   Arccore::ConstArrayView<Arccore::Int64> uniqueIds) const = 0;
    //! Retourne un objet Item à partir de son localId
    virtual Item item(Arccore::Int32 localId) const = 0;
    //! Retourne l'ensemble des owners (propriétaires) d'un ensemble d'item décrits par
    //! leur localIds
    virtual Arccore::SharedArray<Arccore::Integer> owners(
    Arccore::ConstArrayView<Arccore::Int32> localIds) const = 0;
    //! Retourne l'ensemble des uniqueIds d'un ensemble d'item décrits par leur localIds
    virtual Arccore::SharedArray<Arccore::Int64> uids(
    Arccore::ConstArrayView<Arccore::Int32> localIds) const = 0;
    //! Retourne l'ensemble des identifiants locaux de la famille
    virtual Arccore::SharedArray<Arccore::Int32> allLocalIds() const = 0;
  };

 protected:
  //! Interface d'implémentation de \a Entry
  class EntryImpl
  {
   public:
    //! Destructeur
    virtual ~EntryImpl() = default;
    //! Retourne la liste des Index de l'Entry
    virtual Arccore::ConstArrayView<Arccore::Integer> getOwnIndexes() const = 0;
    //! Retourne la liste des Index de l'Entry (own + ghost)
    virtual Arccore::ConstArrayView<Arccore::Integer> getAllIndexes() const = 0;
    //! Retourne la liste des Items de l'Entry
    virtual Arccore::ConstArrayView<Arccore::Integer> getOwnLocalIds() const = 0;
    //! Retourne la liste des Items de l'Entry (own + ghost)
    virtual Arccore::ConstArrayView<Arccore::Integer> getAllLocalIds() const = 0;
    //! Retourne le nom de l'entrée
    virtual Arccore::String getName() const = 0;
    //! Retourne le type de support de l'Entry
    virtual Arccore::Integer getKind() const = 0;
    //! Retourne la famille abstraite de l'Entry
    virtual const IAbstractFamily& getFamily() const = 0;
    //! Ajout d'un tag
    virtual void addTag(
    const Arccore::String& tagname, const Arccore::String& tagvalue) = 0;
    //! Suppression d'un tag
    virtual void removeTag(const Arccore::String& tagname) = 0;
    //! Test d'existance d'un tag
    virtual bool hasTag(const Arccore::String& tagname) = 0;
    //! Lecture d'un tag
    virtual Arccore::String tagValue(const Arccore::String& tagname) = 0;
    //! Référentiel du manager associé
    virtual IIndexManager* manager() const = 0;
  };

 public:
  //! Classe de représentation des Entry
  /*! Cette classe est un proxy; sa copie est donc peu couteuse et son
   *  implémentation variable suivant le contexte
   */
  class Entry
  {
   protected:
    //! Implémentation de ce type d'entrée
    EntryImpl* m_impl;

   public:
    //! Constructeur par défaut
    Entry()
    : m_impl(nullptr)
    {}

    //! Constructeur par copie
    Entry(const Entry& en) = default;

    //! Constructeur
    explicit Entry(EntryImpl* impl)
    : m_impl(impl)
    {}

    //! Opérateur de copie
    Entry& operator=(const Entry& en)
    {
      if (this != &en)
        m_impl = en.m_impl;
      return *this;
    }

    //! Accès interne à l'implementation
    EntryImpl* internal() const { return m_impl; }

    //! Indique si l'entrée est définie
    bool null() const { return m_impl == nullptr; }

    void nullify() { m_impl = nullptr; }

    //! Ensemble des indices 'own' gérés par cette entrée
    Arccore::ConstArrayView<Arccore::Integer> getOwnIndexes() const
    {
      return m_impl->getOwnIndexes();
    }

    //! Ensemble des indices 'own + ghost' gérés par cette entrée (
    Arccore::ConstArrayView<Arccore::Integer> getAllIndexes() const
    {
      return m_impl->getAllIndexes();
    }

    //! Ensemble des items 'own' gérés par cette entrée
    Arccore::ConstArrayView<Arccore::Integer> getOwnLocalIds() const
    {
      return m_impl->getOwnLocalIds();
    }

    //! Ensemble des items 'own + ghost' gérés par cette entrée
    Arccore::ConstArrayView<Arccore::Integer> getAllLocalIds() const
    {
      return m_impl->getAllLocalIds();
    }

    //! Nom de l'entrée
    Arccore::String getName() const { return m_impl->getName(); }

    //! Support de l'entrée (en terme d'item)
    Arccore::Integer getKind() const { return m_impl->getKind(); }

    //! Retourne la famille abstraite de l'Entry
    const IAbstractFamily& getFamily() const { return m_impl->getFamily(); }

    //@{ @name Gestion des tags
    //! Ajout d'un tag
    void addTag(const Arccore::String& tagname, const Arccore::String& tagvalue)
    {
      return m_impl->addTag(tagname, tagvalue);
    }

    //! Suppression d'un tag
    void removeTag(const Arccore::String& tagname) { return m_impl->removeTag(tagname); }

    //! Test d'existance d'un tag
    bool hasTag(const Arccore::String& tagname) { return m_impl->hasTag(tagname); }

    //! Acces en lecture à un tag
    Arccore::String tagValue(const Arccore::String& tagname)
    {
      return m_impl->tagValue(tagname);
    }
    //@}

    //! Référentiel du manager associé
    IIndexManager* manager() const { return m_impl->manager(); }
  };

  //! Interface d'implementation de \a EntryEnumerator
  class EntryEnumeratorImpl
  {
   public:
    EntryEnumeratorImpl() = default;
    virtual ~EntryEnumeratorImpl() = default;
    virtual void moveNext() = 0;
    virtual bool hasNext() const = 0;
    virtual EntryImpl* get() const = 0;
  };

  //! Classe d'énumération des \a Entry connues
  /*! Classe de type proxy; la copie est peu couteuse et l'implémentation variable */
  class EntryEnumerator
  {
   protected:
    std::shared_ptr<EntryEnumeratorImpl> m_impl;

   public:
    //! Constructeur par copie
    EntryEnumerator(const EntryEnumerator& e) = default;

    //! Constructeur par consultation de l'\a IndexManager
    explicit EntryEnumerator(IIndexManager const* manager)
    : m_impl(manager->enumerateEntry().m_impl)
    {}
    //! Constructeur par implémentation
    explicit EntryEnumerator(EntryEnumeratorImpl* impl)
    : m_impl(impl)
    {}
    //! Avance l'énumérateur
    void operator++() { m_impl->moveNext(); }
    //! Teste l'existence d'un élément suivant
    bool hasNext() const { return m_impl->hasNext(); }
    //! Déréférencement
    Entry operator*() const { return Entry(m_impl->get()); }
    //! Déréférencement indirect
    EntryImpl* operator->() const { return m_impl->get(); }
    //! Nombre d'élément dans l'énumérateur
    Arccore::Integer count() const
    {
      Arccore::Integer my_size = 0;
      for (EntryEnumerator i = *this; i.hasNext(); ++i)
        ++my_size;
      return my_size;
    }
    bool null() { return m_impl == nullptr; }
  };

 public:
  //! Constructeur par défaut
  IIndexManager() = default;

  //! Destructeur
  virtual ~IIndexManager() = default;

  //! Indique si la phase de préparation est achevée
  virtual bool isPrepared() const = 0;

  //! Initialisation les structures
  /*! Implicitement appelé par le constructeur */
  virtual void init() = 0;

  //! Préparation : fixe l'indexation (fin des définitions)
  virtual void prepare() = 0;

  //! Définit le gestionnaire de trace
  virtual void setTraceMng(Arccore::ITraceMng* traceMng) = 0;

  //! Statistiques d'indexation
  /*! Uniquement valide après \a prepare */
  virtual void stats(Arccore::Integer& globalSize, Arccore::Integer& minLocalIndex,
                     Arccore::Integer& localSize) const = 0;

  //! Retourne la taille globale
  /*! Uniquement valide après \a prepare */
  virtual Arccore::Integer globalSize() const = 0;

  //! Retourne l'indice minimum local
  /*! Uniquement valide après \a prepare */
  virtual Arccore::Integer minLocalIndex() const = 0;

  //! Retourne l'indice minimum local
  /*! Uniquement valide après \a prepare */
  virtual Arccore::Integer localSize() const = 0;

  //! Construction d'un enumerateur sur les \a Entry
  virtual EntryEnumerator enumerateEntry() const = 0;

  //! Retourne l'entrée associée à un nom
  virtual Entry getEntry(const Arccore::String& name) const = 0;

  typedef Entry ScalarIndexSet;
  typedef Arccore::UniqueArray<ScalarIndexSet> VectorIndexSet;

  //! Construit une nouvelle entrée scalaire sur des items du maillage
  // virtual ScalarIndexSet buildScalarIndexSet(const String name,Arcane::IItemFamily *
  // item_family) = 0;
  // virtual void defineIndex(ScalarIndexSet& set , const Arcane::ItemGroup & itemGroup)
  // =
  // 0;
  // virtual ScalarIndexSet buildScalarIndexSet(const String name, const
  // Arcane::ItemGroup
  // & itemGroup) = 0;

  //! Construit une nouvelle entrée scalaire sur un ensemble d'entités abstraites
  virtual ScalarIndexSet buildScalarIndexSet(const Arccore::String& name,
                                             Arccore::ConstArrayView<Arccore::Integer> localIds,
                                             const IAbstractFamily& family) = 0;

  //! Construit une nouvelle entrée scalaire sur l'ensemble des entités d'une familles
  //! abstraite
  virtual ScalarIndexSet buildScalarIndexSet(
  const Arccore::String& name, const IAbstractFamily& family) = 0;

  //! Construit une nouvelle entrée vectorielle sur des items du maillage
  /*! L'implémentation actuelle considére le multi-scalaire comme du vectoriel */
  // virtual VectorIndexSet buildVectorIndexSet(const String name, const
  // Arcane::ItemGroup
  // & itemGroup, const Integer n) = 0;
  //! Construit une nouvelle entrée vectoriellesur un ensemble d'entités abstraites
  /*! L'implémentation actuelle considére le multi-scalaire comme du vectoriel */
  virtual VectorIndexSet buildVectorIndexSet(const Arccore::String& name,
                                             Arccore::ConstArrayView<Arccore::Integer> localIds,
                                             const IAbstractFamily& family,
                                             Arccore::Integer n) = 0;

  //! Construit une nouvelle entrée scalaire sur l'ensemble des entités d'une familles
  //! abstraite
  /*! L'implémentation actuelle considére le multi-scalaire comme du vectoriel */
  virtual VectorIndexSet buildVectorIndexSet(
  const Arccore::String& name, const IAbstractFamily& family, Arccore::Integer n) = 0;

  //! Demande de dé-indexation d'une partie d'une entrée
  /*! Utilisable uniquement avant prepare */
  // virtual void removeIndex(const ScalarIndexSet & entry, const Arcane::ItemGroup &
  // itemGroup) = 0;

  // virtual Integer getIndex(const Entry & entry, const Item & item) const = 0 ;

  //! Consultation vectorielle d'indexation d'une entrée (après prepare)
  // virtual void getIndex(const ScalarIndexSet & entry, const Arcane::ItemVectorView &
  // items, ArrayView<Integer> indexes) const = 0;

  //! Fournit une table de translation indexé par les items
  virtual Arccore::UniqueArray<Arccore::Integer> getIndexes(
  const ScalarIndexSet& entry) const = 0;

  //! Fournit une table de translation indexé par les items
  virtual Arccore::UniqueArray2<Arccore::Integer> getIndexes(
  const VectorIndexSet& entry) const = 0;

  //! Donne le gestionnaire parallèle ayant servi à l'indexation
  virtual Alien::IMessagePassingMng* parallelMng() const = 0;

  //! define null index : default == -1, if true nullIndex() == max index of current
  //! indexation
  virtual void setMaxNullIndexOpt(bool flag) = 0;

  //! return value of null index
  virtual Arccore::Integer nullIndex() const = 0;

 public:
  //! Permet de gérer la mort d'une famille associée à l'index-manager
  /*! Méthode de bas niveau pour les implémentationsde IAbstractFamily,
   *  usuellement dans le desctructeur des implémentations extérieures de
   * IAbstractFamily
   */
  virtual void keepAlive(const IAbstractFamily* family) = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Alien

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
