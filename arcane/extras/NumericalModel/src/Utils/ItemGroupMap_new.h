// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
#ifndef ITEMGROUPMAP_NEW_H
#define ITEMGROUPMAP_NEW_H

#include <arcane/utils/HashTableMap.h>
#include <arcane/Item.h>
#include <arcane/ItemGroup.h>
#include <arcane/ItemGroupRangeIterator.h>
#include <arcane/ItemEnumerator.h>
#include <arcane/utils/ITraceMng.h>
#include <arcane/utils/Trace.h>
#include <arcane/utils/TraceInfo.h>
#include <arcane/ArcaneVersion.h>

#include <arcane/ItemVectorView.h>
#include <arcane/mesh/ItemFamily.h>

#include <map>

#ifndef ITEMGROUP_USE_OBSERVERS
#error "This implementation of ItemGroupMap is only available with new ItemGroupImpl Observers"
#endif /* ITEMGROUP_USE_OBSERVERS */


using namespace Arcane;

//! Classe fondamentale technique des ItemGroupMap's
/*! \todo optimiser la table de hachage vu que l'on connait l'ensemble
 *  des indices à la construction (les collisions sont consultables via stats())
 */
class ItemGroupMapAbstract
  : protected HashTableBase
{
protected:
  typedef Integer KeyTypeValue;
  typedef HashTraitsT<KeyTypeValue> KeyTraitsType;
  typedef KeyTraitsType::KeyTypeConstRef KeyTypeConstRef;
  //   typedef KeyTraitsType::KeyTypeValue KeyTypeValue;

public:
  enum Property {
    eResetOnResize     = 1<<1,
    eRecomputeOnResize = 1<<2
  };

public:
  //! Constructeur
  ItemGroupMapAbstract();
    
  //! Destructeur
  virtual ~ItemGroupMapAbstract();

  //! Initialisation sur un nouveau groupe
  virtual void init(const ItemGroup & group) = 0;

  //! Accès aux stats
  /*! Inclus la description des collisions */
  void stats(std::ostream & o);

  //! Accès au groupe associé
  ItemGroup group() const { return m_group; }

  //! Controle la cohérence du groupe avec le groupe de référence
  virtual bool checkSameGroup(const ItemGroup & group) const;

  //! Controle la cohérence du groupe avec le groupe de référence
  virtual bool checkSameGroup(const ItemVectorView & group) const;
  
  //! Accès aux propriétés (lecture seule)
  virtual Integer properties() const { return m_properties; }

  //! Modification des propriétés (activation)
  virtual void setProperties(const Integer property);

  //! Modification des propriétés (désactivation)
  virtual void unsetProperties(const Integer property);

  //! Nom de l'objet
  String name() const;

protected:
  //! Initialisation de bas niveau
  void _init(const ItemGroup & group);

  //! Installe les observers adaptés aux propriétés
  void _updateObservers(ItemGroupImpl * group = NULL);

  //! Compactage
  /*! Ne change pas le nombre de items du group
   *  Appelé par le ItemGroupMapAbstractVariable
   * \warning le compactage de la variable est appelé avec la renumérotation effective du groupe 
   * \warning le compactage n'induit pas de changement de taille
   */
  virtual void _executeExtend(const Int32ConstArrayView * old_to_new_ids) = 0;

  virtual void _executeReduce(const Int32ConstArrayView * old_to_new_ids) = 0;

  virtual void _executeCompact(const Int32ConstArrayView * old_to_new_ids) = 0;

  virtual void _executeInvalidate() = 0;

  //! Test l'intégrité de l'ItemGroupMap relativement à son groupe
  bool _checkGroupIntegrity() const;

  //! Fonction de hachage
  /*! Utilise la fonction de hachage de Arcane même si quelques
   *  collisions sont constatées avec les petites valeurs */
  Integer _hash(KeyTypeConstRef id) const;

  //! \a true si une valeur avec la clé \a id est présente
  bool _hasKey(KeyTypeConstRef id) const;

  //! Recherche d'une clef dans un bucket
  Integer _lookupBucket(Integer bucket, KeyTypeConstRef id) const;

  //! Recherche d'une clef dans toute la table
  Integer _lookup(KeyTypeConstRef id) const;

  //! Teste l'initialisation de l'objet
  bool _initialized() const;

  //! Génère une erreur de recherche
  void _throwItemNotFound(const TraceInfo & info, const Item & item) const throw();

  //! Accès aux traces
  ITraceMng * traceMng() const;

protected:
  ItemGroupImpl * m_group; //!< Groupe associé
  Integer m_properties; //! Propriétés
  Array<KeyTypeValue> m_key_buffer; //! Table des clés associées
  Array<Integer> m_next_buffer; //! Table des index suivant associés
  Array<Integer> m_buckets; //! Tableau des buckets
};

/*---------------------------------------------------------------------------*/

//! Classe d'implémentation des parties communes aux ItemGroupMap
template<typename _ValueType>
class ItemGroupMapAbstractT
  : public ItemGroupMapAbstract
{
public:
  typedef _ValueType ValueType;

protected:
  typedef ItemGroupMapAbstractT<ValueType> ThatClass;

public:

  //! \brief Constructeur d'une table vide
  ItemGroupMapAbstractT()
    : ItemGroupMapAbstract()
  {
    ;
  }

  //! \brief Constructeur d'une table adaptée à un groupe
  ItemGroupMapAbstractT(const ItemGroup & group)
    : ItemGroupMapAbstract()
  {
    init(group);
  }
  
  //! Destructeur
  virtual ~ItemGroupMapAbstractT()
  {
    ;
  }

private:
  //! Constructeur par copie
  /*! Zone privé + non défini => usage interdit */
  ItemGroupMapAbstractT(const ItemGroupMapAbstractT &);

  //! Opérateur de recopie
  /*! Zone privé + non défini => usage interdit */
  const ThatClass& operator=(const ThatClass& from);

public:
  //! Initialise la structure sur un groupe
  void init(const ItemGroup & group)
  {
    _init(group);
    m_data_buffer.resize(group.size());
#ifdef ITEMGROUPMAP_FILLINIT
    fill(ValueType());
#endif /* ITEMGROUPMAP_FILLINIT */
    ARCANE_ASSERT( (_checkGroupIntegrity()), ("ItemGroupMap integrity failed") );
  }

  //! Remplit la structure avec une valeur constante
  void fill(const ValueType & v) 
  {
    m_data_buffer.fill(v);
  }

protected:
  virtual void _executeExtend(const Int32ConstArrayView * new_ids_ptr)
  {
    const Int32ConstArrayView & new_ids = *new_ids_ptr;
    if (new_ids.empty()) return;

#ifdef ARCANE_DEBUG_ASSERT
    const Integer old_size = m_key_buffer.size();
#endif
    const Integer group_size = group().size();
    ARCANE_ASSERT((group_size == old_size+new_ids.size()),("Inconsitent extended size"));

    m_data_buffer.resize(group_size);
    _init(group());
    ARCANE_ASSERT( (_checkGroupIntegrity()), ("ItemGroupMap integrity failed") );
  }

  virtual void _executeReduce(const Int32ConstArrayView * removed_ids_ptr)
  {
    // contient la liste des positions supprimés dans le groupe original original
    const Int32ConstArrayView & removed_ids = *removed_ids_ptr; 
    if (removed_ids.empty()) return;

    const Integer old_size = m_key_buffer.size();
    const Integer group_size = group().size();
    ARCANE_ASSERT((group_size == old_size-removed_ids.size()),("Inconsitent reduced size %d vs %d",group_size,old_size));
    
    ItemVectorView view = group().view();
    for(Integer i=0,index=0,removed_index=0; i<old_size ;++i)
      {
        if (removed_index < removed_ids.size() && 
            i == removed_ids[removed_index])
          {
            ++removed_index;
          }
        else
          {
            ARCANE_ASSERT((m_key_buffer[i] == view[index].localId()),
                          ("Inconsistent key (pos=%d,key=%d) vs (pos=%d,key=%d)",i,m_key_buffer[i],index,view[index].localId()));
             if (i != index)
               m_data_buffer[index] = m_data_buffer[i];
            ++index;
          }
      }
    m_data_buffer.resize(group_size);
    _init(group());
    ARCANE_ASSERT( (_checkGroupIntegrity()), ("ItemGroupMap integrity failed") );
  }

//   virtual void _executeCompact(const Int32ConstArrayView * old_to_new_ids_ptr)
//   {
//     // Avec cette version l'ordre relatif du groupe peut avoir changé
//     const Int32ConstArrayView & old_to_new_ids = *old_to_new_ids_ptr;

//     // La taille du groupe n'a pas changé mais on réordonne les données
//     ARCANE_ASSERT((group().size()==m_buffer.size()),("Inconsistent sizes"));
//     m_buckets.fill(-1);

//     traceMng()->debug(Trace::High) << "Compacting group " << group().name();

//     const Integer size = m_buffer.size();

//     Array<Data> old_buffer;
//     old_buffer.setArray(m_buffer); // Echange des données sans recopie
//     m_buffer.setArray(Array<Data>());
//     m_buffer.resize(size);

//     std::map<Integer,Integer> old_lid_to_data;
//     for(Integer i=0;i<size;++i)
//       {
//         const Integer old_key = m_key_buffer[i];
//         old_lid_to_data[old_key] = i;
//         traceMng()->debug(Trace::Highest) << i << ": Known old lid : " << old_key;
//       }

//     std::map<Integer,Data*> new_lid_to_data;
//     for(Integer oldid=0;oldid<old_to_new_ids.size();++oldid)
//       {
//         const Integer new_lid = old_to_new_ids[oldid];
//         typename std::map<Integer,Data*>::const_iterator idata = old_lid_to_data.find(oldid);
//         if (idata != old_lid_to_data.end()) {
//           new_lid_to_data[new_lid] = idata->second;
//         }
//       }
//     ARCANE_ASSERT(((Integer)new_lid_to_data.size() == (Integer)m_group->size()),("Incompatible sizes after conversion"));

//     ENUMERATE_ITEM(iitem,group())
//       {
//         const KeyTypeConstRef key = iitem.localId();
//         const Integer bucket = _hash(key);
//         ARCANE_ASSERT( (_lookupBucket(bucket,key) == NULL), ("Already assigned key"));
//         Data & hd = m_buffer[iitem.index()];
//         hd.m_key = key;
//         hd.m_next = m_buckets[bucket];
//         m_buckets[bucket] = &hd;
//         typename std::map<Integer,Data*>::iterator idata = new_lid_to_data.find(key);
//         // ARCANE_ASSERT((idata!=new_lid_to_data.end()),("Old data not found"));
//         if (idata!=new_lid_to_data.end())
//           hd.m_value = idata->second->m_value;
//       }
//     ARCANE_ASSERT( (_checkGroupIntegrity()), ("ItemGroupMap integrity failed") );
//   }

  virtual void _executeCompact(const Int32ConstArrayView * old_to_new_ids_ptr)
  { 
   // Avec cette version, on suppose que l'ordre relatif des ids n'a pas changé
#ifdef ARCANE_DEBUG_ASSERT
    const Int32ConstArrayView & old_to_new_ids = *old_to_new_ids_ptr;
#endif

    // La taille du groupe n'a pas changé mais on réordonne les données
    ARCANE_ASSERT((group().size()==m_key_buffer.size()),("Inconsistent sizes"));

    traceMng()->debug(Trace::High) << "Compacting group " << group().name();

#ifdef NDEBUG
    _init(group());
#else /* NDEBUG */
    m_buckets.fill(-1);
    ENUMERATE_ITEM(iitem,group())
      {
        const KeyTypeConstRef key = iitem.localId();
        const KeyTypeConstRef old_key = m_key_buffer[iitem.index()];
        const Integer i = iitem.index();
        traceMng()->debug(Trace::High) << i << " " << old_key << " -> " << key << " vs " << old_to_new_ids[old_key];
        const Integer bucket = _hash(key);
        ARCANE_ASSERT( (_lookupBucket(bucket,key) < 0), ("Already assigned key"));
        ARCANE_ASSERT( (old_to_new_ids[old_key] == key),("Inconsistent reorder translation %d vs %d vs %d",old_to_new_ids[old_key],key,old_key));
        m_key_buffer[i] = key;
        m_next_buffer[i] = m_buckets[bucket];
        m_buckets[bucket] = i;
      }
#endif /* NDEBUG */
    ARCANE_ASSERT( (_checkGroupIntegrity()), ("ItemGroupMap integrity failed") );
  }

  virtual void _executeInvalidate()
  {
    init(this->group());
    ARCANE_ASSERT( (_checkGroupIntegrity()), ("ItemGroupMap integrity failed") );
  }

public:

public:
  /*! \brief Recherche la valeur correspondant à l'item \a item
   *
   * Une exception est générée si la valeur n'est pas trouvé.
   */
  inline const ValueType& operator[](const Item & item) const throw() 
  {
    const Integer i = _lookup(item.localId());
    if (i<0) this->_throwItemNotFound(A_FUNCINFO,item);
    return m_data_buffer[i];
  }

  /*! \brief Recherche la valeur correspondant à l'item \a item
   *
   * Une exception est générée si la valeur n'est pas trouvé.
   */
  inline ValueType& operator[](const Item & item) throw()
  {
    const Integer i = _lookup(item.localId());
    if (i<0) this->_throwItemNotFound(A_FUNCINFO,item);
    return m_data_buffer[i];
  }

  /*! \brief Recherche la valeur correspondant à l'item associé à un iterateur
   */
  inline const ValueType& operator[](const ItemGroupRangeIterator & iter) const
  {
    ARCANE_ASSERT((_lookup(iter.itemLocalId()) == iter.index()),("Inconsistency detected on item id"));
    return m_data_buffer[iter.index()];
  }

  /*! \brief Recherche la valeur correspondant à l'item associé à un iterateur
   */
  inline ValueType& operator[](const ItemGroupRangeIterator & iter)
  {
    ARCANE_ASSERT((_lookup(iter.itemLocalId()) == iter.index()),("Inconsistency detected on item id"));
    return m_data_buffer[iter.index()];
  }

  /*! \brief Recherche la valeur correspondant à l'item associé à un énumérateur
   */
  inline const ValueType& operator[](const ItemEnumeratorT<Item> & iter) const
  {
    ARCANE_ASSERT((_lookup(iter.itemLocalId()) == iter.index()),("Inconsistency detected on item id"));
    return m_data_buffer[iter.index()];
  }

  /*! \brief Recherche la valeur correspondant à l'item associé à un énumérateur
   */
  inline ValueType& operator[](const ItemEnumeratorT<Item> & iter)
  {
    ARCANE_ASSERT((_lookup(iter.itemLocalId()) == iter.index()),("Inconsistency detected on item id"));
    return m_data_buffer[iter.index()];
  }

  //! Test l'existence d'une clé
  inline bool hasKey(const Item & item) const
  {
    return _hasKey(item.localId());
  }

protected:
  Array<ValueType> m_data_buffer;
};

/*---------------------------------------------------------------------------*/

//! \brief Classe de base pour les ItemGroupMap
/*  Contient la majorité des fonctionnalités mais sans distinction du
 *  type d'item/
 *
 *  Permet d'indéxer un tableau par les items d'un groupe. Ceci évite
 *  des erreurs principalement en parallèle ou certains items ne sont
 *  pas contigus en localId (et évite le bug d'indexé un Array via les
 *  localId des items).
 *
 *  Il n'est possible d'accèder qu'à des items du groupe
 *  original. Retourne une exception si l'item demandé n'est pas
 *  référencé. L'accès est optimiser pour les ItemGroupRangeIterator
 *
 *  Cette implémentation stocke les valeurs dans la structure Data de
 *  la table de hachage. Elle ne sert qu'à typer les Items.
 *
 * \see ItemGroupMapT et ItemGroupMapArrayT
 */
template <typename ValueType>
class ItemGroupMapBaseT : 
  public ItemGroupMapAbstractT<ValueType>
{
protected:
  typedef ItemGroupMapAbstractT<ValueType> BaseClass;

public:
  //! Constructeur par défaut
  ItemGroupMapBaseT() 
    : BaseClass()
  {
    ;
  }
  
  //! Constructeur à partir d'un groupe
  ItemGroupMapBaseT(const ItemGroup & group) 
    : BaseClass()
  {
    init(group);
  }

  //! Destructeur
  virtual ~ItemGroupMapBaseT()
  {
    ;
  }
  
  //! Initialisation sur un nouveau groupe
  void init(const ItemGroup & group) 
  {
    // Met la valeur par défaut en association aux clefs
    BaseClass::init(group);
  }

  //! Opérateur d'accès à partir d'un item
  ValueType & operator[](const Item & item) 
  {
    return BaseClass::operator[](item);
  }

  //! Opérateur d'accès à partir d'un ItemGroupRangeIteratorT
  ValueType & operator[](const ItemGroupRangeIterator & iter)
  {
    return BaseClass::operator[](iter);
  }

  //! Opérateur d'accès en mode constant à partir d'un item
  const ValueType & operator[](const Item & item) const
  {
    return BaseClass::operator[](item);
  }

  //! Opérateur d'accès en mode constant à partir d'un ItemGroupRangeIteratorT
  const ValueType & operator[](const ItemGroupRangeIterator & iter) const
  {
    return BaseClass::operator[](iter);
  }

  /*! \brief Recherche la valeur correspondant à l'item associé à un énumérateur
   */
  inline const ValueType& operator[](const ItemEnumerator & iter) const
  {
    return BaseClass::operator[](iter);    
  }

  /*! \brief Recherche la valeur correspondant à l'item associé à un énumérateur
   */
  inline ValueType& operator[](const ItemEnumerator & iter)
  {
    return BaseClass::operator[](iter);
  }

  //! Teste l'existence d'un item en temps que clef
  bool hasKey(const Item & item) const
  {
    return BaseClass::hasKey(item);
  }
};

/*---------------------------------------------------------------------------*/

/*! \brief Forme de tableau indéxé sur un groupe d'items de type ItemKind
 *
 *  Permet d'indéxer un tableau par les items d'un groupe. Ceci évite
 *  des erreurs principalement en parallèle ou certains items ne sont
 *  pas contigus en localId (et évite le bug d'indexé un Array via les
 *  localId des items).
 *
 *  Il n'est possible d'accèder qu'à des items du groupe
 *  original. Retourne une exception si l'item demandé n'est pas
 *  référencé. L'accès est optimiser pour les ItemGroupRangeIterator
 *
 *  Cette implémentation stocke les valeurs dans la structure Data de
 *  la table de hachage. Elle ne sert qu'à typer les Items.
 *
 * \see ItemGroupMapArrayT
 */
template <typename ItemKind, typename ValueType>
class ItemGroupMapT : 
  public ItemGroupMapAbstractT<ValueType>
{
protected:
  typedef ItemGroupMapAbstractT<ValueType> BaseClass;

public:
  //! Constructeur par défaut
  ItemGroupMapT()
    : BaseClass()
  {
    ;
  }
  
  //! Constructeur à partir d'un groupe
  ItemGroupMapT(const ItemGroupT<ItemKind> & group) 
    : BaseClass()
  {
    init(group);
  }
  
  //! Destructeur
  virtual ~ItemGroupMapT()
  {
    ;
  }
  
  //! Initialisation sur un nouveau groupe
  void init(const ItemGroupT<ItemKind> & group) 
  {
    // Met la valeur par défaut en association aux clefs
    BaseClass::init(group);
  }

  //! Opérateur d'accès à partir d'un item
  ValueType & operator[](const ItemKind & item) 
  {
    return BaseClass::operator[](item);
  }

  //! Opérateur d'accès à partir d'un ItemGroupRangeIteratorT
  ValueType & operator[](const ItemGroupRangeIteratorT<ItemKind> & iter)
  {
    return BaseClass::operator[](iter);
  }

  //! Opérateur d'accès en mode constant à partir d'un item
  const ValueType & operator[](const ItemKind & item) const
  {
    return BaseClass::operator[](item);
  }

  //! Opérateur d'accès en mode constant à partir d'un ItemGroupRangeIteratorT
  const ValueType & operator[](const ItemGroupRangeIteratorT<ItemKind> & iter) const
  {
    return BaseClass::operator[](iter);
  }

  /*! \brief Recherche la valeur correspondant à l'item associé à un énumérateur
   */
  inline const ValueType& operator[](const ItemEnumeratorT<ItemKind> & iter) const
  {
    return BaseClass::operator[](iter);    
  }

  /*! \brief Recherche la valeur correspondant à l'item associé à un énumérateur
   */
  inline ValueType& operator[](const ItemEnumeratorT<ItemKind> & iter)
  {
    return BaseClass::operator[](iter);    
  }

  bool hasKey(const ItemKind & item) const
  {
    return BaseClass::hasKey(item);
  }
};

/*---------------------------------------------------------------------------*/

//! \brief Classe de base pour les ItemGroupMapArray
/*  Contient la majorité des fonctionnalités mais sans distinction du
 *  type d'item/
 *
 *  Permet d'indéxé un tableau par les items d'un groupe. Ceci évite
 *  des erreurs principalement en parallèle ou certains items ne sont
 *  pas contigus en localId (et évite le bug d'indexé un Array via les
 *  localId des items).
 *
 *  Il n'est possible d'accèder qu'à des items du groupe
 *  original. Retourne une exception si l'item demandé n'est pas
 *  référencé. L'accès en optimiser pour les ItemGroupRangeIterator
 *
 * \see ItemGroupMapT
 */
template <typename ValueType>
class ItemGroupMapArrayBaseT : 
  public ItemGroupMapAbstractT<ValueType*>
{
public:
  typedef ArrayView<ValueType> ArrayType;
  typedef ConstArrayView<ValueType> ConstArrayType;

protected:
  typedef ItemGroupMapAbstractT<ValueType*> BaseClass;
  Integer m_array_size;
  Array<ValueType> m_array_data;

public:
  //! Constructeur par défaut
  ItemGroupMapArrayBaseT() 
    : BaseClass()
    , m_array_size(0)
  {
    ;
  }

  //! Constructeur à partir d'un groupe
  ItemGroupMapArrayBaseT(const ItemGroup & group, 
                         const Integer array_size)
    : BaseClass()
  {
    init(group,array_size);
  }

  //! Destructeur
  virtual ~ItemGroupMapArrayBaseT()
  {
    ;
  }
  
  //! Initialisation sur un nouveau groupe
  void init(const ItemGroup & group, const Integer array_size) 
  {
    BaseClass::init(group);
    m_array_size = array_size;
    const Integer global_size = group.size()*m_array_size;
    m_array_data.resize(global_size);
    ValueType * base_data = m_array_data.unguardedBasePointer();
    for(ItemEnumerator i(group.enumerator()) ; i.hasNext(); ++i)
      {
        ValueType * currentData = base_data + m_array_size * i.index();
        BaseClass::operator[](i) = currentData;
      }
#ifdef ITEMGROUPMAP_FILLINIT
    fill(ValueType());
#endif /* ITEMGROUPMAP_FILLINIT */
  }

  //! Initialisation sans changement de taille des tableaux
  void init(const ItemGroup & group)
  {
    init(group,m_array_size);
  }

  //! Remplit la structure avec une valeur constante
  void fill(const ValueType & v) 
  {
    m_array_data.fill(v);
  }

  //! Opérateur d'accès à partir d'un item
  /*! Retourne une exception si l'item n'est pas référencé */
  ArrayType operator[](const Item & item) 
  {
    return ArrayType(m_array_size,BaseClass::operator[](item));
  }

  //! Opérateur d'accès à partir d'un ItemGroupRangeIteratorT
  /*! Retourne une exception si l'item n'est pas référencé */
  ArrayType operator[](const ItemGroupRangeIterator & iter)
  {
    return ArrayType(m_array_size,BaseClass::operator[](iter));
  }

  /*! \brief Recherche la valeur correspondant à l'item associé à un énumérateur
   */
  ArrayType operator[](const ItemEnumerator & iter)
  {
    return ArrayType(m_array_size,BaseClass::operator[](iter));
  }

  //! Opérateur d'accès en mode constant à partir d'un item
  /*! Retourne une exception si l'item n'est pas référencé */
  ConstArrayType operator[](const Item & item) const
  {
    return ConstArrayType(m_array_size,BaseClass::operator[](item));
  }

  //! Opérateur d'accès en mode constant à partir d'un ItemGroupRangeIteratorT
  /*! Retourne une exception si l'item n'est pas référencé */
  ConstArrayType operator[](const ItemGroupRangeIterator & iter) const
  {
    return ConstArrayType(m_array_size,BaseClass::operator[](iter));
  }

  /*! \brief Recherche la valeur correspondant à l'item associé à un énumérateur
   */
  inline const ConstArrayType operator[](const ItemEnumerator & iter) const
  {
    return ConstArrayType(m_array_size,BaseClass::operator[](iter));
  }

  bool hasKey(const Item & item) const
  {
    return BaseClass::hasKey(item);
  }

protected:
  virtual void _executeExtend(const Int32ConstArrayView * old_to_new_ids)
  {
    BaseClass::_executeExtend(old_to_new_ids);
    _dataShifter();
  }

  virtual void _executeReduce(const Int32ConstArrayView * old_to_new_ids)
  {
    BaseClass::_executeReduce(old_to_new_ids);
    _dataShifter();
  }

  virtual void _executeCompact(const Int32ConstArrayView * old_to_new_ids)
  {
    BaseClass::_executeCompact(old_to_new_ids);
    // _dataShifter(); // inutile tant que cela est de la compaction sans réarrangement ni réduction de taille
  }

  virtual void _executeInvalidate()
  {
    init(this->group());
  }

private:
  void _dataShifter()
  {
    ItemGroup group(this->group());
    Array<ValueType> old_array = m_array_data; // sans copie grace à Array
    const Integer global_size = group.size() * m_array_size;
    m_array_data = Array<ValueType>(global_size);
    ValueType * base_data = m_array_data.unguardedBasePointer();
    for(ItemEnumerator i(group.enumerator()) ; i.hasNext(); ++i)
      {
        ValueType * currentData = base_data + m_array_size * i.index();
        ValueType * oldData = BaseClass::operator[](i);
        for(Integer j=0;j<m_array_size;++j)
          currentData[j] = oldData[j];
        BaseClass::operator[](i) = currentData;
      }
  }
};

/*---------------------------------------------------------------------------*/

/*! \brief Forme de tableau indéxé sur un groupe d'items à valeurs de type tableau
 *
 *  Permet d'indéxé un tableau par les items d'un groupe. Ceci évite
 *  des erreurs principalement en parallèle ou certains items ne sont
 *  pas contigus en localId (et évite le bug d'indexé un Array via les
 *  localId des items).
 *
 *  Il n'est possible d'accèder qu'à des items du groupe
 *  original. Retourne une exception si l'item demandé n'est pas
 *  référencé. L'accès en optimiser pour les ItemGroupRangeIterator
 *
 * \see ItemGroupMapT
 */
template <typename ItemKind, typename ValueType>
class ItemGroupMapArrayT : 
  public ItemGroupMapArrayBaseT<ValueType>
{
protected:
  typedef ItemGroupMapArrayBaseT<ValueType> BaseClass;
  typedef typename BaseClass::ArrayType ArrayType;
  typedef typename BaseClass::ConstArrayType ConstArrayType;
  
public:
  //! Constructeur par défaut
  ItemGroupMapArrayT()
    : BaseClass()
  {
    ;
  }
  
  //! Constructeur à partir d'un groupe
  ItemGroupMapArrayT(const ItemGroupT<ItemKind> & group, 
                     const Integer array_size)
    : BaseClass(group,array_size)
  {
    ;
  }

  //! Destructeur
  virtual ~ItemGroupMapArrayT()
  {
    ;
  }
  
  //! Initialisation sur un nouveau groupe
  void init(const ItemGroupT<ItemKind> & group, const Integer array_size)
  {
    // Met la valeur par défaut en association aux clefs
    BaseClass::init(group,array_size);
  }

  //! Initialisation sur un nouveau groupe
  void init(const ItemGroupT<ItemKind> & group) 
  {
    // Met la valeur par défaut en association aux clefs
    BaseClass::init(group);
  }

  //! Opérateur d'accès à partir d'un item
  ArrayType operator[](const ItemKind & item) 
  {
    return BaseClass::operator[](item);
  }

  //! Opérateur d'accès à partir d'un ItemGroupRangeIteratorT
  ArrayType operator[](const ItemGroupRangeIteratorT<ItemKind> & iter)
  {
    return BaseClass::operator[](iter);
  }

  //! Opérateur d'accès en mode constant à partir d'un item
  ConstArrayType operator[](const ItemKind & item) const
  {
    return BaseClass::operator[](item);
  }

  //! Opérateur d'accès en mode constant à partir d'un ItemGroupRangeIteratorT
  ConstArrayType operator[](const ItemGroupRangeIteratorT<ItemKind> & iter) const
  {
    return BaseClass::operator[](iter);
  }

  /*! \brief Recherche la valeur correspondant à l'item associé à un énumérateur
   */
  ConstArrayType operator[](const ItemEnumeratorT<ItemKind> & iter) const
  {
    return BaseClass::operator[](iter);    
  }

  /*! \brief Recherche la valeur correspondant à l'item associé à un énumérateur
   */
  ArrayType operator[](const ItemEnumeratorT<ItemKind> & iter)
  {
    return BaseClass::operator[](iter);    
  }

  bool hasKey(const ItemKind & item) const
  {
    return BaseClass::hasKey(item);
  }
};

/*---------------------------------------------------------------------------*/

//! Classe vide
/*! Coute un octet en mémoire */
class Void { };
// inline std::ostream & operator<<(std::ostream & o, const Void & v) { return o; }

/*---------------------------------------------------------------------------*/

/*! ItemGroupSet s'utilise essentiellement avec hasKey vu que les données portées sont Void */
class ItemGroupSet : protected ItemGroupMapAbstractT<Void>
{
public:
  ItemGroupSet()
    : BaseClass()
  {
    m_properties |= eResetOnResize;
  }

  ItemGroupSet(const ItemGroup & group) 
    : BaseClass(group) 
  {
    m_properties |= eResetOnResize;
  }

  void init(const ItemGroup & group)
  {
    BaseClass::init(group);
  }

  bool hasKey(const Item & item) const
  {
    return BaseClass::hasKey(item);
  }

protected:
  typedef ItemGroupMapAbstractT<Void> BaseClass;
};

#endif /* ITEMGROUPMAP_NEW_H */
