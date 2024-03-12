// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* HashTableMap.h                                              (C) 2000-2024 */
/*                                                                           */
/* Tableau associatif utilisant une table de hachage.                        */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_UTILS_HASHTABLEMAP_H
#define ARCANE_UTILS_HASHTABLEMAP_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/HashTable.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <typename KeyType, typename ValueType>
class HashTableMapEnumeratorT;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * \brief Table de hachage pour tableaux associatifs.
 
 Cette table permet de stocker une valeur en fonction d'une clé. La
 clé est de type \a KeyType et la valeur \a ValueType.

 Cette table permet pour l'instant uniquement d'ajouter des valeurs.
 La mémoire associée à chaque entrée du tableau est gérée par
 un MultiBufferT.

 Il est possible de spécifier une fonction de hachage différente de
 la fonction par défaut en spécifiant le troisième paramètre
 template \a KeyTraitsType.

 Pour des raisons de performance, il est préférable que la taille
 de la table (buckets) soit un nombre premier.
*/
template <typename KeyType, typename ValueType, typename KeyTraitsType = HashTraitsT<KeyType>>
class HashTableMapT
: public HashTableBase
{
 public:

  typedef typename KeyTraitsType::KeyTypeConstRef KeyTypeConstRef;
  typedef typename KeyTraitsType::KeyTypeValue KeyTypeValue;
  typedef typename KeyTraitsType::Printable Printable;
  typedef typename KeyTraitsType::HashValueType HashValueType;
  typedef HashTableMapT<KeyType, ValueType, KeyTraitsType> ThatClass;
  typedef HashTableMapEnumeratorT<KeyType, ValueType> Enumerator;

 public:

  struct Data
  {
   public:

    Data()
    : m_key(KeyTypeValue())
    , m_value(ValueType())
    {}

   public:

    Data* next() { return m_next; }
    void setNext(Data* anext) { this->m_next = anext; }
    KeyTypeConstRef key() { return m_key; }
    const ValueType& value() const { return m_value; }
    ValueType& value() { return m_value; }
    //! Modifie la valeur de l'instance.
    void setValue(const ValueType& avalue) { m_value = avalue; }
    /*!
     * \brief Change la valeur de la clé.
     *
     * Après avoir changé la valeur d'une ou plusieurs clés, il faut faire un rehash().
     */
    void setKey(const KeyType& new_key)
    {
      m_key = new_key;
    }

   public:

    KeyTypeValue m_key; //!< Clé de recherche
    ValueType m_value; //!< Valeur de l'élément
    Data* m_next = nullptr; //! Elément suivant dans la table de hachage
  };

 public:

  /*! \brief Crée une table de taille \a table_size
   *
   Si \a use_prime est vrai, utilise la fonction nearestPrimeNumber()
   pour avoir une taille de taille qui est un nombre premier.
  */
  HashTableMapT(Integer table_size, bool use_prime)
  : HashTableBase(table_size, use_prime)
  , m_first_free(0)
  , m_nb_collision(0)
  , m_nb_direct(0)
  , m_max_count(0)
  {
    m_buffer = new MultiBufferT<Data>(m_nb_bucket);
    m_buckets.resize(m_nb_bucket);
    m_buckets.fill(0);
    _computeMaxCount();
  }

  /*! \brief Crée une table de taille \a table_size
   *
   Si \a use_prime est vrai, utilise la fonction nearestPrimeNumber()
   pour avoir une taille de taille qui est un nombre premier.
  */
  HashTableMapT(Integer table_size, bool use_prime, Integer buffer_size)
  : HashTableBase(table_size, use_prime)
  , m_first_free(0)
  , m_nb_collision(0)
  , m_nb_direct(0)
  {
    m_buffer = new MultiBufferT<Data>(buffer_size);
    m_buckets.resize(m_nb_bucket);
    m_buckets.fill(0);
    _computeMaxCount();
  }

  ~HashTableMapT()
  {
    delete m_buffer;
  }

  //! Opérateur de recopie
  ThatClass& operator=(const ThatClass& from)
  {
    if (&from == this)
      return *this;
    //cout << "** OPERATOR= this=" << this << '\n';
    Integer nb_bucket = from.m_nb_bucket;
    m_first_free = 0;
    // Remet à zéro le compteur.
    m_count = 0;
    m_buckets.resize(nb_bucket);
    m_buckets.fill(0);
    _computeMaxCount();
    delete m_buffer;
    m_buffer = new MultiBufferT<Data>(nb_bucket);
    ConstArrayView<Data*> from_buckets(from.buckets());
    for (Integer i = 0; i < nb_bucket; ++i)
      for (Data* data = from_buckets[i]; data; data = data->next())
        _add(i, data->key(), data->value());
    this->m_nb_bucket = nb_bucket;
    return *this;
  }

 public:

  //! \a true si une valeur avec la clé \a id est présente
  bool hasKey(KeyTypeConstRef id)
  {
    Integer hf = _keyToBucket(id);
    for (Data* i = m_buckets[hf]; i; i = i->m_next) {
      if (i->key() == id)
        return true;
    }
    return false;
  }

  //! Supprime tous les éléments de la table
  void clear()
  {
    m_buckets.fill(0);
    m_count = 0;
  }

  /*!
   * \brief Recherche la valeur correspondant à la clé \a id.
   *
   * \return la structure associé à la clé \a id (0 si aucune)
   */
  Data* lookup(KeyTypeConstRef id)
  {
    return _lookup(id);
  }

  /*!
   * \brief Recherche la valeur correspondant à la clé \a id.
   *
   * \return la structure associé à la clé \a id (0 si aucune)
   */
  const Data* lookup(KeyTypeConstRef id) const
  {
    return _lookup(id);
  }

  /*!
   * \brief Recherche la valeur correspondant à la clé \a id.
   *
   * Une exception est générée si la valeur n'est pas trouvé.
   */
  ValueType& lookupValue(KeyTypeConstRef id)
  {
    Data* ht = _lookup(id);
    if (!ht) {
      this->_throwNotFound(id, Printable());
    }
    return ht->value();
  }

  /*!
   * \brief Recherche la valeur correspondant à la clé \a id.
   *
   * Une exception est générée si la valeur n'est pas trouvé.
   */
  ValueType& operator[](KeyTypeConstRef id)
  {
    return lookupValue(id);
  }

  /*!
   * \brief Recherche la valeur correspondant à la clé \a id.
   *
   * Une exception est générée si la valeur n'est pas trouvé.
   */
  const ValueType& lookupValue(KeyTypeConstRef id) const
  {
    const Data* ht = _lookup(id);
    if (!ht) {
      this->_throwNotFound(id, Printable());
    }
    return ht->m_value;
  }

  /*!
   * \brief Recherche la valeur correspondant à la clé \a id.
   *
   * Une exception est générée si la valeur n'est pas trouvé.
   */
  const ValueType& operator[](KeyTypeConstRef id) const
  {
    return lookupValue(id);
  }

  /*!
   * \brief Ajoute la valeur \a value correspondant à la clé \a id
   *
   * Si une valeur correspondant à \a id existe déjà, elle est remplacée.
   *
   * \retval true si la clé est ajoutée
   * \retval false si la clé existe déjà et est remplacée
   */
  bool add(KeyTypeConstRef id, const ValueType& value)
  {
    Integer hf = _keyToBucket(id);
    Data* ht = _lookupBucket(hf, id);
    if (ht) {
      ht->m_value = value;
      return false;
    }
    _add(hf, id, value);
    _checkResize();
    return true;
  }

  /*!
   * \brief Supprime la valeur associée à la clé \a id
   */
  void remove(KeyTypeConstRef id)
  {
    Integer hf = _keyToBucket(id);
    Data* ht = _removeBucket(hf, id);
    ht->setNext(m_first_free);
    m_first_free = ht;
  }

  /*!
   * \brief Recherche ou ajoute la valeur correspondant à la clé \a id.
   * 
   * Si la clé \a id est déjà dans la table, retourne une référence sur cette
   * valeur et positionne \a is_add à \c false. Sinon, ajoute la clé \a id
   * avec pour valeur \a value et positionne \a is_add à \c true.
   *
   * La structure retournée n'est jamais nul et peut être conservée car elle
   * ne change pas d'adresse tant que cette instance de la table de hachage existe
   */
  Data* lookupAdd(KeyTypeConstRef id, const ValueType& value, bool& is_add)
  {
    HashValueType hf = _applyHash(id);
    Data* ht = _lookupBucket(_hashValueToBucket(hf), id);
    if (ht) {
      is_add = false;
      return ht;
    }
    is_add = true;
    // Toujours faire le resize avant de retourner le add
    // car cela peut invalider le Data*
    _checkResize();
    ht = _add(_hashValueToBucket(hf), id, value);
    return ht;
  }

  /*!
   * \brief Recherche ou ajoute la valeur correspondant à la clé \a id.
   *
   * Si la clé \a id est déjà dans la table, retourne une référence sur cette
   * valeur et positionne \a is_add à \c false. Sinon, ajoute la clé \a id
   * avec pour valeur \a ValueType() (qui doit exister).
   * 
   * La structure retournée n'est jamais nul et peut être conservée car elle
   * ne change pas d'adresse tant que cette instance de la table de hachage existe
   */
  Data* lookupAdd(KeyTypeConstRef id)
  {
    HashValueType hf = _applyHash(id);
    Data* ht = _lookupBucket(_hashValueToBucket(hf), id);
    if (!ht) {
      // Toujours faire le resize avant de retourner le add
      // car cela peut invalider le Data*
      _checkResize();
      // Le resize provoque change le bucket associé à une clé
      ht = _add(_hashValueToBucket(hf), id, ValueType());
    }
    return ht;
  }

  /*!
   * \brief Ajoute la valeur \a value correspondant à la clé \a id
   *
   * Si une valeur correspondant à \a id existe déjà, le résultat est
   * indéfini.
   */
  void nocheckAdd(KeyTypeConstRef id, const ValueType& value)
  {
    _checkResize();
    Integer hf = _keyToBucket(id);
    _add(hf, id, value);
  }

  ArrayView<Data*> buckets()
  {
    return m_buckets;
  }

  ConstArrayView<Data*> buckets() const
  {
    return m_buckets;
  }

  //! Redimensionne la table de hachage
  void resize(Integer new_size, bool use_prime = false)
  {
    if (use_prime)
      new_size = this->nearestPrimeNumber(new_size);
    if (new_size == 0) {
      m_nb_bucket = new_size;
      clear();
      return;
    }
    if (new_size == m_nb_bucket)
      return;
    _rehash(new_size);
  }

  //! Repositionne les données après changement de valeur des clés
  void rehash()
  {
    _rehash(m_nb_bucket);
  }

 public:

  //! Applique le fonctor \a f à tous les éléments de la collection
  template <class Lambda> void
  each(const Lambda& lambda)
  {
    for (Integer k = 0, n = m_buckets.size(); k < n; ++k) {
      Data* nbid = m_buckets[k];
      for (; nbid; nbid = nbid->next()) {
        lambda(nbid);
      }
    }
  }

  /*!
   * \brief Applique le fonctor \a f à tous les éléments de la collection
   * et utilise x->value() (de type ValueType) comme argument.
   */
  template <class Lambda> void
  eachValue(const Lambda& lambda)
  {
    for (Integer k = 0, n = m_buckets.size(); k < n; ++k) {
      Data* nbid = m_buckets[k];
      for (; nbid; nbid = nbid->next()) {
        lambda(nbid->value());
      }
    }
  }

 private:

  //! Repositionne les données après changement de valeur des clés
  void _rehash(Integer new_size)
  {
    //todo: supprimer l'allocation de ce tableau
    UniqueArray<Data*> old_buckets(m_buckets);
    m_count = 0;
    m_nb_bucket = new_size;
    m_buckets.resize(new_size);
    m_buckets.fill(0);
    MultiBufferT<Data>* old_buffer = m_buffer;
    m_first_free = 0;
    m_buffer = new MultiBufferT<Data>(m_nb_bucket);
    for (Integer z = 0, zs = old_buckets.size(); z < zs; ++z) {
      for (Data* i = old_buckets[z]; i; i = i->next()) {
        Data* current = i;
        {
          _add(_keyToBucket(current->key()), current->key(), current->value());
          //Data* new_data = m_buffer->allocOne();
          //new_data->setValue(current->value());
          //_baseAdd(_hash(current->key()),current->key(),new_data);
        }
      }
    }
    delete old_buffer;
    _computeMaxCount();
  }

 private:

  MultiBufferT<Data>* m_buffer; //!< Tampon d'allocation des valeurs
  Data* m_first_free = nullptr; //!< Pointeur vers le premier Data utilisable

 public:

  mutable Int64 m_nb_collision = 0;
  mutable Int64 m_nb_direct = 0;

 private:

  Data* _add(Integer bucket, KeyTypeConstRef key, const ValueType& value)
  {
    Data* hd = 0;
    if (m_first_free) {
      hd = m_first_free;
      m_first_free = m_first_free->next();
    }
    else
      hd = m_buffer->allocOne();
    hd->setValue(value);
    _baseAdd(bucket, key, hd);
    return hd;
  }

  HashValueType _applyHash(KeyTypeConstRef id) const
  {
    //return (Integer)(KeyTraitsType::hashFunction(id) % m_nb_bucket);
    return KeyTraitsType::hashFunction(id);
  }

  Integer _keyToBucket(KeyTypeConstRef id) const
  {
    return (Integer)(_applyHash(id) % m_nb_bucket);
  }

  Integer _hashValueToBucket(KeyTypeValue id) const
  {
    return (Integer)(id % m_nb_bucket);
  }

  Data* _baseLookupBucket(Integer bucket, KeyTypeConstRef id) const
  {
    for (Data* i = m_buckets[bucket]; i; i = i->next()) {
      if (!(i->key() == id)) {
        ++m_nb_collision;
        continue;
      }
      ++m_nb_direct;
      return i;
    }
    return 0;
  }

  Data* _baseRemoveBucket(Integer bucket, KeyTypeConstRef id)
  {
    Data* i = m_buckets[bucket];
    if (i) {
      if (i->m_key == id) {
        m_buckets[bucket] = i->next();
        --m_count;
        return i;
      }
      for (; i->next(); i = i->next()) {
        if (i->next()->key() == id) {
          Data* r = i->next();
          i->setNext(i->next()->next());
          --m_count;
          return r;
        }
      }
    }
    this->_throwNotFound(id, Printable());
    return 0;
  }

  inline Data* _baseLookup(KeyTypeConstRef id) const
  {
    return _baseLookupBucket(_keyToBucket(id), id);
  }

  inline Data* _baseRemove(KeyTypeConstRef id)
  {
    return _baseRemoveBucket(_keyToBucket(id), id);
  }

  void _baseAdd(Integer bucket, KeyTypeConstRef id, Data* hd)
  {
    Data* buck = m_buckets[bucket];
    hd->m_key = id;
    hd->m_next = buck;
    m_buckets[bucket] = hd;
    ++m_count;
  }

  Data* _lookup(KeyTypeConstRef id)
  {
    return _baseLookup(id);
  }

  const Data* _lookup(KeyTypeConstRef id) const
  {
    return _baseLookup(id);
  }

  Data* _lookupBucket(Integer bucket, KeyTypeConstRef id) const
  {
    return _baseLookupBucket(bucket, id);
  }

  Data* _removeBucket(Integer bucket, KeyTypeConstRef id)
  {
    return _baseRemoveBucket(bucket, id);
  }

  void _checkResize()
  {
    // Retaille si besoin.
    if (m_count > m_max_count) {
      //cout << "** BEFORE BUCKET RESIZE this=" << this << " count=" << m_count
      //     << " bucket=" << m_nb_bucket << " m_max_count=" << m_max_count
      //     << " memory=" << (m_buckets.capacity()*sizeof(Data*)) << '\n';
      //_print(Printable());
      // Pour les grosses tables, augmente moins vite pour limiter la
      // consommation memoire
      if (m_nb_bucket > 200000) {
        resize((Integer)(1.3 * m_nb_bucket), true);
      }
      else if (m_nb_bucket > 10000) {
        resize((Integer)(1.5 * m_nb_bucket), true);
      }
      else
        resize(2 * m_nb_bucket, true);
      //cout << "** AFTER BUCKET RESIZE this=" << this << " count=" << m_count
      //     << " bucket=" << m_nb_bucket  << " m_max_count=" << m_max_count
      //     << " memory=" << (m_buckets.capacity()*sizeof(Data*)) << '\n';
      //_print(Printable());
      std::cout.flush();
    }
  }

  void _print(FalseType)
  {
  }

  void _print(TrueType)
  {
    for (Integer z = 0, zs = m_buckets.size(); z < zs; ++z) {
      for (Data* i = m_buckets[z]; i; i = i->next()) {
        cout << "* KEY=" << i->key() << " bucket=" << z << '\n';
      }
    }
  }

  void _throwNotFound ARCANE_NORETURN(KeyTypeConstRef, FalseType) const
  {
    HashTableBase::_throwNotFound();
  }

  void _throwNotFound ARCANE_NORETURN(KeyTypeConstRef id, TrueType) const
  {
    std::cout << "ERROR: can not find key=" << id << " bucket=" << _keyToBucket(id) << "\n";
    std::cout.flush();
    HashTableBase::_throwNotFound();
  }

  void _computeMaxCount()
  {
    m_max_count = (Integer)(m_nb_bucket * 0.85);
  }

 private:

  //! Nombre maximal d'élément avant retaillage
  Integer m_max_count = 0;
  UniqueArray<Data*> m_buckets; //! Tableau des buckets
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * \brief Enumerateur sur un HashTableMap.
 */
template <typename KeyType, typename ValueType>
class HashTableMapEnumeratorT
{
  typedef HashTableMapT<KeyType, ValueType> HashType;
  typedef typename HashType::Data Data;

 public:

  HashTableMapEnumeratorT(const HashType& rhs)
  : m_buckets(rhs.buckets())
  , m_current_data(0)
  , m_current_bucket(-1)
  {}

 public:

  bool operator++()
  {
    if (m_current_data)
      m_current_data = m_current_data->next();
    if (!m_current_data) {
      while (m_current_data == 0 && (m_current_bucket + 1) < m_buckets.size()) {
        ++m_current_bucket;
        m_current_data = m_buckets[m_current_bucket];
      }
    }
    return m_current_data != 0;
  }
  ValueType& operator*() { return m_current_data->value(); }
  const ValueType& operator*() const { return m_current_data->value(); }
  Data* data() { return m_current_data; }
  const Data* data() const { return m_current_data; }

 public:

  ConstArrayView<Data*> m_buckets;
  Data* m_current_data;
  Integer m_current_bucket;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
