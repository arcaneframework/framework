// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* HashTable.h                                                 (C) 2000-2007 */
/*                                                                           */
/* Table de hachage.                                                         */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_UTILS_HASHTABLE_H
#define ARCANE_UTILS_HASHTABLE_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/MultiBuffer.h"
#include "arcane/utils/Array.h"
#include "arcane/utils/HashFunction.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * \brief Classe de base d'une table de hachage simple pour les entités

 \todo Ajouter des itérateurs pour cette collection et les classes
 dérivées
 */
class ARCANE_UTILS_EXPORT HashTableBase
{
 public:
  /*! \brief Crée une table de taille \a table_size
   *
   Si \a use_prime est vrai, utilise la fonction nearestPrimeNumber()
   pour avoir une taille de taille qui est un nombre premier.
  */  
  HashTableBase(Integer table_size,bool use_prime)
  : m_count(0), m_nb_bucket(use_prime ? nearestPrimeNumber(table_size) : table_size)
  {
  }
  virtual ~HashTableBase() {}
 public:
  /*! \brief Retourne le nombre premier le plus proche de \a n par excès.
   * Le nombre premier le plus proche et supérieur à \a n est renvoyé en utilisant une
   * table de nombre premier déterminée à l'avance.
   */
  Integer nearestPrimeNumber(Integer n);
 public:
  //! Nombre d'éléments dans la table
  Integer count() const
  {
    return m_count;
  }
 protected:
  void _throwNotFound ARCANE_NORETURN () const;
 protected:
  Integer m_count; //!< Nombre d'éléments
  Integer m_nb_bucket; //!< Nombre de buckets
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * \brief Classe de base d'une table de hachage pour tableaux associatifs
 
 Cette table permet de stocker une valeur en fonction d'une clé. Le
 type de la valeur est gérée par la classe dérivée HashTableMapT.
 
 La table de hachage est gérée sous forme d'un tableau dont le nombre
 d'éléments est donné par la taille de la table (m_nb_bucket).
 Les éléments sont ensuite rangés dans une liste chaînée.

 Cette table permet uniquement d'ajouter des valeurs.

 Pour des raisons de performance, il est préférable que le taille
 de la table (buckets) soit un nombre premier.
 */
template<typename KeyType,typename TraitsType>
class HashTableBaseT
: public HashTableBase
{
 public:
  typedef typename TraitsType::KeyTypeConstRef KeyTypeConstRef;
  typedef typename TraitsType::KeyTypeValue KeyTypeValue;
 public:
  struct HashData
  {
   public:
    friend class HashTableBaseT<KeyType,TraitsType>;
   public:
    HashData() : m_key(KeyType()), m_next(0) {}
   public:
    /*! \brief Change la valeur de la clé.
     *
     * Après avoir changé la valeur d'une ou plusieurs clés, il faut faire un rehash().
     */
    void changeKey(const KeyType& new_key)
      {
        m_key = new_key;
      }
   protected:
    KeyTypeValue m_key; //!< Clé de recherche
    HashData *m_next;  //! Elément suivant dans la table de hachage
  };
 public:
  /*! \brief Crée une table de taille \a table_size
   *
   Si \a use_prime est vrai, utilise la fonction nearestPrimeNumber()
   pour avoir une taille de taille qui est un nombre premier.
  */  
  HashTableBaseT(Integer table_size,bool use_prime)
  : HashTableBase(table_size,use_prime), m_buckets(m_nb_bucket)
  {
    m_buckets.fill(0);
  }

 public:
  //! \a true si une valeur avec la clé \a id est présente
  bool hasKey(KeyTypeConstRef id) const
  {
    KeyType hf = _hash(id);
    for( HashData* i = m_buckets[hf]; i; i=i->m_next ){
      if (i->m_key==id)
        return true;
    }
    return false;
  }

  //! Supprime tous les éléments de la table
  void clear()
  {
    m_buckets.fill(0);
  }

  //! Redimensionne la table de hachage
  void resize(Integer new_table_size,bool use_prime=false)
  {
    m_nb_bucket = new_table_size;
    if (new_table_size==0){
      clear();
      return;
    }
    if (use_prime)
      new_table_size = nearestPrimeNumber(new_table_size);
    //todo: supprimer l'allocation de ce tableau
    UniqueArray<HashData*> old_buckets(m_buckets.clone());
    m_buckets.resize(new_table_size);
    m_buckets.fill(0);
    for( Integer z=0, zs=old_buckets.size(); z<zs; ++z ){
      for( HashData* i = old_buckets[z]; i; i=i->m_next ){
        _baseAdd(_hash(i->m_key),i->m_key,i);
      }
    }
  }

  //! Repositionne les données après changement de valeur des clés
  void rehash()
  {
    //todo: supprimer l'allocation de ce tableau
    UniqueArray<HashData*> old_buckets(m_buckets.clone());
    m_buckets.fill(0);
    
    for( Integer z=0, zs=old_buckets.size(); z<zs; ++z ){
      for( HashData* i = old_buckets[z]; i; ){
        HashData* current = i;
        i = i->m_next; // Il faut le faire ici, car i->m_next change avec _baseAdd
        _baseAdd(_hash(current->m_key),current->m_key,current);
      }
    }
  }

 protected:

  inline Integer _hash(KeyTypeConstRef id) const
  {
    return TraitsType::hashFunction(id) % m_nb_bucket;
  }
  inline HashData* _baseLookupBucket(Integer bucket,KeyTypeConstRef id) const
  {
    for( HashData* i = m_buckets[bucket]; i; i=i->m_next ){
      if (i->m_key==id)
        return i;
    }
    return 0;
  }
  inline HashData* _baseRemoveBucket(Integer bucket,KeyTypeConstRef id)
  {
    HashData* i = m_buckets[bucket];
    if (i){
      if (i->m_key==id){
        m_buckets[bucket] = i->m_next;
        --m_count;
        return i;
      }
      for( ; i->m_next; i=i->m_next ){
        if (i->m_next->m_key==id){
          HashData* r = i->m_next;
          i->m_next = i->m_next->m_next;
          --m_count;
          return r;
        }
      }
    }
    _throwNotFound();
  }
  inline HashData* _baseLookup(KeyTypeConstRef id) const
  {
    return _baseLookupBucket(_hash(id),id);
  }
  inline HashData* _baseRemove(KeyTypeConstRef id)
  {
    return _baseRemoveBucket(_hash(id),id);
  }
  inline void _baseAdd(Integer bucket,KeyTypeConstRef id,HashData* hd)
  {
    HashData* buck = m_buckets[bucket];
    hd->m_key = id;
    hd->m_next = buck;
    m_buckets[bucket] = hd;
    ++m_count;
  }

 protected:
  UniqueArray<HashData*> m_buckets; //! Tableau des buckets
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
