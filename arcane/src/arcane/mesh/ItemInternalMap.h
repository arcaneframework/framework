// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ItemInternalMap.h                                           (C) 2000-2024 */
/*                                                                           */
/* Tableau associatif de ItemInternal.                                       */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_MESH_ITEMINTERNALMAP_H
#define ARCANE_MESH_ITEMINTERNALMAP_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/HashTableMap.h"
#include "arcane/utils/CheckedConvert.h"

#include "arcane/mesh/MeshGlobal.h"
#include "arcane/core/ItemInternal.h"

#include <unordered_map>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

// Indique si on utilise la nouvelle implémentation pour ItemInternalMap
// #define ARCANE_USE_NEW_IMPL_FOR_ITEMINTERNALMAP

namespace Arcane
{
class ItemInternal;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::mesh
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Tableau associatif de ItemInternal.
 *
 * Cette classe est interne à Arcane.
 *
 * La clé de ce tableau associatif est le UniqueId des entités.
 * S'il change, il faut appeler notifyUniqueIdsChanged() pour remettre
 * à jour le tableau associatif.
 *
 * \note Toutes les méthodes qui utilisent ou retournent un 'ItemInternal*'
 * sont obsolètes et ne doivent pas être utilisées.
 */
class ItemInternalMap
{
  // Pour accès aux méthodes qui utilisent ItemInternal.
  friend class DynamicMeshKindInfos;

 private:

  using LegacyImpl = HashTableMapT<Int64, ItemInternal*>;
  using NewImpl = std::unordered_map<Int64, ItemInternal*>;
  using BaseData = LegacyImpl::Data;

#ifdef ARCANE_USE_NEW_IMPL_FOR_ITEMINTERNALMAP
  static constexpr bool UseNewImpl = 1;
#else
  static constexpr bool UseNewImpl = 0;
#endif

 public:

  class LookupData
  {
    friend ItemInternalMap;

   public:

    void setValue(ItemInternal* v)
    {
      m_value = v;
      if (m_legacy_data)
        m_legacy_data->setValue(v);
      else
        m_iter->second = v;
    }
    ItemInternal* value() const { return m_value; }

   private:

    explicit LookupData(NewImpl::iterator x)
    : m_iter(x)
    , m_value(x->second)
    {}
    explicit LookupData(BaseData* d)
    : m_legacy_data(d)
    , m_value(d->value())
    {}
    NewImpl::iterator m_iter;
    BaseData* m_legacy_data = nullptr;
    ItemInternal* m_value;
  };

 public:

  using Data ARCANE_DEPRECATED_REASON("Y2024: Data type is internal to Arcane") = LegacyImpl::Data;

 public:

  using ValueType = ItemInternal*;

 public:

  ItemInternalMap();

 public:

  /*!
   * \brief Ajoute la valeur \a v correspondant à la clé \a key
   *
   * Si une valeur correspondant à \a id existe déjà, elle est remplacée.
   *
   * \retval true si la clé est ajoutée
   * \retval false si la clé existe déjà et est remplacée
   */
  bool add(Int64 key, ItemInternal* v)
  {
    if constexpr (UseNewImpl)
      return m_new_impl.insert(std::make_pair(key, v)).second;
    else
      return m_impl.add(key, v);
  }

  //! Supprime tous les éléments de la table
  void clear()
  {
    if constexpr (UseNewImpl)
      return m_new_impl.clear();
    else
      return m_impl.clear();
  }

  //! Nombre d'éléments de la table
  Int32 count() const
  {
    if constexpr (UseNewImpl)
      return CheckedConvert::toInt32(m_new_impl.size());
    else
      return m_impl.count();
  }

  /*!
   * \brief Supprime la valeur associée à la clé \a key
   *
   * Lève une exception s'il n'y a pas de valeurs associées à la clé
   */
  void remove(Int64 key)
  {
    if constexpr (UseNewImpl) {
      auto x = m_new_impl.find(key);
      if (x == m_new_impl.end())
        _throwNotFound(key);
      m_new_impl.erase(x);
    }
    else
      m_impl.remove(key);
  }

  //! \a true si une valeur avec la clé \a id est présente
  bool hasKey(Int64 key)
  {
    if constexpr (UseNewImpl)
      return (m_new_impl.find(key) != m_new_impl.end());
    else
      return m_impl.hasKey(key);
  }

  //! Redimensionne la table de hachage
  void resize(Int32 new_size, bool use_prime = false)
  {
    if constexpr (UseNewImpl) {
      // Nothing to do
    }
    else
      m_impl.resize(new_size, use_prime);
  }

  /*!
   * \brief Notifie que les numéros uniques des entités ont changés.
   *
   * Cet appel peut provoquer un recalcul complet du tableau associatif.
   */
  void notifyUniqueIdsChanged();

  /*!
   * \brief Fonction template pour itérer sur les entités de l'instance.
   *
   * Le type de l'arguments template peut-être n'importe quel type d'entité
   * qui peut être construit à partir d'un impl::ItemBase.
   * \code
   * ItemInternalMap item_map = ...
   * item_map.eachItem([&](Item item){
   *   std::cout << "LID=" << item_base.localId() << "\n";
   * });
   * \endcode
   */
  template <class Lambda> void
  eachItem(const Lambda& lambda)
  {
    if constexpr (UseNewImpl) {
      for (auto [key, value] : m_new_impl)
        lambda(Arcane::impl::ItemBase(value));
    }
    else {
      ConstArrayView<BaseData*> b = m_impl.buckets();
      for (Int32 k = 0, n = b.size(); k < n; ++k) {
        BaseData* nbid = b[k];
        for (; nbid; nbid = nbid->next()) {
          lambda(Arcane::impl::ItemBase(nbid->value()));
        }
      }
    }
  }
  //! Nombre de buckets
  Int32 nbBucket() const
  {
    if constexpr (UseNewImpl)
      return CheckedConvert::toInt32(m_new_impl.bucket_count());
    else
      return m_impl.buckets().size();
  }

 public:

  //! Retourne l'entité associée à \a key si trouvé ou l'entité nulle sinon
  impl::ItemBase tryFind(Int64 key) const
  {
    if constexpr (UseNewImpl) {
      auto x = m_new_impl.find(key);
      return (x != m_new_impl.end()) ? x->second : impl::ItemBase{};
    }
    else {
      const BaseData* d = m_impl.lookup(key);
      return (d ? impl::ItemBase(d->value()) : impl::ItemBase{});
    }
  }
  //! Retourne le localId() associé à \a key si trouvé ou NULL_ITEM_LOCAL_ID sinon aucun
  Int32 tryFindLocalId(Int64 key) const
  {
    if constexpr (UseNewImpl) {
      auto x = m_new_impl.find(key);
      return (x != m_new_impl.end()) ? x->second->localId() : NULL_ITEM_LOCAL_ID;
    }
    else {
      const BaseData* d = m_impl.lookup(key);
      return (d ? d->value()->localId() : NULL_ITEM_LOCAL_ID);
    }
  }

  /*!
   * \brief Retourne l'entité de numéro unique \a uid.
   *
   * Lève une exception si l'entité n'est pas dans la table.
   */
  impl::ItemBase findItem(Int64 uid) const
  {
    if constexpr (UseNewImpl) {
      auto x = m_new_impl.find(uid);
      if (x == m_new_impl.end())
        _throwNotFound(uid);
      return x->second;
    }
    else
      return impl::ItemBase(m_impl.lookupValue(uid));
  }

  /*!
   * \brief Retourne le numéro local de l'entité de numéro unique \a uid.
   *
   * Lève une exception si l'entité n'est pas dans la table.
   */
  Int32 findLocalId(Int64 uid) const
  {
    if constexpr (UseNewImpl) {
      auto x = m_new_impl.find(uid);
      if (x == m_new_impl.end())
        _throwNotFound(uid);
      return x->second->localId();
    }
    else
      return m_impl.lookupValue(uid)->localId();
  }

  void checkValid() const;

 public:

  ARCANE_DEPRECATED_REASON("Y2024: This method is internal to Arcane")
  Data* lookup(Int64 key)
  {
    if constexpr (UseNewImpl) {
      _throwNotSupported("lookup");
    }
    else
      return m_impl.lookup(key);
  }

  ARCANE_DEPRECATED_REASON("Y2024: This method is internal to Arcane")
  const Data* lookup(Int64 key) const
  {
    if constexpr (UseNewImpl) {
      _throwNotSupported("lookup");
    }
    else
      return m_impl.lookup(key);
  }

  ARCANE_DEPRECATED_REASON("Y2024: This method is internal to Arcane")
  ConstArrayView<BaseData*> buckets() const
  {
    if constexpr (UseNewImpl) {
      _throwNotSupported("buckets");
    }
    else
      return m_impl.buckets();
  }

  ARCANE_DEPRECATED_REASON("This method is internal to Arcane")
  BaseData* lookupAdd(Int64 id, ItemInternal* value, bool& is_add)
  {
    if constexpr (UseNewImpl) {
      _throwNotSupported("lookupAdd(id,value,is_add)");
    }
    else
      return m_impl.lookupAdd(id, value, is_add);
  }

  ARCANE_DEPRECATED_REASON("Y2024: This method is internal to Arcane")
  BaseData* lookupAdd(Int64 uid)
  {
    if constexpr (UseNewImpl) {
      _throwNotSupported("lookupAdd(uid)");
    }
    else
      return m_impl.lookupAdd(uid);
  }

  ARCANE_DEPRECATED_REASON("Y2024: Use findItem() instead")
  ItemInternal* lookupValue(Int64 uid) const
  {
    if constexpr (UseNewImpl) {
      _throwNotSupported("lookupValue");
    }
    else
      return m_impl.lookupValue(uid);
  }

  ARCANE_DEPRECATED_REASON("Y2024: Use findItem() instead")
  ItemInternal* operator[](Int64 uid) const
  {
    if constexpr (UseNewImpl) {
      _throwNotSupported("operator[]");
    }
    else
      return m_impl.lookupValue(uid);
  }

 private:

  NewImpl m_new_impl;
  LegacyImpl m_impl;

 private:

  // Les trois méthodes suivantes sont uniquement pour la
  // classe DynamicMeshKindInfos.

  //! Change la valeurs des localId()
  void _changeLocalIds(ArrayView<ItemInternal*> items_internal,
                       ConstArrayView<Int32> old_to_new_local_ids);

  LookupData _lookupAdd(Int64 id, ItemInternal* value, bool& is_add)
  {
    if constexpr (UseNewImpl) {
      auto x = m_new_impl.insert(std::make_pair(id, value));
      is_add = x.second;
      return LookupData(x.first);
    }
    else
      return LookupData(m_impl.lookupAdd(id, value, is_add));
  }

  //! Retourne l'entité associée à \a key si trouvé ou nullptr sinon
  ItemInternal* _tryFindItemInternal(Int64 key) const
  {
    if constexpr (UseNewImpl) {
      auto x = m_new_impl.find(key);
      if (x == m_new_impl.end())
        return nullptr;
      _checkValid(key, x->second);
      return x->second;
    }
    else {
      const BaseData* d = m_impl.lookup(key);
      return (d ? d->value() : nullptr);
    }
  }

 private:

  void _throwNotFound ARCANE_NORETURN(Int64 id) const;
  void _throwNotSupported ARCANE_NORETURN(const char* func_name) const;
  void _checkValid(Int64 uid, ItemInternal* v) const;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Macro pour itérer sur les valeurs d'un ItemInternalMap.
 *
 * \deprecated Utiliser ItemInternalMap::eachItem() à la place.
 */
#define ENUMERATE_ITEM_INTERNAL_MAP_DATA(iter,item_list) \
for( auto __i__##iter : item_list .buckets() ) \
    for (auto* iter = __i__##iter; iter; iter = iter->next())

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  
