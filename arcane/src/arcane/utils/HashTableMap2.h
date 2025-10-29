// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
#ifndef ARCANE_UTILS_HASHTABLEMAP2_H
#define ARCANE_UTILS_HASHTABLEMAP2_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/UtilsTypes.h"
#include "arccore/common/AllocatedMemoryInfo.h"
#include "arccore/common/IMemoryAllocator.h"

// Version initiale issue du commit bdebddbdce1b473bbc189178fd523ef4a876ea01 (27 aout 2024)
// emhash8::HashMap for C++14/17
// version 1.6.5
// https://github.com/ktprime/emhash/blob/master/hash_table8.hpp
//
// Licensed under the MIT License <http://opensource.org/licenses/MIT>.
// SPDX-License-Identifier: MIT
// Copyright (c) 2021-2024 Huang Yuanbing & bailuzhou AT 163.com
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE

#include <cstring>
#include <string>
#include <cstdlib>
#include <type_traits>
#include <cassert>
#include <utility>
#include <cstdint>
#include <functional>
#include <iterator>
#include <algorithm>
#include <memory>

#undef EMH_NEW
#undef EMH_EMPTY

// likely/unlikely
#if defined(__GNUC__) || defined(__INTEL_COMPILER) || defined(__clang__)
#define EMH_LIKELY(condition) __builtin_expect(condition, 1)
#define EMH_UNLIKELY(condition) __builtin_expect(condition, 0)
#else
#define EMH_LIKELY(condition) condition
#define EMH_UNLIKELY(condition) condition
#endif

#define EMH_EMPTY(n) (0 > (int)(m_index[n].next))
#define EMH_EQHASH(n, key_hash) (((size_type)(key_hash) & ~m_mask) == (m_index[n].slot & ~m_mask))
#define EMH_NEW(key, val, bucket, key_hash) \
  new (m_pairs + m_num_filled) value_type(key, val); \
  m_etail = bucket; \
  m_index[bucket] = { bucket, m_num_filled++ | ((size_type)(key_hash) & ~m_mask) }

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::impl
{
//! Base class for HashTableMap2
class ARCANE_UTILS_EXPORT HashTableMap2Base
{
 public:

  using size_type = uint32_t;

  struct Index
  {
    size_type next;
    size_type slot;
  };

 protected:

  constexpr static size_type EAD = 2;

 protected:

  Index* m_index = nullptr;
  uint32_t m_mlf = 0;
  size_type m_mask = 0;
  size_type m_num_buckets = 0;
  size_type m_num_filled = 0;
  size_type m_last = 0;
  size_type m_etail = 0;
  IMemoryAllocator* m_memory_allocator = _defaultAllocator();

 private:

  Int64 m_index_allocated_size = 0;

 protected:

  void _allocIndex(size_type num_buckets)
  {
    m_index_allocated_size = (uint64_t)(EAD + num_buckets) * sizeof(Index);
    AllocatedMemoryInfo mem_info = m_memory_allocator->allocate({}, m_index_allocated_size);
    m_index = reinterpret_cast<Index*>(mem_info.baseAddress());
  }
  void _freeIndex()
  {
    m_memory_allocator->deallocate({}, { m_index, m_index_allocated_size });
    m_index = nullptr;
    m_index_allocated_size = 0;
  }

  void _doSwap(HashTableMap2Base& rhs)
  {
    std::swap(m_index, rhs.m_index);
    std::swap(m_num_buckets, rhs.m_num_buckets);
    std::swap(m_num_filled, rhs.m_num_filled);
    std::swap(m_mask, rhs.m_mask);
    std::swap(m_mlf, rhs.m_mlf);
    std::swap(m_last, rhs.m_last);
    std::swap(m_etail, rhs.m_etail);
    std::swap(m_index_allocated_size, rhs.m_index_allocated_size);
    std::swap(m_memory_allocator, rhs.m_memory_allocator);
  }

  void _doClone(const HashTableMap2Base& rhs)
  {
    m_num_buckets = rhs.m_num_buckets;
    m_num_filled = rhs.m_num_filled;
    m_mlf = rhs.m_mlf;
    m_last = rhs.m_last;
    m_mask = rhs.m_mask;
    m_etail = rhs.m_etail;
    m_index_allocated_size = rhs.m_index_allocated_size;
    m_memory_allocator = rhs.m_memory_allocator;
  };

 private:

  static IMemoryAllocator* _defaultAllocator();
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Implementation of std::unordered_map.
 *
 * \warning This class is experimental and internal to Arcane.
 */
template <typename KeyT, typename ValueT,
          typename HashT = std::hash<KeyT>,
          typename EqT = std::equal_to<KeyT>>
class HashTableMap2
: public HashTableMap2Base
{
  constexpr static float EMH_DEFAULT_LOAD_FACTOR = 0.80f;
  constexpr static float EMH_MIN_LOAD_FACTOR = 0.25f; //< 0.5
  constexpr static uint32_t EMH_CACHE_LINE_SIZE = 64; //debug only

 public:

  using htype = HashTableMap2<KeyT, ValueT, HashT, EqT>;
  using value_type = std::pair<KeyT, ValueT>;
  using key_type = KeyT;
  using mapped_type = ValueT;
  using hasher = HashT;
  using key_equal = EqT;

  constexpr static size_type INACTIVE = 0xFFFFFFFF;
  constexpr static size_type END = 0xFFFFFFFF;

  class const_iterator;
  class iterator
  {
   public:

    using iterator_category = std::bidirectional_iterator_tag;
    using difference_type = std::ptrdiff_t;
    using value_type = typename htype::value_type;
    using pointer = value_type*;
    using const_pointer = const value_type*;
    using reference = value_type&;
    using const_reference = const value_type&;

    iterator()
    : kv_(nullptr)
    {}
    iterator(const_iterator& cit)
    {
      kv_ = cit.kv_;
    }

    iterator(const htype* hash_map, size_type bucket)
    {
      kv_ = hash_map->m_pairs + (int)bucket;
    }

    iterator& operator++()
    {
      kv_++;
      return *this;
    }

    iterator operator++(int)
    {
      auto cur = *this;
      kv_++;
      return cur;
    }

    iterator& operator--()
    {
      kv_--;
      return *this;
    }

    iterator operator--(int)
    {
      auto cur = *this;
      kv_--;
      return cur;
    }

    reference operator*() const { return *kv_; }
    pointer operator->() const { return kv_; }

    bool operator==(const iterator& rhs) const { return kv_ == rhs.kv_; }
    bool operator!=(const iterator& rhs) const { return kv_ != rhs.kv_; }
    bool operator==(const const_iterator& rhs) const { return kv_ == rhs.kv_; }
    bool operator!=(const const_iterator& rhs) const { return kv_ != rhs.kv_; }

   public:

    value_type* kv_;
  };

  class const_iterator
  {
   public:

    using iterator_category = std::bidirectional_iterator_tag;
    using value_type = typename htype::value_type;
    using difference_type = std::ptrdiff_t;
    using pointer = value_type*;
    using const_pointer = const value_type*;
    using reference = value_type&;
    using const_reference = const value_type&;

    const_iterator(const iterator& it)
    {
      kv_ = it.kv_;
    }

    const_iterator(const htype* hash_map, size_type bucket)
    {
      kv_ = hash_map->m_pairs + (int)bucket;
    }

    const_iterator& operator++()
    {
      kv_++;
      return *this;
    }

    const_iterator operator++(int)
    {
      auto cur = *this;
      kv_++;
      return cur;
    }

    const_iterator& operator--()
    {
      kv_--;
      return *this;
    }

    const_iterator operator--(int)
    {
      auto cur = *this;
      kv_--;
      return cur;
    }

    const_reference operator*() const { return *kv_; }
    const_pointer operator->() const { return kv_; }

    bool operator==(const iterator& rhs) const { return kv_ == rhs.kv_; }
    bool operator!=(const iterator& rhs) const { return kv_ != rhs.kv_; }
    bool operator==(const const_iterator& rhs) const { return kv_ == rhs.kv_; }
    bool operator!=(const const_iterator& rhs) const { return kv_ != rhs.kv_; }

   public:

    const value_type* kv_;
  };

  void init(size_type bucket, float mlf = EMH_DEFAULT_LOAD_FACTOR)
  {
    m_pairs = nullptr;
    m_index = nullptr;
    m_mask = m_num_buckets = 0;
    m_num_filled = 0;
    m_mlf = (uint32_t)((1 << 27) / EMH_DEFAULT_LOAD_FACTOR);
    max_load_factor(mlf);
    rehash(bucket);
  }

  HashTableMap2(size_type bucket = 2, float mlf = EMH_DEFAULT_LOAD_FACTOR)
  {
    init(bucket, mlf);
  }

  HashTableMap2(const HashTableMap2& rhs)
  {
    if (rhs.load_factor() > EMH_MIN_LOAD_FACTOR) {
      m_pairs = _allocBucket((size_type)(rhs.m_num_buckets * rhs.max_load_factor()) + 4);
      _allocIndex(rhs.m_num_buckets);
      clone(rhs);
    }
    else {
      init(rhs.m_num_filled + 2, rhs.max_load_factor());
      for (auto it = rhs.begin(); it != rhs.end(); ++it)
        insert_unique(it->first, it->second);
    }
  }

  HashTableMap2(HashTableMap2&& rhs) noexcept
  {
    init(0);
    *this = std::move(rhs);
  }

  HashTableMap2(std::initializer_list<value_type> ilist)
  {
    init((size_type)ilist.size());
    for (auto it = ilist.begin(); it != ilist.end(); ++it)
      _doInsert(*it);
  }

  template <class InputIt>
  HashTableMap2(InputIt first, InputIt last, size_type bucket_count = 4)
  {
    init(std::distance(first, last) + bucket_count);
    for (; first != last; ++first)
      emplace(*first);
  }

  HashTableMap2& operator=(const HashTableMap2& rhs)
  {
    if (this == &rhs)
      return *this;

    if (rhs.load_factor() < EMH_MIN_LOAD_FACTOR) {
      clear();
      _freeBuckets();
      rehash(rhs.m_num_filled + 2);
      for (auto it = rhs.begin(); it != rhs.end(); ++it)
        insert_unique(it->first, it->second);
      return *this;
    }

    clearkv();

    if (m_num_buckets != rhs.m_num_buckets) {
      _freeIndex();
      _freeBuckets();
      _allocIndex(rhs.m_num_buckets);
      m_pairs = _allocBucket((size_type)(rhs.m_num_buckets * rhs.max_load_factor()) + 4);
    }

    clone(rhs);
    return *this;
  }

  HashTableMap2& operator=(HashTableMap2&& rhs) noexcept
  {
    if (this != &rhs) {
      swap(rhs);
      rhs.clear();
    }
    return *this;
  }

  template <typename Con>
  bool operator==(const Con& rhs) const
  {
    if (size() != rhs.size())
      return false;

    for (auto it = begin(), last = end(); it != last; ++it) {
      auto oi = rhs.find(it->first);
      if (oi == rhs.end() || it->second != oi->second)
        return false;
    }
    return true;
  }

  template <typename Con>
  bool operator!=(const Con& rhs) const
  {
    return !(*this == rhs);
  }

  ~HashTableMap2() noexcept
  {
    clearkv();
    _freeBuckets();
    _freeIndex();
  }

  void clone(const HashTableMap2& rhs)
  {
    _doClone(rhs);
    m_hasher = rhs.m_hasher;
    m_pairs_allocated_size = rhs.m_pairs_allocated_size;

    auto opairs = rhs.m_pairs;
    memcpy((char*)m_index, (char*)rhs.m_index, (m_num_buckets + EAD) * sizeof(Index));

    if (is_copy_trivially()) {
      memcpy((char*)m_pairs, (char*)opairs, m_num_filled * sizeof(value_type));
    }
    else {
      for (size_type slot = 0; slot < m_num_filled; slot++)
        new (m_pairs + slot) value_type(opairs[slot]);
    }
  }

  void swap(HashTableMap2& rhs)
  {
    _doSwap(rhs);
    std::swap(m_hasher, rhs.m_hasher);
    std::swap(m_pairs, rhs.m_pairs);
    std::swap(m_pairs_allocated_size, rhs.m_pairs_allocated_size);
  }

  // -------------------------------------------------------------
  iterator first() const
  {
    return { this, 0 };
  }
  iterator last() const
  {
    return { this, m_num_filled - 1 };
  }

  value_type& front()
  {
    return m_pairs[0];
  }
  const value_type& front() const
  {
    return m_pairs[0];
  }
  value_type& back()
  {
    return m_pairs[m_num_filled - 1];
  }
  const value_type& back() const
  {
    return m_pairs[m_num_filled - 1];
  }

  void pop_front()
  {
    erase(begin());
  } //TODO. only erase first without move last
  void pop_back()
  {
    erase(last());
  }

  iterator begin()
  {
    return first();
  }
  const_iterator cbegin() const
  {
    return first();
  }
  const_iterator begin() const
  {
    return first();
  }

  iterator end()
  {
    return { this, m_num_filled };
  }
  const_iterator cend() const
  {
    return { this, m_num_filled };
  }
  const_iterator end() const
  {
    return cend();
  }

  const value_type* values() const
  {
    return m_pairs;
  }
  const Index* index() const
  {
    return m_index;
  }

  size_type size() const
  {
    return m_num_filled;
  }
  bool empty() const
  {
    return m_num_filled == 0;
  }
  size_type bucket_count() const
  {
    return m_num_buckets;
  }

  /// Returns average number of elements per bucket.
  double load_factor() const
  {
    return static_cast<double>(m_num_filled) / (m_mask + 1);
  }

  HashT& hash_function() const
  {
    return m_hasher;
  }
  EqT& key_eq() const
  {
    return m_eq;
  }

  void max_load_factor(double mlf)
  {
    if (mlf < 0.992 && mlf > EMH_MIN_LOAD_FACTOR) {
      m_mlf = (uint32_t)((1 << 27) / mlf);
      if (m_num_buckets > 0)
        rehash(m_num_buckets);
    }
  }

  constexpr double max_load_factor() const
  {
    return (1 << 27) / static_cast<double>(m_mlf);
  }
  constexpr size_type max_size() const
  {
    return (1ull << (sizeof(size_type) * 8 - 1));
  }
  constexpr size_type max_bucket_count() const
  {
    return max_size();
  }

  // ------------------------------------------------------------
  template <typename K = KeyT>
  iterator find(const K& key) noexcept
  {
    return { this, find_filled_slot(key) };
  }

  template <typename K = KeyT>
  const_iterator find(const K& key) const noexcept
  {
    return { this, find_filled_slot(key) };
  }

  template <typename K = KeyT>
  ValueT& at(const K& key)
  {
    const auto slot = find_filled_slot(key);
    //throw
    return m_pairs[slot].second;
  }

  template <typename K = KeyT>
  const ValueT& at(const K& key) const
  {
    const auto slot = find_filled_slot(key);
    //throw
    return m_pairs[slot].second;
  }

  const ValueT& index(const uint32_t index) const
  {
    return m_pairs[index].second;
  }

  ValueT& index(const uint32_t index)
  {
    return m_pairs[index].second;
  }

  template <typename K = KeyT>
  bool contains(const K& key) const noexcept
  {
    return find_filled_slot(key) != m_num_filled;
  }

  template <typename K = KeyT>
  size_type count(const K& key) const noexcept
  {
    return find_filled_slot(key) == m_num_filled ? 0 : 1;
  }

  template <typename K = KeyT>
  std::pair<iterator, iterator> equal_range(const K& key)
  {
    const auto found = find(key);
    if (found.second == m_num_filled)
      return { found, found };
    else
      return { found, std::next(found) };
  }

  void merge(HashTableMap2& rhs)
  {
    if (empty()) {
      *this = std::move(rhs);
      return;
    }

    for (auto rit = rhs.begin(); rit != rhs.end();) {
      auto fit = find(rit->first);
      if (fit == end()) {
        insert_unique(rit->first, std::move(rit->second));
        rit = rhs.erase(rit);
      }
      else {
        ++rit;
      }
    }
  }

  std::pair<iterator, bool> add(const KeyT& key, const ValueT& value)
  {
    return insert(std::make_pair(key, value));
  }

  std::pair<iterator, bool> insert(const value_type& p)
  {
    check_expand_need();
    return _doInsert(p);
  }

  std::pair<iterator, bool> insert(value_type&& p)
  {
    check_expand_need();
    return _doInsert(std::move(p));
  }

  void insert(std::initializer_list<value_type> ilist)
  {
    reserve(ilist.size() + m_num_filled, false);
    for (auto it = ilist.begin(); it != ilist.end(); ++it)
      _doInsert(*it);
  }

  template <typename Iter>
  void insert(Iter first, Iter last)
  {
    reserve(std::distance(first, last) + m_num_filled, false);
    for (; first != last; ++first)
      _doInsert(first->first, first->second);
  }

  template <class... Args>
  std::pair<iterator, bool> emplace(Args&&... args) noexcept
  {
    check_expand_need();
    return _doInsert(std::forward<Args>(args)...);
  }

  //no any optimize for position
  template <class... Args>
  iterator emplace_hint(const_iterator hint, Args&&... args)
  {
    (void)hint;
    check_expand_need();
    return _doInsert(std::forward<Args>(args)...).first;
  }

  template <class... Args>
  std::pair<iterator, bool> try_emplace(const KeyT& k, Args&&... args)
  {
    check_expand_need();
    return _doInsert(k, std::forward<Args>(args)...);
  }

  template <class... Args>
  std::pair<iterator, bool> try_emplace(KeyT&& k, Args&&... args)
  {
    check_expand_need();
    return _doInsert(std::move(k), std::forward<Args>(args)...);
  }

  template <class... Args>
  size_type emplace_unique(Args&&... args)
  {
    return insert_unique(std::forward<Args>(args)...);
  }

  std::pair<iterator, bool> insert_or_assign(const KeyT& key, ValueT&& val)
  {
    return do_assign(key, std::forward<ValueT>(val));
  }
  std::pair<iterator, bool> insert_or_assign(KeyT&& key, ValueT&& val)
  {
    return do_assign(std::move(key), std::forward<ValueT>(val));
  }

  /// Return the old value or ValueT() if it didn't exist.
  ValueT set_get(const KeyT& key, const ValueT& val)
  {
    check_expand_need();
    const auto key_hash = hash_key(key);
    const auto bucket = _findOrAllocate(key, key_hash);
    if (EMH_EMPTY(bucket)) {
      EMH_NEW(key, val, bucket, key_hash);
      return ValueT();
    }
    else {
      const auto slot = m_index[bucket].slot & m_mask;
      ValueT old_value(val);
      std::swap(m_pairs[slot].second, old_value);
      return old_value;
    }
  }

  /// Like std::map<KeyT, ValueT>::operator[].
  ValueT& operator[](const KeyT& key) noexcept
  {
    check_expand_need();
    const auto key_hash = hash_key(key);
    const auto bucket = _findOrAllocate(key, key_hash);
    if (EMH_EMPTY(bucket)) {
      /* Check if inserting a value rather than overwriting an old entry */
      EMH_NEW(key, std::move(ValueT()), bucket, key_hash);
    }

    const auto slot = m_index[bucket].slot & m_mask;
    return m_pairs[slot].second;
  }

  ValueT& operator[](KeyT&& key) noexcept
  {
    check_expand_need();
    const auto key_hash = hash_key(key);
    const auto bucket = _findOrAllocate(key, key_hash);
    if (EMH_EMPTY(bucket)) {
      EMH_NEW(std::move(key), std::move(ValueT()), bucket, key_hash);
    }

    const auto slot = m_index[bucket].slot & m_mask;
    return m_pairs[slot].second;
  }

  /// Erase an element from the hash table.
  /// return 0 if element was not found
  size_type erase(const KeyT& key) noexcept
  {
    const auto key_hash = hash_key(key);
    const auto sbucket = find_filled_bucket(key, key_hash);
    if (sbucket == INACTIVE)
      return 0;

    const auto main_bucket = key_hash & m_mask;
    erase_slot(sbucket, (size_type)main_bucket);
    return 1;
  }

  //iterator erase(const_iterator begin_it, const_iterator end_it)
  iterator erase(const const_iterator& cit) noexcept
  {
    const auto slot = (size_type)(cit.kv_ - m_pairs);
    size_type main_bucket;
    const auto sbucket = find_slot_bucket(slot, main_bucket); //TODO
    erase_slot(sbucket, main_bucket);
    return { this, slot };
  }

  //only last >= first
  iterator erase(const_iterator first, const_iterator last) noexcept
  {
    auto esize = long(last.kv_ - first.kv_);
    auto tsize = long((m_pairs + m_num_filled) - last.kv_); //last to tail size
    auto next = first;
    while (tsize-- > 0) {
      if (esize-- <= 0)
        break;
      next = ++erase(next);
    }

    //fast erase from last
    next = this->last();
    while (esize-- > 0)
      next = --erase(next);

    return { this, size_type(next.kv_ - m_pairs) };
  }

  template <typename Pred>
  size_type erase_if(Pred pred)
  {
    auto old_size = size();
    for (auto it = begin(); it != end();) {
      if (pred(*it))
        it = erase(it);
      else
        ++it;
    }
    return old_size - size();
  }

  static constexpr bool is_triviall_destructable()
  {
#if __cplusplus >= 201402L || _MSC_VER > 1600
    return !(std::is_trivially_destructible<KeyT>::value && std::is_trivially_destructible<ValueT>::value);
#else
    return !(std::is_pod<KeyT>::value && std::is_pod<ValueT>::value);
#endif
  }

  static constexpr bool is_copy_trivially()
  {
#if __cplusplus >= 201103L || _MSC_VER > 1600
    return (std::is_trivially_copyable<KeyT>::value && std::is_trivially_copyable<ValueT>::value);
#else
    return (std::is_pod<KeyT>::value && std::is_pod<ValueT>::value);
#endif
  }

  /// Remove all elements, keeping full capacity.
  void clear() noexcept
  {
    clearkv();

    if (m_num_filled > 0)
      memset((char*)m_index, INACTIVE, sizeof(m_index[0]) * m_num_buckets);

    m_last = m_num_filled = 0;
    m_etail = INACTIVE;

  }

  void shrink_to_fit(const float min_factor = EMH_DEFAULT_LOAD_FACTOR / 4)
  {
    if (load_factor() < min_factor && bucket_count() > 10) //safe guard
      rehash(m_num_filled + 1);
  }


  /// Make room for this many elements
  bool reserve(uint64_t num_elems, bool force)
  {
    (void)force;
    const auto required_buckets = num_elems * m_mlf >> 27;
    if (EMH_LIKELY(required_buckets < m_mask)) // && !force
      return false;

    //assert(required_buckets < max_size());
    rehash(required_buckets + 2);
    return true;
  }

  bool reserve(size_type required_buckets) noexcept
  {
    if (m_num_filled != required_buckets)
      return reserve(required_buckets, true);

    m_last = 0;

    memset((char*)m_index, INACTIVE, sizeof(m_index[0]) * m_num_buckets);
    for (size_type slot = 0; slot < m_num_filled; slot++) {
      const auto& key = m_pairs[slot].first;
      const auto key_hash = hash_key(key);
      const auto bucket = size_type(key_hash & m_mask);
      auto& next_bucket = m_index[bucket].next;
      if ((int)next_bucket < 0)
        m_index[bucket] = { 1, slot | ((size_type)(key_hash) & ~m_mask) };
      else {
        m_index[bucket].slot |= (size_type)(key_hash) & ~m_mask;
        next_bucket++;
      }
    }
    return true;
  }

  void rehash(uint64_t required_buckets)
  {
    if (required_buckets < m_num_filled)
      return;

    assert(required_buckets < max_size());
    auto num_buckets = m_num_filled > (1u << 16) ? (1u << 16) : 4u;
    while (num_buckets < required_buckets) {
      num_buckets *= 2;
    }
    m_last = 0;

    m_mask = num_buckets - 1;
#if EMH_PACK_TAIL > 1
    m_last = m_mask;
    num_buckets += num_buckets * EMH_PACK_TAIL / 100; //add more 5-10%
#endif
    m_num_buckets = num_buckets;

    rebuild(num_buckets);

    m_etail = INACTIVE;
    for (size_type slot = 0; slot < m_num_filled; ++slot) {
      const auto& key = m_pairs[slot].first;
      const auto key_hash = hash_key(key);
      const auto bucket = _findUniqueBucket(key_hash);
      m_index[bucket] = { bucket, slot | ((size_type)(key_hash) & ~m_mask) };
    }
  }

 private:

  void clearkv()
  {
    if (is_triviall_destructable()) {
      while (m_num_filled--)
        m_pairs[m_num_filled].~value_type();
    }
  }

  void rebuild(size_type num_buckets) noexcept
  {
    _freeIndex();
    auto new_pairs = _allocBucket((size_type)(num_buckets * max_load_factor()) + 4);
    if (is_copy_trivially()) {
      if (m_pairs)
        memcpy((char*)new_pairs, (char*)m_pairs, m_num_filled * sizeof(value_type));
    }
    else {
      for (size_type slot = 0; slot < m_num_filled; slot++) {
        new (new_pairs + slot) value_type(std::move(m_pairs[slot]));
        if (is_triviall_destructable())
          m_pairs[slot].~value_type();
      }
    }
    _freeBuckets();
    m_pairs = new_pairs;
    _allocIndex(num_buckets);

    memset((char*)m_index, INACTIVE, sizeof(m_index[0]) * num_buckets);
    memset((char*)(m_index + num_buckets), 0, sizeof(m_index[0]) * EAD);
  }

  void pack_zero(ValueT zero)
  {
    m_pairs[m_num_filled] = { KeyT(), zero };
  }

  /// Returns the matching ValueT or nullptr if k isn't found.
  bool try_get(const KeyT& key, ValueT& val) const noexcept
  {
    const auto slot = find_filled_slot(key);
    const auto found = slot != m_num_filled;
    if (found) {
      val = m_pairs[slot].second;
    }
    return found;
  }

  /// Returns the matching ValueT or nullptr if k isn't found.
  ValueT* try_get(const KeyT& key) noexcept
  {
    const auto slot = find_filled_slot(key);
    return slot != m_num_filled ? &m_pairs[slot].second : nullptr;
  }

  /// Const version of the above
  ValueT* try_get(const KeyT& key) const noexcept
  {
    const auto slot = find_filled_slot(key);
    return slot != m_num_filled ? &m_pairs[slot].second : nullptr;
  }

  /// set value if key exist
  bool try_set(const KeyT& key, const ValueT& val) noexcept
  {
    const auto slot = find_filled_slot(key);
    if (slot == m_num_filled)
      return false;

    m_pairs[slot].second = val;
    return true;
  }

  /// set value if key exist
  bool try_set(const KeyT& key, ValueT&& val) noexcept
  {
    const auto slot = find_filled_slot(key);
    if (slot == m_num_filled)
      return false;

    m_pairs[slot].second = std::move(val);
    return true;
  }

  /// Convenience function.
  ValueT get_or_return_default(const KeyT& key) const noexcept
  {
    const auto slot = find_filled_slot(key);
    return slot == m_num_filled ? ValueT() : m_pairs[slot].second;
  }

  // -----------------------------------------------------
  std::pair<iterator, bool> _doInsert(const value_type& value) noexcept
  {
    const auto key_hash = hash_key(value.first);
    const auto bucket = _findOrAllocate(value.first, key_hash);
    const auto bempty = EMH_EMPTY(bucket);
    if (bempty) {
      EMH_NEW(value.first, value.second, bucket, key_hash);
    }

    const auto slot = m_index[bucket].slot & m_mask;
    return { { this, slot }, bempty };
  }

  std::pair<iterator, bool> _doInsert(value_type&& value) noexcept
  {
    const auto key_hash = hash_key(value.first);
    const auto bucket = _findOrAllocate(value.first, key_hash);
    const auto bempty = EMH_EMPTY(bucket);
    if (bempty) {
      EMH_NEW(std::move(value.first), std::move(value.second), bucket, key_hash);
    }

    const auto slot = m_index[bucket].slot & m_mask;
    return { { this, slot }, bempty };
  }

  template <typename K, typename V>
  std::pair<iterator, bool> _doInsert(K&& key, V&& val) noexcept
  {
    const auto key_hash = hash_key(key);
    const auto bucket = _findOrAllocate(key, key_hash);
    const auto bempty = EMH_EMPTY(bucket);
    if (bempty) {
      EMH_NEW(std::forward<K>(key), std::forward<V>(val), bucket, key_hash);
    }

    const auto slot = m_index[bucket].slot & m_mask;
    return { { this, slot }, bempty };
  }

  template <typename K, typename V>
  std::pair<iterator, bool> do_assign(K&& key, V&& val) noexcept
  {
    check_expand_need();
    const auto key_hash = hash_key(key);
    const auto bucket = _findOrAllocate(key, key_hash);
    const auto bempty = EMH_EMPTY(bucket);
    if (bempty) {
      EMH_NEW(std::forward<K>(key), std::forward<V>(val), bucket, key_hash);
    }
    else {
      m_pairs[m_index[bucket].slot & m_mask].second = std::forward(val);
    }

    const auto slot = m_index[bucket].slot & m_mask;
    return { { this, slot }, bempty };
  }

  template <typename K, typename V>
  size_type insert_unique(K&& key, V&& val)
  {
    check_expand_need();
    const auto key_hash = hash_key(key);
    auto bucket = _findUniqueBucket(key_hash);
    EMH_NEW(std::forward<K>(key), std::forward<V>(val), bucket, key_hash);
    return bucket;
  }

  size_type insert_unique(value_type&& value)
  {
    return insert_unique(std::move(value.first), std::move(value.second));
  }

  size_type insert_unique(const value_type& value)
  {
    return insert_unique(value.first, value.second);
  }

  // Can we fit another element?
  bool check_expand_need()
  {
    return reserve(m_num_filled, false);
  }

  static void prefetch_heap_block(char* ctrl)
  {
    // Prefetch the heap-allocated memory region to resolve potential TLB
    // misses.  This is intended to overlap with execution of calculating the hash for a key.
#if __linux__
    __builtin_prefetch(static_cast<const void*>(ctrl));
#elif _WIN32
    // TODO: need to fix error:
    // error C2065: '_MM_HINT_T0': undeclared identifier
    //_mm_prefetch((const char*)ctrl, _MM_HINT_T0);
#endif
  }

  size_type slot_to_bucket(const size_type slot) const noexcept
  {
    size_type main_bucket;
    return find_slot_bucket(slot, main_bucket); //TODO
  }

  //very slow
  void erase_slot(const size_type sbucket, const size_type main_bucket) noexcept
  {
    const auto slot = m_index[sbucket].slot & m_mask;
    const auto ebucket = erase_bucket(sbucket, main_bucket);
    const auto last_slot = --m_num_filled;
    if (EMH_LIKELY(slot != last_slot)) {
      const auto last_bucket = (m_etail == INACTIVE || ebucket == m_etail)
      ? slot_to_bucket(last_slot)
      : m_etail;

      m_pairs[slot] = std::move(m_pairs[last_slot]);
      m_index[last_bucket].slot = slot | (m_index[last_bucket].slot & ~m_mask);
    }

    if (is_triviall_destructable())
      m_pairs[last_slot].~value_type();

    m_etail = INACTIVE;
    m_index[ebucket] = { INACTIVE, 0 };
  }

  size_type erase_bucket(const size_type bucket, const size_type main_bucket) noexcept
  {
    const auto next_bucket = m_index[bucket].next;
    if (bucket == main_bucket) {
      if (main_bucket != next_bucket) {
        const auto nbucket = m_index[next_bucket].next;
        m_index[main_bucket] = {
          (nbucket == next_bucket) ? main_bucket : nbucket,
          m_index[next_bucket].slot
        };
      }
      return next_bucket;
    }

    const auto prev_bucket = find_prev_bucket(main_bucket, bucket);
    m_index[prev_bucket].next = (bucket == next_bucket) ? prev_bucket : next_bucket;
    return bucket;
  }

  // Find the slot with this key, or return bucket size
  size_type find_slot_bucket(const size_type slot, size_type& main_bucket) const
  {
    const auto key_hash = hash_key(m_pairs[slot].first);
    const auto bucket = main_bucket = size_type(key_hash & m_mask);
    if (slot == (m_index[bucket].slot & m_mask))
      return bucket;

    auto next_bucket = m_index[bucket].next;
    while (true) {
      if (EMH_LIKELY(slot == (m_index[next_bucket].slot & m_mask)))
        return next_bucket;
      next_bucket = m_index[next_bucket].next;
    }

    return INACTIVE;
  }

  // Find the slot with this key, or return bucket size
  size_type find_filled_bucket(const KeyT& key, uint64_t key_hash) const noexcept
  {
    const auto bucket = size_type(key_hash & m_mask);
    auto next_bucket = m_index[bucket].next;
    if (EMH_UNLIKELY((int)next_bucket < 0))
      return INACTIVE;

    const auto slot = m_index[bucket].slot & m_mask;
    //prefetch_heap_block((char*)&m_pairs[slot]);
    if (EMH_EQHASH(bucket, key_hash)) {
      if (EMH_LIKELY(m_eq(key, m_pairs[slot].first)))
        return bucket;
    }
    if (next_bucket == bucket)
      return INACTIVE;

    while (true) {
      if (EMH_EQHASH(next_bucket, key_hash)) {
        const auto slot = m_index[next_bucket].slot & m_mask;
        if (EMH_LIKELY(m_eq(key, m_pairs[slot].first)))
          return next_bucket;
      }

      const auto nbucket = m_index[next_bucket].next;
      if (nbucket == next_bucket)
        return INACTIVE;
      next_bucket = nbucket;
    }

    return INACTIVE;
  }

  // Find the slot with this key, or return bucket size
  template <typename K = KeyT>
  size_type find_filled_slot(const K& key) const noexcept
  {
    const auto key_hash = hash_key(key);
    const auto bucket = size_type(key_hash & m_mask);
    auto next_bucket = m_index[bucket].next;
    if ((int)next_bucket < 0)
      return m_num_filled;

    const auto slot = m_index[bucket].slot & m_mask;
    //prefetch_heap_block((char*)&m_pairs[slot]);
    if (EMH_EQHASH(bucket, key_hash)) {
      if (EMH_LIKELY(m_eq(key, m_pairs[slot].first)))
        return slot;
    }
    if (next_bucket == bucket)
      return m_num_filled;

    while (true) {
      if (EMH_EQHASH(next_bucket, key_hash)) {
        const auto slot = m_index[next_bucket].slot & m_mask;
        if (EMH_LIKELY(m_eq(key, m_pairs[slot].first)))
          return slot;
      }

      const auto nbucket = m_index[next_bucket].next;
      if (nbucket == next_bucket)
        return m_num_filled;
      next_bucket = nbucket;
    }

    return m_num_filled;
  }

  // kick out bucket and find empty to occupy
  // it will break the origin link and relink again.
  // before: main_bucket-->prev_bucket --> bucket   --> next_bucket
  // after : main_bucket-->prev_bucket --> (removed)--> new_bucket--> next_bucket
  size_type kickout_bucket(const size_type kmain, const size_type bucket) noexcept
  {
    const auto next_bucket = m_index[bucket].next;
    const auto new_bucket = _findEmptyBucket(next_bucket, 2);
    const auto prev_bucket = find_prev_bucket(kmain, bucket);

    const auto last = next_bucket == bucket ? new_bucket : next_bucket;
    m_index[new_bucket] = { last, m_index[bucket].slot };

    m_index[prev_bucket].next = new_bucket;
    m_index[bucket].next = INACTIVE;

    return bucket;
  }

  /*
   ** inserts a new key into a hash table; first, check whether key's main
   ** bucket/position is free. If not, check whether colliding node/bucket is in its main
   ** position or not: if it is not, move colliding bucket to an empty place and
   ** put new key in its main position; otherwise (colliding bucket is in its main
   ** position), new key goes to an empty position.
   */
  template <typename K = KeyT>
  size_type _findOrAllocate(const K& key, uint64_t key_hash) noexcept
  {
    const auto bucket = size_type(key_hash & m_mask);
    auto next_bucket = m_index[bucket].next;
    prefetch_heap_block((char*)&m_pairs[bucket]);
    if ((int)next_bucket < 0) {
      return bucket;
    }

    const auto slot = m_index[bucket].slot & m_mask;
    if (EMH_EQHASH(bucket, key_hash))
      if (EMH_LIKELY(m_eq(key, m_pairs[slot].first)))
        return bucket;

    //check current bucket_key is in main bucket or not
    const auto kmain = hash_bucket(m_pairs[slot].first);
    if (kmain != bucket)
      return kickout_bucket(kmain, bucket);
    else if (next_bucket == bucket)
      return m_index[next_bucket].next = _findEmptyBucket(next_bucket, 1);

    uint32_t csize = 1;
    //find next linked bucket and check key
    while (true) {
      const auto eslot = m_index[next_bucket].slot & m_mask;
      if (EMH_EQHASH(next_bucket, key_hash)) {
        if (EMH_LIKELY(m_eq(key, m_pairs[eslot].first)))
          return next_bucket;
      }

      csize += 1;
      const auto nbucket = m_index[next_bucket].next;
      if (nbucket == next_bucket)
        break;
      next_bucket = nbucket;
    }

    //find a empty and link it to tail
    const auto new_bucket = _findEmptyBucket(next_bucket, csize);
    prefetch_heap_block((char*)&m_pairs[new_bucket]);
    return m_index[next_bucket].next = new_bucket;
  }

  size_type _findUniqueBucket(uint64_t key_hash) noexcept
  {
    const auto bucket = size_type(key_hash & m_mask);
    auto next_bucket = m_index[bucket].next;
    if ((int)next_bucket < 0) {
      return bucket;
    }

    //check current bucket_key is in main bucket or not
    const auto kmain = hash_main(bucket);
    if (EMH_UNLIKELY(kmain != bucket))
      return kickout_bucket(kmain, bucket);
    else if (EMH_UNLIKELY(next_bucket != bucket))
      next_bucket = find_last_bucket(next_bucket);

    return m_index[next_bucket].next = _findEmptyBucket(next_bucket, 2);
  }

  /***
      Different probing techniques usually provide a trade-off between memory locality and avoidance of clustering.
      Since Robin Hood hashing is relatively resilient to clustering (both primary and secondary), linear probing is the most cache friendly alternativeis typically used.

      It's the core algorithm of this hash map with highly optimization/benchmark.
      normaly linear probing is inefficient with high load factor, it use a new 3-way linear
      probing strategy to search empty slot. from benchmark even the load factor > 0.9, it's more 2-3 timer fast than
      one-way search strategy.

      1. linear or quadratic probing a few cache line for less cache miss from input slot "bucket_from".
      2. the first  search  slot from member variant "m_last", init with 0
      3. the second search slot from calculated pos "(m_num_filled + m_last) & m_mask", it's like a rand value
      */
  // key is not in this mavalue. Find a place to put it.
  size_type _findEmptyBucket(const size_type bucket_from, uint32_t csize) noexcept
  {
    (void)csize;

    auto bucket = bucket_from;
    if (EMH_EMPTY(++bucket) || EMH_EMPTY(++bucket))
      return bucket;

#ifdef EMH_QUADRATIC
    constexpr size_type linear_probe_length = 2 * EMH_CACHE_LINE_SIZE / sizeof(Index); //16
    for (size_type offset = csize + 2, step = 4; offset <= linear_probe_length;) {
      bucket = (bucket_from + offset) & m_mask;
      if (EMH_EMPTY(bucket) || EMH_EMPTY(++bucket))
        return bucket;
      offset += step; //7/8. 12. 16
    }
#else
    constexpr size_type quadratic_probe_length = 6u;
    for (size_type offset = 4u, step = 3u; step < quadratic_probe_length;) {
      bucket = (bucket_from + offset) & m_mask;
      if (EMH_EMPTY(bucket) || EMH_EMPTY(++bucket))
        return bucket;
      offset += step++;
    }
#endif

#if EMH_PREFETCH
    __builtin_prefetch(static_cast<const void*>(_index + m_last + 1), 0, EMH_PREFETCH);
#endif

    for (;;) {
      m_last &= m_mask;
      if (EMH_EMPTY(++m_last)) // || EMH_EMPTY(++m_last))
        return m_last;

      auto medium = (m_num_buckets / 2 + m_last) & m_mask;
      if (EMH_EMPTY(medium)) // || EMH_EMPTY(++medium))
        return medium;
    }

    return 0;
  }

  size_type find_last_bucket(size_type main_bucket) const
  {
    auto next_bucket = m_index[main_bucket].next;
    if (next_bucket == main_bucket)
      return main_bucket;

    while (true) {
      const auto nbucket = m_index[next_bucket].next;
      if (nbucket == next_bucket)
        return next_bucket;
      next_bucket = nbucket;
    }
  }

  size_type find_prev_bucket(const size_type main_bucket, const size_type bucket) const
  {
    auto next_bucket = m_index[main_bucket].next;
    if (next_bucket == bucket)
      return main_bucket;

    while (true) {
      const auto nbucket = m_index[next_bucket].next;
      if (nbucket == bucket)
        return next_bucket;
      next_bucket = nbucket;
    }
  }

  size_type hash_bucket(const KeyT& key) const noexcept
  {
    return (size_type)hash_key(key) & m_mask;
  }

  size_type hash_main(const size_type bucket) const noexcept
  {
    const auto slot = m_index[bucket].slot & m_mask;
    return (size_type)hash_key(m_pairs[slot].first) & m_mask;
  }

 private:

  template <typename UType, typename std::enable_if<std::is_integral<UType>::value, uint32_t>::type = 0>
  inline uint64_t hash_key(const UType key) const
  {
    return m_hasher(key);
  }

  template <typename UType, typename std::enable_if<std::is_same<UType, std::string>::value, uint32_t>::type = 0>
  inline uint64_t hash_key(const UType& key) const
  {
    return m_hasher(key);
  }

  template <typename UType, typename std::enable_if<!std::is_integral<UType>::value && !std::is_same<UType, std::string>::value, uint32_t>::type = 0>
  inline uint64_t hash_key(const UType& key) const
  {
    return m_hasher(key);
  }

 private:

  value_type* m_pairs = nullptr;
  HashT m_hasher;
  EqT m_eq;
  Int64 m_pairs_allocated_size = 0;

 private:

  value_type* _allocBucket(size_type num_buckets)
  {
    m_pairs_allocated_size = (uint64_t)num_buckets * sizeof(value_type);
    AllocatedMemoryInfo mem_info = m_memory_allocator->allocate({}, m_pairs_allocated_size);
    return reinterpret_cast<value_type*>(mem_info.baseAddress());
  }

  void _freeBuckets()
  {
    m_memory_allocator->deallocate({}, { m_pairs, m_pairs_allocated_size });
    m_pairs = nullptr;
    m_pairs_allocated_size = 0;
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::impl

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#undef EMH_EMPTY
#undef EMH_EQHASH
#undef EMH_NEW

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
