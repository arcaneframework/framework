// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------

#include <gtest/gtest.h>

#include "arcane/utils/HashTableMap.h"
#include "arcane/utils/String.h"
#include "arcane/utils/PlatformUtils.h"
#include "arcane/utils/HashTableMap2.h"

#include <chrono>
#include <unordered_map>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

using namespace Arcane;

/*
  ThatClass& operator=(const ThatClass& from)
  Data* lookup(KeyTypeConstRef id)
  ValueType& lookupValue(KeyTypeConstRef id)
  Data* lookupAdd(KeyTypeConstRef id, const ValueType& value, bool& is_add)
  Data* lookupAdd(KeyTypeConstRef id)
  void nocheckAdd(KeyTypeConstRef id, const ValueType& value)

 */
TEST(TestHashTable, Misc)
{
  {
    HashTableMapT<Int64, String> hash1(50, true);

    ASSERT_EQ(hash1.count(), 0);
    hash1.add(25, "Test1");

    ASSERT_EQ(hash1.count(), 1);

    hash1.add(32, "Test2");
    ASSERT_EQ(hash1.count(), 2);

    hash1.add(32, "Test3");
    ASSERT_EQ(hash1.count(), 2);

    ASSERT_TRUE(hash1.hasKey(32));
    ASSERT_FALSE(hash1.hasKey(47));

    ASSERT_EQ(hash1[32], "Test3");
    ASSERT_EQ(hash1[25], "Test1");

    hash1.remove(32);
    ASSERT_FALSE(hash1.hasKey(32));
    ASSERT_EQ(hash1.count(), 1);

    hash1.add(32, "Test4");
    ASSERT_EQ(hash1.count(), 2);
    ASSERT_EQ(hash1[32], "Test4");

    hash1.clear();
    ASSERT_EQ(hash1.count(), 0);
  }
  {
    HashTableMapT<Int64, Int32> hash2(1050, true);
    int n = 1000;
    for (int i = 0; i < n; ++i)
      hash2.add((i + 1), (i + 1) * 10);
    ASSERT_EQ(hash2.count(), n);

    hash2.clear();
    ASSERT_EQ(hash2.count(), 0);
    int n2 = 2000;
    for (int i = 0; i < n2; ++i)
      hash2.add((i + 1), (i + 1) * 10);
    ASSERT_EQ(hash2.count(), n2);

    hash2.resize(3000, true);
    ASSERT_EQ(hash2.count(), n2);
    for (int i = 0; i < n2; ++i)
      ASSERT_EQ(hash2[i + 1], (i + 1) * 10);

    hash2.rehash();
    ASSERT_EQ(hash2.count(), n2);
    for (int i = 0; i < n2; ++i)
      ASSERT_EQ(hash2[i + 1], (i + 1) * 10);

    HashTableMapT<Int64, Int32> hash3(50, true);
    hash3 = hash2;
    ASSERT_EQ(hash3.count(), n2);
    for (int i = 0; i < n2; ++i)
      ASSERT_EQ(hash3[i + 1], (i + 1) * 10);

    {
      int nx = 2000000;
      HashTableMapT<Int64, Int32> hashx(nx / 2, true);
      std::cout << "Test Hash n=" << nx << "\n";
      for (int i = 0; i < nx; ++i)
        hash2.add((i + 1), (i + 1) * 5);
      std::cout << "MEM=" << platform::getMemoryUsed() << "\n";
    }
  }
}

Int64 _getRealTimeUS()
{
  auto x = std::chrono::high_resolution_clock::now();
  // Converti la valeur en microsecondes.
  auto y = std::chrono::time_point_cast<std::chrono::microseconds>(x);
  return static_cast<Int64>(y.time_since_epoch().count());
}

template <typename Key, typename Value>
class ArcaneLegacyMap
{
 public:

  using Data = typename Arcane::HashTableMapT<Key, Value>::Data;
  using value_type = std::pair<Key, Value>;

 public:

  ArcaneLegacyMap()
  : m_map(100, true)
  {}

 public:

  void insert(std::pair<Key, Value> v)
  {
    m_map.add(v.first, v.second);
  }
  void clear() { m_map.clear(); }
  const Data* end() const { return nullptr; }
  const Data* find(const Key& k) const
  {
    return m_map.lookup(k);
  }
  void erase(const Data* d)
  {
    if (d) {
      Data* dd = const_cast<Data*>(d);
      Key v = dd->key();
      m_map.remove(v);
    }
  }
  size_t size() const { return m_map.count(); }
  template <typename Lambda> void eachValue(const Lambda& v)
  {
    m_map.eachValue(v);
  }

 private:

  Arcane::HashTableMapT<Key, Value> m_map;
};

template <bool HasIter, typename HashType> void
_addMultiple(const char* name, HashType& map_instance, int nb_key)
{
  using value_type = typename HashType::value_type;
  std::cout << "ADD_MULTIPLE name=" << name << "\n";
  map_instance.clear();

  // Teste l'ajout
  Int64 t0 = _getRealTimeUS();
  for (Int32 i = 0; i < nb_key; i++) {
    Int32 value = (i + 1) * 5;
    map_instance.insert(value_type(i, value));
  }
  Int64 t1 = _getRealTimeUS();
  std::cout << "ADD_TIME=" << (t1 - t0) << "\n";

  // Teste le find
  int nb_found = 0;
  auto map_end = map_instance.end();
  for (Int32 i = (2 * nb_key - 1); i >= 0; i -= 2) {
    if (map_instance.find(i) != map_end)
      ++nb_found;
  }
  Int64 t2 = _getRealTimeUS();
  std::cout << "FIND_TIME=" << (t2 - t1) << " nb_found=" << nb_found << "\n";
  ASSERT_EQ(nb_found, nb_key / 2);
  {
    Int64 total = 0;
    if constexpr (HasIter) {
      auto iter_begin = map_instance.begin();
      auto iter_end = map_instance.end();
      for (; iter_begin != iter_end; ++iter_begin)
        total += iter_begin->second;
    }
    else {
      map_instance.eachValue([&](Int32 v) {
        total += v;
      });
    }
    Int64 t3 = _getRealTimeUS();
    Int64 i64_nb_key = nb_key;
    Int64 expected_total = 5 * ((i64_nb_key) * (i64_nb_key + 1) / 2);

    std::cout << "ITER_TIME=" << (t3 - t2) << " total=" << total << " T2=" << expected_total << "\n";
    ASSERT_EQ(total, expected_total);
  }

  // Teste la suppression
  Int64 t5 = _getRealTimeUS();
  for (Int32 i = nb_key - 1; i >= 0; i -= 2) {
    auto x = map_instance.find(i);
    if (x != map_instance.end())
      map_instance.erase(x);
  }
  Int64 t6 = _getRealTimeUS();
  std::cout << "ERASE_TIME=" << (t6 - t5) << " remaining_size=" << map_instance.size() << "\n";
  ASSERT_EQ(map_instance.size(), nb_key / 2);
  std::cout << "MEM (MB)=" << static_cast<Int64>(platform::getMemoryUsed() / 1000000.0) << "\n";
}

int num_keys = 100000;

TEST(TestHashTable, StdMap)
{
  std::unordered_map<Int64, Int32> std_map;
  _addMultiple<true>("std::unordered_map", std_map, num_keys);
}

TEST(TestHashTable, ArcaneLegacyMap)
{
  ArcaneLegacyMap<Int64, Int32> arcane_map;
  _addMultiple<false>("ArcaneLegacyMap", arcane_map, num_keys);
}

TEST(TestHashTable, ArcaneHashMap2)
{
  impl::HashTableMap2<Int64, Int32> arcane_map;
  _addMultiple<true>("ArcaneHashMap2", arcane_map, num_keys);
}

TEST(TestArcaneHashMap2, Misc)
{
  std::cout << "STRUCT_SIZE=" << sizeof(impl::HashTableMap2<Int64, Int32>) << "\n";
  impl::HashTableMap2<Int64, Int32> arcane_map;
  arcane_map.add(5, 23);
  arcane_map.add(29, 12);
  arcane_map.add(97, 3);
  ASSERT_EQ(arcane_map.size(), 3);

  impl::HashTableMap2<Int64, Int32> arcane_map2;
  ASSERT_EQ(arcane_map2.size(), 0);

  arcane_map2 = arcane_map;
  ASSERT_EQ(arcane_map2.size(), arcane_map.size());
  ASSERT_EQ(arcane_map2[5], 23);
  ASSERT_EQ(arcane_map2[29], 12);
  ASSERT_EQ(arcane_map2[97], 3);
  ASSERT_EQ(arcane_map2, arcane_map);

  impl::HashTableMap2<Int64, Int32> arcane_map3;
  arcane_map3.clone(arcane_map);
  ASSERT_EQ(arcane_map3, arcane_map);

  arcane_map3.clear();
  ASSERT_EQ(arcane_map3.size(), 0);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
