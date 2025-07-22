// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------

#include <gtest/gtest.h>

#include "arcane/utils/MemoryView.h"
#include "arcane/utils/UniqueArray.h"
#include "arcane/utils/Exception.h"
#include "arcane/utils/MemoryUtils.h"
#include "arcane/utils/NumericTypes.h"
#include "arcane/utils/internal/MemoryUtilsInternal.h"

#include <random>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

using namespace Arcane;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <typename DataType>
class MemoryTester
{
  char _initValue(Int32 v, char*)
  {
    auto x = static_cast<char>(v + 5);
    return x;
  }
  Int16 _initValue(Int32 v, Int16*)
  {
    auto x = static_cast<Int16>(v + 5);
    return x;
  }
  Int32 _initValue(Int32 v, Int32*)
  {
    auto x = static_cast<Int32>(v + 5);
    return x;
  }
  Int64 _initValue(Int32 v, Int64*)
  {
    auto x = static_cast<Int64>(v + 5);
    return x;
  }
  Real _initValue(Int32 v, Real*)
  {
    auto x = static_cast<Real>(v + 5);
    return x;
  }
  Real3 _initValue(Int32 v, Real3*)
  {
    Real x = static_cast<Real>(v + 5);
    return Real3(x, x / 2.0, x + 1.5);
  }
  Real2x2 _initValue(Int32 v, Real2x2*)
  {
    Real x = static_cast<Real>(v + 5);
    Real2 a(x, x / 2.0);
    Real2 b(x - 7.9, x * 2.0);
    return { a, b };
  }
  Real3x3 _initValue(Int32 v, Real3x3*)
  {
    Real x = static_cast<Real>(v + 5);
    Real3 a(x, x / 2.0, x + 1.5);
    Real3 b(x - 7.9, x * 2.0, x / 1.5);
    Real3 c(x + 3.2, x + 4.7, x + 2.5);
    return { a, b, c };
  }

 public:

  void apply()
  {
    Int32 nb_value = 1000;
    DataType* dummy = nullptr;
    UniqueArray<DataType> array1(nb_value);
    for (Int32 i = 0; i < nb_value; ++i) {
      DataType x = _initValue(i, dummy);
      array1[i] = x;
    }

    // Teste MutableMemoryView::copyHost()
    {
      UniqueArray<DataType> array2(nb_value);
      MutableMemoryView to(array2.span());
      ConstMemoryView from(array1.span());
      to.copyHost(from);
      ASSERT_EQ(array1, array2);
    }

    // Liste des indexs qu'on veut copier.
    // Cette liste est générée aléatoirement
    UniqueArray<Int32> copy_indexes;
    unsigned int seed0 = 942244294;
    std::mt19937 mt1(seed0);
    auto diff_2 = (mt1.max() - mt1.min()) / 2;
    for (Int32 i = 0; i < nb_value; ++i) {
      auto r = mt1() - mt1.min();
      if (r > diff_2)
        copy_indexes.add(i);
    }
    Int32 nb_index = copy_indexes.size();
    std::cout << "NB_COPY=" << nb_index << "\n";

    // Teste MutableMemoryView::copyFromIndexesHost()
    {
      // array2 contient la référence à laquelle
      // il faudra comparer l'opération de recopie
      // avec index
      UniqueArray<DataType> array2(nb_index);
      for (Int32 i = 0; i < nb_index; ++i)
        array2[i] = array1[copy_indexes[i]];

      UniqueArray<DataType> array3(nb_index);
      MutableMemoryView to(array3.span());
      ConstMemoryView from(array1.span());
      to.copyFromIndexesHost(from, copy_indexes);
      ASSERT_EQ(array2, array3);
      ConstMemoryView view2(array1.view());
      ASSERT_EQ(view2.bytes(), asBytes(array1));
      ConstMemoryView view3(array1.view(), 1);
      ASSERT_EQ(view3.bytes(), asBytes(array1));
    }

    // Teste MutableMemoryView::copyToIndexesHost()
    {
      // array2 contient la référence à laquelle
      // il faudra comparer l'opération de recopie
      // avec index
      UniqueArray<DataType> array2(nb_value);
      UniqueArray<DataType> array3(nb_value);
      for (Int32 i = 0; i < nb_value; ++i) {
        DataType x = _initValue(i + 27, dummy);
        array2[i] = x;
        array3[i] = x;
      }

      for (Int32 i = 0; i < nb_index; ++i)
        array2[copy_indexes[i]] = array1[i];

      MutableMemoryView to(array3.span());
      ConstMemoryView from(array1.span());
      MemoryUtils::copyToIndexesHost(to, from,copy_indexes);
      ASSERT_EQ(array2, array3);
    }
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

TEST(Memory, Basic)
{
  // TODO: Tester NumVector et NumMatrix
  try {
    MemoryTester<char>{}.apply();
    MemoryTester<Real>{}.apply();
    MemoryTester<Real3>{}.apply();
    MemoryTester<Int16>{}.apply();
    MemoryTester<Int32>{}.apply();
    MemoryTester<Int64>{}.apply();
    MemoryTester<Real2x2>{}.apply();
    MemoryTester<Real3x3>{}.apply();
  }
  catch (const Exception& ex) {
    std::cerr << "ERROR=" << ex << "\n";
    throw;
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
namespace
{
void _checkSetDataMemoryResource(const String& name, eMemoryResource expected_mem_resource)
{
  eMemoryResource v = MemoryUtils::getMemoryResourceFromName(name);
  ASSERT_EQ(v, expected_mem_resource);
  MemoryUtils::setDefaultDataMemoryResource(v);
  eMemoryResource v2 = MemoryUtils::getDefaultDataMemoryResource();
  ASSERT_EQ(v2, expected_mem_resource);
}
} // namespace

TEST(Memory, Allocator)
{
  _checkSetDataMemoryResource("Device", eMemoryResource::Device);
  _checkSetDataMemoryResource("HostPinned", eMemoryResource::HostPinned);
  _checkSetDataMemoryResource("Host", eMemoryResource::Host);
  _checkSetDataMemoryResource("UnifiedMemory", eMemoryResource::UnifiedMemory);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

TEST(Memory, Copy)
{
  Int32 n = 2500;
  UniqueArray<Int32> r1(n);
  for (Int32 i = 0; i < n; ++i)
    r1[i] = i + 1;

  {
    UniqueArray<Int32> r2(n);
    Span<Int32> r2_view(r2);
    Span<const Int32> r1_view(r1);
    MemoryUtils::copy(r2_view, r1_view);
    ASSERT_EQ(r1, r2);
  }

  {
    UniqueArray<Int32> r3(n);
    SmallSpan<Int32> r3_view(r3.view());
    SmallSpan<const Int32> r1_view(r1.view());
    MemoryUtils::copy(r3_view, r1_view);
    ASSERT_EQ(r1, r3);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
